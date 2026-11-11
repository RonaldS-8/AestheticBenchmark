"""Score a trained head end to end (image -> tower -> head) on one split.

Writes ``runs/<run_id>/config.json``, ``predictions.jsonl`` and
``report.json``. The report is computed exactly as the main repository's
``build_eva_report``: SRCC/PLCC against raw vote means, MAE against the
rounded 1-5 targets, every prediction counted as parsed. Latency covers the
tower and the head on the device (decoding and patchifying excluded).

Scoring val end to end must reproduce the val numbers the head was selected
on from cached features; a gap beyond ``VAL_TOLERANCE`` means the two paths
differ and is an error, so run val before eval.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import torch

from qv_head.console import echo
from qv_head.data import ids_fingerprint, load_dataset, load_rgb
from qv_head.head import load_head
from qv_head.metrics import MetricsError, percentile, plcc, srcc
from qv_head.records import (
    EvaEvalReport,
    EvalSplit,
    HeadEvalRunConfig,
    HeadPrediction,
    TrainSummary,
)
from qv_head.schema import EVA_DIMENSIONS, DimensionMetrics, HeadItem
from qv_head.tower import load_tower, sync
from qv_head.training import SUMMARY_FILE, TrainError

VAL_TOLERANCE = 0.01


def build_report(
    config: HeadEvalRunConfig,
    predictions: Sequence[HeadPrediction],
    items: Sequence[HeadItem],
    *,
    now: datetime,
) -> EvaEvalReport:
    """Aggregate predictions for every item of the split into a report."""

    by_id = {item.image_id: item for item in items}
    predicted = {p.image_id: p for p in predictions}
    if len(predicted) != len(predictions) or predicted.keys() != by_id.keys():
        raise TrainError("预测与该划分的图片不一一对应")
    metrics: dict[str, DimensionMetrics] = {}
    metric_errors: dict[str, str] = {}
    for dim in EVA_DIMENSIONS:
        pred = [getattr(p, dim) for p in predictions]
        means = [by_id[p.image_id].summary(dim).mean for p in predictions]
        targets = [
            by_id[p.image_id].summary(dim).to_reported_target() for p in predictions
        ]
        try:
            metrics[dim] = DimensionMetrics(
                srcc=srcc(pred, means),
                plcc=plcc(pred, means),
                mae_1_5=sum(abs(a - b) for a, b in zip(pred, targets, strict=True))
                / len(predictions),
            )
        except MetricsError as exc:
            metric_errors[dim] = str(exc)
    latencies = [p.latency_ms for p in predictions]
    return EvaEvalReport(
        schema_version="1.0",
        run_id=config.run_id,
        split=config.split,
        ids_fingerprint=config.ids_fingerprint,
        created_at=now,
        n_items=len(predictions),
        n_parsed=len(predictions),
        parse_success_rate=1.0,
        metrics=metrics,
        metric_errors=metric_errors,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )


def _check_val(summary: TrainSummary, report: EvaEvalReport) -> None:
    for dim in EVA_DIMENSIONS:
        cached = summary.best.val[dim].srcc
        direct = report.metrics[dim].srcc
        if abs(cached - direct) > VAL_TOLERANCE:
            raise TrainError(
                f"{dim}：端到端 val SRCC {direct:.4f} 与缓存特征上的 {cached:.4f} "
                f"相差超过 {VAL_TOLERANCE}，两条路径不一致"
            )


def evaluate_run(
    *,
    data_root: Path,
    models_dir: Path,
    runs_dir: Path,
    train_run_id: str,
    split: EvalSplit,
    run_id: str,
    now: datetime,
    log: Callable[[str], None] = echo,
) -> EvaEvalReport:
    """Score ``train_run_id``'s best head on ``split``; never overwrites."""

    if split == "all":
        raise TrainError("只在 val 或 eval 上评测")
    out_dir = runs_dir / run_id
    if out_dir.exists():
        raise TrainError(f"{out_dir} 已存在，换一个 run_id")
    summary_path = runs_dir / train_run_id / SUMMARY_FILE
    if not summary_path.is_file():
        raise TrainError(f"找不到训练记录 {summary_path}")
    summary = TrainSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    if summary.limited:
        raise TrainError("冒烟（--limit）运行不能评测")
    dataset = load_dataset(data_root)
    if summary.items_sha256 != dataset.info.items_sha256:
        raise TrainError("检查点训练时用的数据与当前 data/ 不同")
    head, config, head_tower = load_head(
        runs_dir / summary.checkpoint.path, summary.checkpoint.sha256
    )
    device = config.runtime.device
    tower, tower_sha256 = load_tower(models_dir, config.tower, device)
    if tower_sha256 != head_tower or tower_sha256 != summary.tower_sha256:
        raise TrainError("视觉塔与训练时用的不是同一份权重")
    head.to(device).eval()
    items = dataset.split(split)
    run_config = HeadEvalRunConfig(
        schema_version="1.0",
        run_id=run_id,
        split=split,
        ids_fingerprint=ids_fingerprint(item.image_id for item in items),
        train_run_id=train_run_id,
        checkpoint=summary.checkpoint,
        tower_sha256=tower_sha256,
        device=device,
        tower_precision=config.tower.precision,
    )
    log(f"{train_run_id}: 在 {split} 上端到端评测 {len(items)} 张")
    predictions = []
    for item in items:
        encoded = tower.encode(load_rgb(data_root, item))
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            features = tower(encoded).unsqueeze(0)
            scores = head.to_scores(head(features))[0].cpu()
        sync(device)
        predictions.append(
            HeadPrediction(
                run_id=run_id,
                image_id=item.image_id,
                aesthetic=float(scores[0]),
                technical=float(scores[1]),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
    report = build_report(run_config, predictions, items, now=now)
    if split == "val":
        _check_val(summary, report)
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(
        run_config.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "predictions.jsonl").write_text(
        "".join(p.model_dump_json() + "\n" for p in predictions), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return report
