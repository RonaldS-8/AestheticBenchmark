"""Score a trained run's checkpoint on one split and write a LoRA-style report.

Writes ``runs/<run_id>/config.json``, ``predictions.jsonl`` and
``report.json``. The report is computed exactly as the main repository's
``build_eva_report``: SRCC/PLCC against raw vote means, MAE against the
rounded 1-5 targets, every prediction counted as parsed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from aes_head.console import echo
from aes_head.data import ids_fingerprint, load_dataset
from aes_head.finetune import (
    SUMMARY_FILE,
    TrainError,
    check_runtime,
    load_head_model,
    predict,
)
from aes_head.metrics import MetricsError, percentile, plcc, srcc
from aes_head.records import (
    EvaEvalReport,
    EvalSplit,
    HeadEvalRunConfig,
    HeadPrediction,
    TrainSummary,
)
from aes_head.schema import EVA_DIMENSIONS, DimensionMetrics, HeadItem


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


def evaluate_run(
    *,
    data_root: Path,
    runs_dir: Path,
    train_run_id: str,
    split: EvalSplit,
    run_id: str,
    device: str,
    precision: str,
    now: datetime,
    log: Callable[[str], None] = echo,
) -> EvaEvalReport:
    """Score ``train_run_id``'s best checkpoint on ``split``; never overwrites."""

    if split == "all":
        raise TrainError("美学头只在 val 或 eval 上评测")
    check_runtime(device, precision)
    out_dir = runs_dir / run_id
    if out_dir.exists():
        raise TrainError(f"{out_dir} 已存在，换一个 run_id")
    summary_path = runs_dir / train_run_id / SUMMARY_FILE
    if not summary_path.is_file():
        raise TrainError(f"找不到训练记录 {summary_path}")
    summary = TrainSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    if summary.items_sha256 != dataset.info.items_sha256:
        raise TrainError("检查点训练时用的数据与当前 data/ 不同")
    model, _ = load_head_model(
        runs_dir / summary.checkpoint.path, summary.checkpoint.sha256
    )
    model.to(device)
    items = dataset.split(split)
    config = HeadEvalRunConfig(
        schema_version="1.0",
        run_id=run_id,
        split=split,
        ids_fingerprint=ids_fingerprint(item.image_id for item in items),
        train_run_id=train_run_id,
        checkpoint=summary.checkpoint,
        device=device,
        precision=precision,  # type: ignore[arg-type]
    )
    log(f"{train_run_id}: 在 {split} 上评测 {len(items)} 张")
    scores, latency = predict(
        model, data_root, items, device=device, precision=precision
    )
    predictions = [
        HeadPrediction(
            run_id=run_id,
            image_id=item.image_id,
            aesthetic=float(row[0]),
            technical=float(row[1]),
            latency_ms=ms,
        )
        for item, row, ms in zip(items, scores.tolist(), latency, strict=True)
    ]
    report = build_report(config, predictions, items, now=now)
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(
        config.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "predictions.jsonl").write_text(
        "".join(p.model_dump_json() + "\n" for p in predictions), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return report
