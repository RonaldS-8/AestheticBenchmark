"""Train score heads on cached tower features; a learning-rate x seed sweep.

The tower is frozen, so each run trains only the heads on the cached train
features (no augmentation) and scores the cached val features after every
epoch; the epoch with the best mean val SRCC of the two scores is kept.
Runs of a sweep go one after another in one process; finished runs are
skipped and a half-written run stops the sweep before anything trains.
"""

from __future__ import annotations

import gc
import math
import random
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import torch
from pydantic import TypeAdapter

from qv_head.config import JobConfig
from qv_head.console import echo
from qv_head.data import load_dataset, sha256_file
from qv_head.features import CachedFeatures, load_features
from qv_head.head import FeatureHead, build_head, save_head
from qv_head.losses import score_loss
from qv_head.metrics import plcc, srcc
from qv_head.records import EpochLog, FileRef, TrainSummary
from qv_head.schema import EVA_DIMENSIONS, DimensionMetrics, HeadItem, RunName

CHECKPOINT_FILE = "best.pt"
SUMMARY_FILE = "train-summary.json"
HISTORY_FILE = "history.jsonl"
RUN_NAME = TypeAdapter(RunName)


class TrainError(ValueError):
    """Raised for an existing run, a bad sweep or a too-small train set."""


def targets(items: Sequence[HeadItem]) -> torch.Tensor:
    """``[N, 2]`` unrounded 1-5 targets (aesthetic, technical)."""

    return torch.tensor(
        [[i.summary(d).to_project_scale() for d in EVA_DIMENSIONS] for i in items]
    )


def agreement(
    items: Sequence[HeadItem], scores: torch.Tensor
) -> tuple[dict[str, DimensionMetrics], float]:
    """Per-score SRCC/PLCC against vote means, MAE against 1-5 targets."""

    metrics: dict[str, DimensionMetrics] = {}
    for column, dim in enumerate(EVA_DIMENSIONS):
        predicted = scores[:, column].tolist()
        means = [item.summary(dim).mean for item in items]
        reported = [item.summary(dim).to_reported_target() for item in items]
        metrics[dim] = DimensionMetrics(
            srcc=srcc(predicted, means),
            plcc=plcc(predicted, means),
            mae_1_5=sum(abs(p - t) for p, t in zip(predicted, reported, strict=True))
            / len(items),
        )
    return metrics, sum(m.srcc for m in metrics.values()) / len(metrics)


def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR factor: linear warmup, then cosine decay to zero."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _score(head: FeatureHead, features: torch.Tensor) -> torch.Tensor:
    head.eval()
    with torch.inference_mode():
        return head.to_scores(head(features)).cpu()


def train_run(
    config: JobConfig,
    cached: CachedFeatures,
    *,
    runs_dir: Path,
    run_id: str,
    seed: int,
    lr: float,
    now: datetime,
    log: Callable[[str], None] = echo,
) -> TrainSummary:
    """Train one head and write ``best.pt``, ``history.jsonl`` and the summary."""

    RUN_NAME.validate_python(run_id)
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise TrainError(f"{run_dir} 已存在；确认后删除再重跑，或换 run_id")
    settings = config.train
    device = config.runtime.device
    steps = cached.train.shape[0] // settings.batch_size
    if steps == 0:
        raise TrainError("训练集比一个 batch 还小")
    random.seed(seed)
    torch.manual_seed(seed)
    y = targets(cached.train_items)
    head = build_head(config, cached.train, y).to(device)
    z_target = ((y - y.mean(dim=0)) / y.std(dim=0)).to(device)
    train_x = cached.train.to(device)
    val_x = cached.val.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, weight_decay=settings.weight_decay
    )
    total = steps * settings.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total, int(total * settings.warmup_ratio))
    )
    generator = torch.Generator().manual_seed(seed)
    run_dir.mkdir(parents=True)
    log(
        f"{run_id}: train {len(cached.train_items)} / val {len(cached.val_items)}，"
        f"特征 {train_x.shape[1]} 维，batch {settings.batch_size}，"
        f"{settings.epochs} 轮 × {steps} 步，lr {lr:g}，种子 {seed}，设备 {device}"
    )
    started = time.perf_counter()
    history: list[EpochLog] = []
    best: EpochLog | None = None
    for epoch in range(1, settings.epochs + 1):
        head.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        losses = []
        for batch in order[: steps * settings.batch_size].split(settings.batch_size):
            batch = batch.to(device)
            loss = score_loss(head(train_x[batch]), z_target[batch], config.loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), settings.max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        metrics, mean = agreement(cached.val_items, _score(head, val_x))
        entry = EpochLog(
            epoch=epoch,
            train_loss=sum(losses) / len(losses),
            val=metrics,
            val_srcc_mean=mean,
        )
        history.append(entry)
        with (run_dir / HISTORY_FILE).open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        improved = best is None or mean > best.val_srcc_mean
        if improved:
            save_head(
                run_dir / CHECKPOINT_FILE, head, config, cached.record.tower_sha256
            )
            best = entry
        assert best is not None
        log(
            f"[{epoch}/{settings.epochs}] loss {entry.train_loss:.4f}  val SRCC "
            f"美学 {metrics['aesthetic'].srcc:.3f} 技术 {metrics['technical'].srcc:.3f}"
            f" 均值 {mean:.3f}"
            + ("  ← 新的最佳" if improved else f"  （最佳 {best.epoch}）")
        )
    assert best is not None
    checkpoint = run_dir / CHECKPOINT_FILE
    summary = TrainSummary(
        schema_version="1.0",
        run_id=run_id,
        created_at=now,
        config=config,
        seed=seed,
        lr=lr,
        items_sha256=cached.record.items_sha256,
        tower_sha256=cached.record.tower_sha256,
        features=cached.record.file,
        n_train=len(cached.train_items),
        n_val=len(cached.val_items),
        limited=cached.record.limited,
        history=tuple(history),
        best=best,
        checkpoint=FileRef(
            path=f"{run_id}/{CHECKPOINT_FILE}", sha256=sha256_file(checkpoint)
        ),
        seconds=time.perf_counter() - started,
    )
    (run_dir / SUMMARY_FILE).write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return summary


def plan_runs(
    prefix: str, lrs: Sequence[str], seeds: Sequence[int]
) -> list[tuple[str, float, int]]:
    """``(run_id, lr, seed)`` for every pair; the lr text is kept in the id."""

    if len(set(lrs)) != len(lrs) or len(set(seeds)) != len(seeds):
        raise TrainError("学习率或种子有重复")
    return [
        (RUN_NAME.validate_python(f"{prefix}-lr{lr}-s{seed}"), float(lr), seed)
        for lr in lrs
        for seed in seeds
    ]


def run_sweep(
    config: JobConfig,
    *,
    data_root: Path,
    runs_dir: Path,
    features_run_id: str,
    prefix: str,
    lrs: Sequence[str],
    seeds: Sequence[int],
    clock: Callable[[], datetime],
    log: Callable[[str], None] = echo,
) -> list[TrainSummary]:
    """Train every unfinished run of the sweep, in order, in this process."""

    plan = plan_runs(prefix, lrs, seeds)
    for run_id, _, _ in plan:
        if (runs_dir / run_id).exists() and not (
            runs_dir / run_id / SUMMARY_FILE
        ).is_file():
            raise TrainError(
                f"{runs_dir / run_id} 没跑完；"
                f"检查后改名为 aborted-{run_id} 或删除再重跑"
            )
    cached = load_features(runs_dir, features_run_id, load_dataset(data_root))
    if cached.record.tower != config.tower:
        raise TrainError("特征缓存的 [tower] 设置与配置不同")
    summaries = []
    for k, (run_id, lr, seed) in enumerate(plan, 1):
        tag = f"[{k}/{len(plan)}]"
        path = runs_dir / run_id / SUMMARY_FILE
        if path.is_file():
            log(f"{tag} 跳过已完成的 {run_id}")
            summaries.append(
                TrainSummary.model_validate_json(path.read_text(encoding="utf-8"))
            )
            continue
        log(f"{tag} {clock():%Y-%m-%d %H:%M:%S} 开始 {run_id}")
        summary = train_run(
            config,
            cached,
            runs_dir=runs_dir,
            run_id=run_id,
            seed=seed,
            lr=lr,
            now=clock(),
            log=log,
        )
        log(
            f"{tag} 完成 {run_id}：最佳第 {summary.best.epoch} 轮，"
            f"val 均值 {summary.best.val_srcc_mean:.3f}，"
            f"用时 {summary.seconds / 60:.1f} 分"
        )
        summaries.append(summary)
        gc.collect()
    return summaries
