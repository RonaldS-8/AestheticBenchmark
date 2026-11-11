"""Pick the best run of a learning-rate x seed sweep on val only.

Runs are grouped by learning rate; the rate with the highest seed-averaged
best val SRCC (mean of the two scores) wins, and within it the seed with the
highest value supplies the checkpoint. Runs must share one recipe and one
feature cache, and none may be a ``--limit`` smoke run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from qv_head.records import SweepGroup, SweepSelection, TrainSummary
from qv_head.training import SUMMARY_FILE


class SelectionError(ValueError):
    """Raised when the sweep is empty or its runs are not comparable."""


def select_sweep(runs_dir: Path, prefix: str, *, now: datetime) -> SweepSelection:
    """Read every ``<prefix>-lr*/train-summary.json`` and choose a run."""

    paths = sorted(runs_dir.glob(f"{prefix}-lr*/{SUMMARY_FILE}"))
    summaries = [
        TrainSummary.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    if not summaries:
        raise SelectionError(f"{runs_dir} 下没有 {prefix}-lr* 的训练记录")
    if any(s.limited for s in summaries):
        raise SelectionError("冒烟（--limit）运行不能参与选择")
    if len({s.config for s in summaries}) != 1:
        raise SelectionError("这些运行的训练配置不同，不能放在一起比")
    if len({s.features for s in summaries}) != 1:
        raise SelectionError("这些运行用的特征缓存不同，不能放在一起比")
    by_lr: dict[float, list[TrainSummary]] = {}
    for summary in summaries:
        by_lr.setdefault(summary.lr, []).append(summary)
    groups = []
    for lr in sorted(by_lr):
        runs = by_lr[lr]
        values = tuple(r.best.val_srcc_mean for r in runs)
        groups.append(
            SweepGroup(
                lr=lr,
                run_ids=tuple(r.run_id for r in runs),
                best_val_srcc_mean=values,
                mean=sum(values) / len(values),
            )
        )
    winner = max(groups, key=lambda g: g.mean)
    chosen = max(by_lr[winner.lr], key=lambda r: r.best.val_srcc_mean)
    return SweepSelection(
        schema_version="1.0",
        prefix=prefix,
        created_at=now,
        recipe=summaries[0].config.recipe,
        groups=tuple(groups),
        chosen_run_id=chosen.run_id,
        checkpoint=chosen.checkpoint,
    )
