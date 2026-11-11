"""Run a learning-rate x seed sweep in one process.

One process matters for speed: cuDNN prepares its convolution plans once per
input size and keeps them for the life of the process, and EVA has over a
thousand distinct image sizes. In separate processes every run pays that
warm-up again (about 12 minutes for the first epoch, against 0.4 minutes once
warm); here only the first run does. Each run still seeds and builds
everything afresh, so results equal separate runs.

Runs that already have a train summary are skipped. A planned run whose
directory exists without one (an interrupted run) stops the sweep before
anything is trained.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TextIO

import torch
from pydantic import TypeAdapter

from aes_head.config import FinetuneConfig
from aes_head.console import echo
from aes_head.finetune import SUMMARY_FILE, finetune
from aes_head.records import TrainSummary
from aes_head.schema import RunName

RUN_NAME: TypeAdapter[str] = TypeAdapter(RunName)


class SweepError(ValueError):
    """Raised for duplicate settings or an interrupted run in the plan."""


def _tee(log: Callable[[str], None], handle: TextIO) -> Callable[[str], None]:
    """Send each message to ``log`` and append it to a run's own log file."""

    def write(message: str) -> None:
        log(message)
        handle.write(message + "\n")
        handle.flush()

    return write


def plan_runs(
    prefix: str, lrs: Sequence[str], seeds: Sequence[int]
) -> list[tuple[str, float, int]]:
    """``(run_id, backbone_lr, seed)`` in sweep order; ids as run.sh names them.

    The learning rate keeps the text it was given (``3e-5``), so ids match
    runs started earlier by name.
    """

    if len(set(lrs)) != len(lrs) or len(set(seeds)) != len(seeds):
        raise SweepError("学习率或种子有重复")
    plan = []
    for lr in lrs:
        for seed in seeds:
            run_id = RUN_NAME.validate_python(f"{prefix}-lr{lr}-s{seed}")
            plan.append((run_id, float(lr), seed))
    return plan


def run_sweep(
    config: FinetuneConfig,
    *,
    prefix: str,
    lrs: Sequence[str],
    seeds: Sequence[int],
    data_root: Path,
    models_dir: Path,
    runs_dir: Path,
    logs_dir: Path,
    clock: Callable[[], datetime],
    log: Callable[[str], None] = echo,
) -> list[TrainSummary]:
    """Train every planned run not finished yet; return all their summaries."""

    plan = plan_runs(prefix, lrs, seeds)
    for run_id, _, _ in plan:
        run_dir = runs_dir / run_id
        if run_dir.exists() and not (run_dir / SUMMARY_FILE).is_file():
            raise SweepError(
                f"{run_dir} 没跑完；确认后改名留存（aborted-…）或删除，再启动"
            )
    logs_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[TrainSummary] = []
    for index, (run_id, lr, seed) in enumerate(plan, 1):
        tag = f"[{index}/{len(plan)}]"
        summary_path = runs_dir / run_id / SUMMARY_FILE
        if summary_path.is_file():
            log(f"{tag} 跳过已完成的 {run_id}")
            text = summary_path.read_text(encoding="utf-8")
            summaries.append(TrainSummary.model_validate_json(text))
            continue
        log(f"{tag} {clock():%Y-%m-%d %H:%M:%S} 开始 {run_id}")
        with (logs_dir / f"{run_id}.log").open("a", encoding="utf-8") as handle:
            summary = finetune(
                config,
                data_root=data_root,
                models_dir=models_dir,
                runs_dir=runs_dir,
                run_id=run_id,
                seed=seed,
                backbone_lr=lr,
                limit=None,
                now=clock(),
                log=_tee(log, handle),
            )
        summaries.append(summary)
        log(
            f"{tag} {clock():%Y-%m-%d %H:%M:%S} 完成 {run_id}：最佳 "
            f"{summary.best.stage} {summary.best.epoch}，val 均值 "
            f"{summary.best.val_srcc_mean:.3f}，用时 {summary.seconds / 60:.0f} 分"
        )
        # Free this run's tensors; cuDNN plans live outside the allocator.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries
