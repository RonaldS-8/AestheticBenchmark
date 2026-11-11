"""Two-stage fine-tuning of a backbone with two score heads.

Stage 1 trains the heads on a frozen backbone (in eval mode, no gradients);
stage 2 trains everything, the backbone at a lower learning rate. After every
epoch the model scores the whole val split; the checkpoint with the best mean
SRCC over the two scores is kept. Images run at their stored size, one at a
time; the loss is taken over the batch, which is exact for ConvNeXt (no
BatchNorm).
"""

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import torch
from pydantic import TypeAdapter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from aes_head.backbones import build_backbone, load_backbone, to_tensor
from aes_head.config import FinetuneConfig
from aes_head.console import echo
from aes_head.data import load_dataset, load_rgb, sha256_file
from aes_head.losses import score_loss
from aes_head.metrics import plcc, srcc
from aes_head.model import HeadModel
from aes_head.records import EpochLog, FileRef, TrainSummary
from aes_head.schema import EVA_DIMENSIONS, DimensionMetrics, HeadItem, RunName
from aes_head.weights import load_index

CHECKPOINT_FILE = "best.pt"
SUMMARY_FILE = "train-summary.json"
# One line per finished epoch, appended as training goes (tail -f to watch).
HISTORY_FILE = "history.jsonl"
CHECKPOINT_FORMAT = 1
LOG_EVERY = 20


class TrainError(ValueError):
    """Raised for a bad runtime setting, an existing run or a bad checkpoint."""


class ImageSet(TorchDataset[tuple[torch.Tensor, int]]):
    """Normalized images at native size, optionally flipped left-right."""

    def __init__(self, root: Path, items: Sequence[HeadItem], hflip: bool) -> None:
        self.root = root
        self.items = tuple(items)
        self.hflip = hflip

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pixels = to_tensor(load_rgb(self.root, self.items[index]))[0]
        if self.hflip and bool(torch.rand(()) < 0.5):
            pixels = pixels.flip(-1)
        return pixels, index


def collate(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[list[torch.Tensor], list[int]]:
    """Keep images of different sizes as a list."""

    return [pixels for pixels, _ in batch], [index for _, index in batch]


def targets(items: Sequence[HeadItem]) -> torch.Tensor:
    """``[N, 2]`` unrounded 1-5 targets (aesthetic, technical)."""

    return torch.tensor(
        [[i.summary(d).to_project_scale() for d in EVA_DIMENSIONS] for i in items]
    )


def check_runtime(device: str, precision: str) -> None:
    """bf16 autocast is only used on CUDA; CPU runs must say fp32."""

    if precision == "bf16" and not device.startswith("cuda"):
        raise TrainError("bf16 只在 CUDA 上用；CPU 上请用 --precision fp32")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise TrainError("CUDA 不可用")


def _autocast(device: str, precision: str) -> torch.autocast:
    return torch.autocast(
        device_type=device.split(":")[0],
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    )


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def predict(
    model: HeadModel,
    root: Path,
    items: Sequence[HeadItem],
    *,
    device: str,
    precision: str,
) -> tuple[torch.Tensor, list[float]]:
    """``[N, 2]`` scores on the 1-5 scale and per-image model latency in ms."""

    model.eval()
    scores: list[torch.Tensor] = []
    latency: list[float] = []
    with torch.inference_mode():
        for item in items:
            pixels = to_tensor(load_rgb(root, item))[0].to(device)
            _sync(device)
            start = time.perf_counter()
            with _autocast(device, precision):
                z = model([pixels])
            _sync(device)
            latency.append((time.perf_counter() - start) * 1000)
            scores.append(model.to_scores(z).cpu())
    return torch.cat(scores), latency


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


def save_checkpoint(path: Path, model: HeadModel, config: FinetuneConfig) -> None:
    """Write the full model state; replaces the run's previous best atomically."""

    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": config.model_dump(mode="json"),
            "state_dict": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_head_model(path: Path, sha256: str) -> tuple[HeadModel, FinetuneConfig]:
    """Rebuild a model from a checkpoint after checking its hash."""

    if not path.is_file():
        raise TrainError(f"检查点不存在: {path}")
    if sha256_file(path) != sha256:
        raise TrainError(f"检查点哈希与训练记录不符: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != CHECKPOINT_FORMAT:
        raise TrainError(f"检查点格式 {saved['format']} 不支持")
    config = FinetuneConfig.model_validate(saved["config"])
    model = HeadModel(
        config.backbone,
        build_backbone(config.backbone),
        hidden_dim=config.head.hidden_dim,
        dropout=config.head.dropout,
        target_mean=[0.0] * len(EVA_DIMENSIONS),
        target_std=[1.0] * len(EVA_DIMENSIONS),
    )
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, config


class _Trainer:
    """Shared state of one run: data, model, val scoring and the best epoch."""

    def __init__(
        self,
        config: FinetuneConfig,
        model: HeadModel,
        root: Path,
        train: Sequence[HeadItem],
        val: Sequence[HeadItem],
        run_dir: Path,
        seed: int,
        log: Callable[[str], None],
    ) -> None:
        self.config = config
        self.model = model
        self.root = root
        self.val = val
        self.run_dir = run_dir
        self.log = log
        self.device = config.runtime.device
        self.precision = config.runtime.precision
        self.targets = (targets(train) - model.target_mean.cpu()) / (
            model.target_std.cpu()
        )
        workers = config.runtime.num_workers
        self.loader = DataLoader(
            ImageSet(root, train, config.train.hflip),
            batch_size=config.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=workers > 0,
        )
        if len(self.loader) == 0:
            raise TrainError("训练集比一个 batch 还小")
        self.history: list[EpochLog] = []
        self.best: EpochLog | None = None

    def run_stage(
        self,
        stage: str,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        train_backbone: bool,
    ) -> None:
        settings = self.config.train
        total = epochs * len(self.loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine(total, int(total * settings.warmup_ratio))
        )
        params = [p for group in optimizer.param_groups for p in group["params"]]
        steps = len(self.loader)
        for epoch in range(1, epochs + 1):
            tag = f"[{stage} {epoch}/{epochs}]"
            self.model.train()
            if not train_backbone:
                self.model.extractor.model.eval()
            losses = []
            started = time.perf_counter()
            for step, (images, indices) in enumerate(self.loader, 1):
                images = [image.to(self.device, non_blocking=True) for image in images]
                target = self.targets[indices].to(self.device)
                with _autocast(self.device, self.precision):
                    z = self.model(images, train_backbone=train_backbone)
                loss = score_loss(z.float(), target, self.config.loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, settings.max_grad_norm)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                if step % LOG_EVERY == 0 or step == steps:
                    elapsed = (time.perf_counter() - started) / 60
                    recent = losses[-LOG_EVERY:]
                    self.log(
                        f"{tag} 步 {step}/{steps}  loss {sum(recent) / len(recent):.4f}"
                        f"  lr {optimizer.param_groups[0]['lr']:.1e}"
                        f"  本轮已用 {elapsed:.1f} 分，"
                        f"约剩 {elapsed / step * (steps - step):.1f} 分"
                    )
            self.log(f"{tag} 在 val {len(self.val)} 张上打分")
            self._after_epoch(stage, epoch, sum(losses) / len(losses), tag)
            minutes = (time.perf_counter() - started) / 60
            left = epochs - epoch
            # The latest epoch, not the average: the first one carries the
            # per-image-size warm-up and is many times slower.
            self.log(
                f"{tag} 本轮 {minutes:.1f} 分；"
                f"本阶段还剩 {left} 轮，约 {minutes * left:.0f} 分"
            )

    def _after_epoch(self, stage: str, epoch: int, train_loss: float, tag: str) -> None:
        scores, _ = predict(
            self.model,
            self.root,
            self.val,
            device=self.device,
            precision=self.precision,
        )
        metrics, mean = agreement(self.val, scores)
        entry = EpochLog(
            stage=stage,  # type: ignore[arg-type]
            epoch=epoch,
            train_loss=train_loss,
            val=metrics,
            val_srcc_mean=mean,
        )
        self.history.append(entry)
        with (self.run_dir / HISTORY_FILE).open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        improved = self.best is None or mean > self.best.val_srcc_mean
        if improved:
            save_checkpoint(self.run_dir / CHECKPOINT_FILE, self.model, self.config)
            self.best = entry
        assert self.best is not None
        self.log(
            f"{tag} 训练 loss {train_loss:.4f}  val SRCC 美学 "
            f"{metrics['aesthetic'].srcc:.3f} 技术 {metrics['technical'].srcc:.3f}"
            f" 均值 {mean:.3f}"
            + (
                "  ← 新的最佳，已存检查点"
                if improved
                else f"  （最佳仍是 {self.best.stage} {self.best.epoch}："
                f"{self.best.val_srcc_mean:.3f}）"
            )
        )


def _build_model(
    config: FinetuneConfig, models_dir: Path, train: Sequence[HeadItem]
) -> tuple[HeadModel, str]:
    entry = load_index(models_dir).find(config.backbone)
    backbone = load_backbone(config.backbone, models_dir / entry.path, entry.sha256)
    y = targets(train)
    model = HeadModel(
        config.backbone,
        backbone,
        hidden_dim=config.head.hidden_dim,
        dropout=config.head.dropout,
        target_mean=y.mean(dim=0).tolist(),
        target_std=y.std(dim=0).tolist(),
    )
    return model, entry.sha256


def _splits(
    data_root: Path, limit: int | None
) -> tuple[str, tuple[HeadItem, ...], tuple[HeadItem, ...]]:
    dataset = load_dataset(data_root)
    train, val = dataset.split("train"), dataset.split("val")
    if limit is not None:
        train, val = train[:limit], val[:limit]
    return dataset.info.items_sha256, train, val


def dry_run(
    config: FinetuneConfig,
    *,
    data_root: Path,
    models_dir: Path,
    runs_dir: Path,
    run_id: str,
    limit: int | None,
) -> dict[str, object]:
    """Check everything a run needs and do one loss step's forward; write nothing."""

    check_runtime(config.runtime.device, config.runtime.precision)
    if (runs_dir / run_id).exists():
        raise TrainError(f"{runs_dir / run_id} 已存在")
    _, train, val = _splits(data_root, limit)
    model, _ = _build_model(config, models_dir, train)
    model.to(config.runtime.device).train()
    images = [to_tensor(load_rgb(data_root, item))[0] for item in train[:2]]
    target = (targets(train[:2]) - model.target_mean.cpu()) / model.target_std.cpu()
    with _autocast(config.runtime.device, config.runtime.precision):
        z = model([image.to(config.runtime.device) for image in images])
    loss = score_loss(z.float(), target.to(config.runtime.device), config.loss)
    steps = len(train) // config.train.batch_size
    return {
        "n_train": len(train),
        "n_val": len(val),
        "steps_per_epoch": steps,
        "head_steps": steps * config.train.head_epochs,
        "finetune_steps": steps * config.train.finetune_epochs,
        "backbone_params_m": round(
            sum(p.numel() for p in model.backbone_parameters()) / 1e6, 2
        ),
        "head_params_m": round(
            sum(p.numel() for p in model.head_parameters()) / 1e6, 3
        ),
        "first_loss": round(loss.item(), 4),
    }


def finetune(
    config: FinetuneConfig,
    *,
    data_root: Path,
    models_dir: Path,
    runs_dir: Path,
    run_id: str,
    seed: int,
    backbone_lr: float,
    limit: int | None,
    now: datetime,
    log: Callable[[str], None] = echo,
) -> TrainSummary:
    """Train one run and write ``best.pt`` and ``train-summary.json``."""

    check_runtime(config.runtime.device, config.runtime.precision)
    # Fail now, not after hours of training when the summary is validated.
    TypeAdapter(RunName).validate_python(run_id)
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise TrainError(f"{run_dir} 已存在；确认后删除再重跑，或换 run_id")
    random.seed(seed)
    torch.manual_seed(seed)
    items_sha256, train, val = _splits(data_root, limit)
    model, pretrained_sha256 = _build_model(config, models_dir, train)
    model.to(config.runtime.device)
    started = time.perf_counter()
    trainer = _Trainer(config, model, data_root, train, val, run_dir, seed, log)
    run_dir.mkdir(parents=True)
    settings = config.train
    steps = len(trainer.loader)
    log(
        f"{run_id}: train {len(train)} 张 / val {len(val)} 张，"
        f"batch {settings.batch_size}，每轮 {steps} 步；"
        f"阶段 1 冻结骨干训头 {settings.head_epochs} 轮，"
        f"阶段 2 全部放开 {settings.finetune_epochs} 轮"
        f"（骨干 lr {backbone_lr:g}，种子 {seed}）；"
        f"设备 {config.runtime.device} {config.runtime.precision}"
    )
    if settings.head_epochs > 0:
        for parameter in model.backbone_parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.head_parameters(),
            lr=settings.head_lr,
            weight_decay=settings.weight_decay,
        )
        trainer.run_stage("head", settings.head_epochs, optimizer, False)
    for parameter in model.backbone_parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": backbone_lr},
            {
                "params": list(model.head_parameters()),
                "lr": backbone_lr * settings.head_lr_multiplier,
            },
        ],
        weight_decay=settings.weight_decay,
    )
    trainer.run_stage("full", settings.finetune_epochs, optimizer, True)
    assert trainer.best is not None
    checkpoint = run_dir / CHECKPOINT_FILE
    summary = TrainSummary(
        schema_version="1.0",
        run_id=run_id,
        created_at=now,
        config=config,
        seed=seed,
        backbone_lr=backbone_lr,
        items_sha256=items_sha256,
        pretrained_sha256=pretrained_sha256,
        n_train=len(train),
        n_val=len(val),
        limited=limit is not None,
        history=tuple(trainer.history),
        best=trainer.best,
        checkpoint=FileRef(
            path=f"{run_id}/{CHECKPOINT_FILE}", sha256=sha256_file(checkpoint)
        ),
        seconds=time.perf_counter() - started,
    )
    (run_dir / SUMMARY_FILE).write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return summary
