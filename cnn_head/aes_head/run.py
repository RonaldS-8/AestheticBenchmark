"""Run the frozen-feature probe for several backbones and write a report."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from aes_head.backbones import MultiLevelExtractor, get_spec, load_backbone
from aes_head.console import echo
from aes_head.data import load_dataset
from aes_head.features import (
    FeatureSet,
    extract_features,
    load_features,
    save_features,
)
from aes_head.metrics import percentile
from aes_head.probe import probe_dimension
from aes_head.schema import (
    EVA_DIMENSIONS,
    BackboneRun,
    HeadItem,
    ProbeReport,
    ProbeResult,
)
from aes_head.weights import load_index

LAYER_SETS = ("all", "last")


class RunError(ValueError):
    """Raised when a run would overwrite results or uses the wrong weights."""


def _features(
    data_root: Path,
    models_dir: Path,
    run_dir: Path,
    backbone: str,
    items: Sequence[HeadItem],
    items_sha256: str,
    device: str,
    log: Callable[[str], None],
) -> FeatureSet:
    entry = load_index(models_dir).find(backbone)
    spec = get_spec(backbone)
    if entry.url != spec.weights_url:
        raise RunError(f"{backbone} 的权重来源 {entry.url} 不是 {spec.weights_url}")
    cache = run_dir / f"features-{backbone}.pt"
    ids = [item.image_id for item in items]
    if cache.exists():
        log(f"{backbone}: 复用特征缓存 {cache.name}")
        return load_features(
            cache,
            backbone=backbone,
            weights_sha256=entry.sha256,
            items_sha256=items_sha256,
            image_ids=ids,
        )
    model = load_backbone(backbone, models_dir / entry.path, entry.sha256)
    features = extract_features(
        data_root,
        items,
        MultiLevelExtractor(spec, model),
        weights_sha256=entry.sha256,
        items_sha256=items_sha256,
        device=device,
        log=log,
    )
    save_features(cache, features)
    return features


def run_probe(
    *,
    data_root: Path,
    models_dir: Path,
    runs_dir: Path,
    run_id: str,
    backbones: Sequence[str],
    device: str,
    limit: int | None,
    folds: int,
    alphas: Sequence[float],
    bootstrap: int,
    seed: int,
    now: datetime,
    log: Callable[[str], None] = echo,
) -> ProbeReport:
    """Extract (or reuse) features, probe both scores, write ``report.json``."""

    dataset = load_dataset(data_root)
    train, val = dataset.split("train"), dataset.split("val")
    if limit is not None:
        train, val = train[:limit], val[:limit]
    run_dir = runs_dir / run_id
    report_path = run_dir / "report.json"
    if report_path.exists():
        raise RunError(f"{report_path} 已存在，换一个 run_id")
    run_dir.mkdir(parents=True, exist_ok=True)
    items = (*train, *val)
    results: list[ProbeResult] = []
    runs: list[BackboneRun] = []
    for backbone in backbones:
        features = _features(
            data_root,
            models_dir,
            run_dir,
            backbone,
            items,
            dataset.info.items_sha256,
            device,
            log,
        )
        runs.append(
            BackboneRun(
                backbone=backbone,
                weights_sha256=features.weights_sha256,
                n_images=len(items),
                latency_p50_ms=percentile(features.latency_ms, 50),
                latency_p95_ms=percentile(features.latency_ms, 95),
            )
        )
        for layers in LAYER_SETS:
            matrix = features.matrix(layers)
            train_x, val_x = matrix[: len(train)], matrix[len(train) :]
            for dimension in EVA_DIMENSIONS:
                outcome = probe_dimension(
                    train_x,
                    [i.summary(dimension).to_project_scale() for i in train],
                    val_x,
                    [i.summary(dimension).mean for i in val],
                    [i.summary(dimension).to_reported_target() for i in val],
                    alphas=alphas,
                    folds=folds,
                    bootstrap=bootstrap,
                    seed=seed,
                )
                results.append(
                    ProbeResult(
                        backbone=backbone,
                        layers=layers,
                        dimension=dimension,
                        feature_dim=matrix.shape[1],
                        alpha=outcome.alpha,
                        cv_srcc=outcome.cv_srcc,
                        val=outcome.val,
                        val_srcc_ci95=outcome.val_srcc_ci95,
                    )
                )
            log(f"{backbone}/{layers}: 完成")
    report = ProbeReport(
        schema_version="1.0",
        run_id=run_id,
        created_at=now,
        items_sha256=dataset.info.items_sha256,
        preprocess="native_rgb_imagenet_norm",
        n_train=len(train),
        n_val=len(val),
        limited=limit is not None,
        folds=folds,
        alphas=tuple(alphas),
        bootstrap_samples=bootstrap,
        seed=seed,
        backbones=tuple(runs),
        results=tuple(results),
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def format_report(report: ProbeReport) -> str:
    """A plain-text table of val SRCC/PLCC per backbone, layer set and score."""

    lines = [
        f"run {report.run_id}: train {report.n_train} / val {report.n_val}"
        + ("（--limit 冒烟，不作结论）" if report.limited else ""),
        f"{'backbone':<16}{'layers':<7}{'score':<10}"
        f"{'SRCC':>7}{'95% CI':>16}{'PLCC':>7}{'MAE':>7}{'alpha':>10}",
    ]
    for r in report.results:
        low, high = r.val_srcc_ci95
        lines.append(
            f"{r.backbone:<16}{r.layers:<7}{r.dimension:<10}"
            f"{r.val.srcc:>7.3f}{f'[{low:.3f}, {high:.3f}]':>16}"
            f"{r.val.plcc:>7.3f}{r.val.mae_1_5:>7.3f}{r.alpha:>10.0e}"
        )
    for b in report.backbones:
        lines.append(
            f"{b.backbone}: 每张 p50 {b.latency_p50_ms:.0f} ms，"
            f"p95 {b.latency_p95_ms:.0f} ms"
        )
    return "\n".join(lines)
