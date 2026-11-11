"""Ridge probe on cached tower features, by layer set, scored on val.

The same method as ``cnn_head``'s CNN probe (``docs/probe-native-v1.md``
there), so frozen-feature numbers of the two routes compare directly. Layer
sets: ``all`` (every tapped block plus the merger), ``last`` (the last tapped
block) and ``merger`` (what the language model receives).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
from pydantic import TypeAdapter

from qv_head.console import echo
from qv_head.data import load_dataset
from qv_head.features import CachedFeatures, load_features
from qv_head.probe import DEFAULT_ALPHAS, ProbeError, probe_dimension
from qv_head.records import FeatureSet
from qv_head.schema import (
    EVA_DIMENSIONS,
    LayerSet,
    ProbeReport,
    ProbeResult,
    RunName,
)

REPORT_FILE = "report.json"
LAYER_SETS: tuple[LayerSet, ...] = ("all", "last", "merger")


def layer_columns(record: FeatureSet, layers: LayerSet) -> slice:
    """Columns of the concatenated features that belong to ``layers``."""

    if layers == "all":
        return slice(0, record.feature_dim)
    ends = [sum(record.layer_dims[: k + 1]) for k in range(len(record.layer_dims))]
    k = len(record.layer_dims) - (1 if layers == "merger" else 2)
    return slice(ends[k] - record.layer_dims[k], ends[k])


def _probe(
    cached: CachedFeatures,
    *,
    folds: int,
    bootstrap: int,
    seed: int,
    log: Callable[[str], None],
) -> list[ProbeResult]:
    results = []
    for layers in LAYER_SETS:
        columns = layer_columns(cached.record, layers)
        train_x = cached.train[:, columns].double()
        val_x = cached.val[:, columns].double()
        for dim in EVA_DIMENSIONS:
            outcome = probe_dimension(
                train_x,
                [i.summary(dim).to_project_scale() for i in cached.train_items],
                val_x,
                [i.summary(dim).mean for i in cached.val_items],
                [i.summary(dim).to_reported_target() for i in cached.val_items],
                alphas=DEFAULT_ALPHAS,
                folds=folds,
                bootstrap=bootstrap,
                seed=seed,
            )
            log(
                f"[探针] {layers:<6} {dim:<9} 维度 {train_x.shape[1]:>5}  "
                f"val SRCC {outcome.val.srcc:.3f} "
                f"[{outcome.val_srcc_ci95[0]:.3f}, {outcome.val_srcc_ci95[1]:.3f}]"
            )
            results.append(
                ProbeResult(
                    layers=layers,
                    dimension=dim,  # type: ignore[arg-type]
                    feature_dim=train_x.shape[1],
                    alpha=outcome.alpha,
                    cv_srcc=outcome.cv_srcc,
                    val=outcome.val,
                    val_srcc_ci95=outcome.val_srcc_ci95,
                )
            )
    return results


def run_probe(
    *,
    data_root: Path,
    runs_dir: Path,
    features_run_id: str,
    run_id: str,
    folds: int,
    bootstrap: int,
    seed: int,
    now: datetime,
    log: Callable[[str], None] = echo,
) -> ProbeReport:
    """Probe every layer set and score; write ``runs/<run_id>/report.json``."""

    TypeAdapter(RunName).validate_python(run_id)
    out_dir = runs_dir / run_id
    if out_dir.exists():
        raise ProbeError(f"{out_dir} 已存在，换一个 run_id")
    cached = load_features(runs_dir, features_run_id, load_dataset(data_root))
    torch.manual_seed(seed)
    results = _probe(cached, folds=folds, bootstrap=bootstrap, seed=seed, log=log)
    report = ProbeReport(
        schema_version="1.0",
        run_id=run_id,
        created_at=now,
        features_run_id=features_run_id,
        features_sha256=cached.record.file.sha256,
        items_sha256=cached.record.items_sha256,
        n_train=cached.record.n_train,
        n_val=cached.record.n_val,
        limited=cached.record.limited,
        folds=folds,
        alphas=DEFAULT_ALPHAS,
        bootstrap_samples=bootstrap,
        seed=seed,
        results=tuple(results),
    )
    out_dir.mkdir(parents=True)
    (out_dir / REPORT_FILE).write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return report
