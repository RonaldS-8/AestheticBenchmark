from __future__ import annotations

from pathlib import Path

import pytest
from qv_head.records import EpochLog, FileRef, TrainSummary
from qv_head.schema import DimensionMetrics
from qv_head.selection import SelectionError, select_sweep
from qv_head.training import SUMMARY_FILE
from tests.helpers import NOW

from tests.conftest import tiny_config


def _write(
    runs_dir: Path,
    run_id: str,
    lr: float,
    value: float,
    *,
    limited: bool = False,
    features_sha: str = "f" * 64,
    hidden: int = 8,
) -> None:
    metrics = DimensionMetrics(srcc=value, plcc=value, mae_1_5=0.5)
    epoch = EpochLog(
        epoch=1,
        train_loss=1.0,
        val={"aesthetic": metrics, "technical": metrics},
        val_srcc_mean=value,
    )
    config = tiny_config()
    config = config.model_copy(
        update={"head": config.head.model_copy(update={"hidden_dim": hidden})}
    )
    summary = TrainSummary(
        schema_version="1.0",
        run_id=run_id,
        created_at=NOW,
        config=config,
        seed=1,
        lr=lr,
        items_sha256="a" * 64,
        tower_sha256="b" * 64,
        features=FileRef(path="f/features.pt", sha256=features_sha),
        n_train=10,
        n_val=5,
        limited=limited,
        history=(epoch,),
        best=epoch,
        checkpoint=FileRef(path=f"{run_id}/best.pt", sha256="c" * 64),
        seconds=1.0,
    )
    (runs_dir / run_id).mkdir(parents=True)
    (runs_dir / run_id / SUMMARY_FILE).write_text(summary.model_dump_json())


def test_best_mean_lr_then_best_seed(tmp_path: Path) -> None:
    _write(tmp_path, "p-lr1e-4-s1", 1e-4, 0.70)
    _write(tmp_path, "p-lr1e-4-s2", 1e-4, 0.60)
    _write(tmp_path, "p-lr1e-3-s1", 1e-3, 0.66)
    _write(tmp_path, "p-lr1e-3-s2", 1e-3, 0.68)

    selection = select_sweep(tmp_path, "p", now=NOW)

    assert [g.lr for g in selection.groups] == [1e-4, 1e-3]
    assert selection.groups[1].mean == pytest.approx(0.67)
    assert selection.chosen_run_id == "p-lr1e-3-s2"
    assert selection.checkpoint.path == "p-lr1e-3-s2/best.pt"


def test_rejects_empty_smoke_and_mixed_runs(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match="没有"):
        select_sweep(tmp_path, "p", now=NOW)
    _write(tmp_path / "a", "p-lr1e-4-s1", 1e-4, 0.7, limited=True)
    with pytest.raises(SelectionError, match="冒烟"):
        select_sweep(tmp_path / "a", "p", now=NOW)
    _write(tmp_path / "b", "p-lr1e-4-s1", 1e-4, 0.7)
    _write(tmp_path / "b", "p-lr1e-4-s2", 1e-4, 0.7, hidden=16)
    with pytest.raises(SelectionError, match="配置"):
        select_sweep(tmp_path / "b", "p", now=NOW)
    _write(tmp_path / "c", "p-lr1e-4-s1", 1e-4, 0.7)
    _write(tmp_path / "c", "p-lr1e-4-s2", 1e-4, 0.7, features_sha="e" * 64)
    with pytest.raises(SelectionError, match="特征"):
        select_sweep(tmp_path / "c", "p", now=NOW)
