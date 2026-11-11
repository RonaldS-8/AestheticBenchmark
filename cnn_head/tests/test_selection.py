from __future__ import annotations

from pathlib import Path

import pytest
import select_run
from aes_head.finetune import SUMMARY_FILE
from aes_head.records import EpochLog, FileRef, TrainSummary
from aes_head.schema import DimensionMetrics
from aes_head.selection import SelectionError, select_sweep
from tests.helpers import NOW

from tests.conftest import tiny_config


def _write(
    runs_dir: Path, run_id: str, lr: float, best: float, **changes: object
) -> None:
    metric = DimensionMetrics(srcc=best, plcc=best, mae_1_5=0.2)
    epoch = EpochLog(
        stage="full",
        epoch=1,
        train_loss=1.0,
        val={"aesthetic": metric, "technical": metric},
        val_srcc_mean=best,
    )
    fields: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": NOW,
        "config": tiny_config(),
        "seed": 1,
        "backbone_lr": lr,
        "items_sha256": "a" * 64,
        "pretrained_sha256": "b" * 64,
        "n_train": 10,
        "n_val": 5,
        "limited": False,
        "history": (epoch,),
        "best": epoch,
        "checkpoint": FileRef(path=f"{run_id}/best.pt", sha256="c" * 64),
        "seconds": 1.0,
    }
    summary = TrainSummary.model_validate(fields | changes)
    (runs_dir / run_id).mkdir(parents=True)
    (runs_dir / run_id / SUMMARY_FILE).write_text(summary.model_dump_json())


def test_select_by_seed_average_then_best_seed(tmp_path: Path) -> None:
    _write(tmp_path, "ft-lr1e-5-s1", 1e-5, 0.70)
    _write(tmp_path, "ft-lr1e-5-s2", 1e-5, 0.60)
    _write(tmp_path, "ft-lr3e-5-s1", 3e-5, 0.68)
    _write(tmp_path, "ft-lr3e-5-s2", 3e-5, 0.67)
    _write(tmp_path, "other-lr1e-4-s1", 1e-4, 0.99)

    selection = select_sweep(tmp_path, "ft", now=NOW)

    assert [g.backbone_lr for g in selection.groups] == [1e-5, 3e-5]
    assert selection.groups[1].mean == pytest.approx(0.675)
    assert selection.chosen_run_id == "ft-lr3e-5-s1"
    assert selection.checkpoint.path == "ft-lr3e-5-s1/best.pt"


def test_select_rejects_empty_smoke_or_mixed_recipes(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match="没有"):
        select_sweep(tmp_path, "ft", now=NOW)
    _write(tmp_path / "a", "ft-s1", 1e-5, 0.7, limited=True)
    with pytest.raises(SelectionError, match="冒烟"):
        select_sweep(tmp_path / "a", "ft", now=NOW)
    _write(tmp_path / "b", "ft-s1", 1e-5, 0.7)
    _write(tmp_path / "b", "ft-s2", 1e-5, 0.7, config=tiny_config(batch_size=8))
    with pytest.raises(SelectionError, match="配置不同"):
        select_sweep(tmp_path / "b", "ft", now=NOW)


def test_cli_writes_selection_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "ft-lr1e-5-s1", 1e-5, 0.7)
    argv = ["--prefix", "ft", "--runs-dir", str(tmp_path)]

    assert select_run.main(argv) == 0
    assert "选中 ft-lr1e-5-s1" in capsys.readouterr().out
    assert (tmp_path / "ft-selection.json").is_file()
    with pytest.raises(SelectionError, match="不覆盖"):
        select_run.main(argv)
