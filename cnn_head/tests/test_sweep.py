from __future__ import annotations

from pathlib import Path

import pytest
import sweep
from aes_head.finetune import SUMMARY_FILE
from aes_head.sweep import SweepError, plan_runs, run_sweep
from pydantic import ValidationError
from tests.helpers import NOW

from tests.conftest import TinyJob


def _sweep(job: TinyJob, lrs: list[str], seeds: list[int], **kwargs: object) -> list:
    options: dict[str, object] = {
        "prefix": "tiny",
        "lrs": lrs,
        "seeds": seeds,
        "data_root": job.data_root,
        "models_dir": job.models_dir,
        "runs_dir": job.runs_dir,
        "logs_dir": job.runs_dir.parent / "logs",
        "clock": lambda: NOW,
        "log": lambda _message: None,
    }
    return run_sweep(job.config, **{**options, **kwargs})  # type: ignore[arg-type]


def test_plan_keeps_lr_text_in_run_ids() -> None:
    plan = plan_runs("convnext-t-ft", ["1e-5", "3e-5"], [1, 2])

    assert [run_id for run_id, _, _ in plan] == [
        "convnext-t-ft-lr1e-5-s1",
        "convnext-t-ft-lr1e-5-s2",
        "convnext-t-ft-lr3e-5-s1",
        "convnext-t-ft-lr3e-5-s2",
    ]
    assert plan[2][1:] == (3e-5, 1)


def test_plan_rejects_duplicates_and_bad_ids() -> None:
    with pytest.raises(SweepError, match="重复"):
        plan_runs("p", ["1e-5", "1e-5"], [1])
    with pytest.raises(ValidationError):
        plan_runs("p", ["0.001"], [1])


def test_sweep_trains_in_order_and_logs_each_run(tiny_job: TinyJob) -> None:
    messages: list[str] = []

    summaries = _sweep(tiny_job, ["1e-3", "3e-3"], [1], log=messages.append)

    assert [s.run_id for s in summaries] == ["tiny-lr1e-3-s1", "tiny-lr3e-3-s1"]
    assert [s.backbone_lr for s in summaries] == [1e-3, 3e-3]
    assert any(m.startswith("[1/2]") and "开始 tiny-lr1e-3-s1" in m for m in messages)
    assert any(m.startswith("[2/2]") and "完成 tiny-lr3e-3-s1" in m for m in messages)
    log = tiny_job.runs_dir.parent / "logs" / "tiny-lr3e-3-s1.log"
    assert "[full 1/1]" in log.read_text(encoding="utf-8")


def test_sweep_skips_finished_runs(tiny_job: TinyJob) -> None:
    first = _sweep(tiny_job, ["1e-3"], [1])
    messages: list[str] = []

    again = _sweep(tiny_job, ["1e-3"], [1, 2], log=messages.append)

    assert again[0] == first[0]
    assert any("跳过已完成的 tiny-lr1e-3-s1" in m for m in messages)
    assert (tiny_job.runs_dir / "tiny-lr1e-3-s2" / SUMMARY_FILE).is_file()


def test_sweep_stops_before_training_on_an_interrupted_run(tiny_job: TinyJob) -> None:
    (tiny_job.runs_dir / "tiny-lr1e-3-s2").mkdir(parents=True)

    with pytest.raises(SweepError, match="没跑完"):
        _sweep(tiny_job, ["1e-3"], [1, 2])
    assert not (tiny_job.runs_dir / "tiny-lr1e-3-s1").exists()


def test_cli_runs_the_sweep(
    tiny_job: TinyJob, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "tiny.toml"
    config.write_text(
        "\n".join(
            [
                'schema_version = "1.0"',
                'recipe = "tiny-ft"',
                'backbone = "tiny_cnn"',
                "[head]\nhidden_dim = 8\ndropout = 0.0",
                "[loss]\nmse_weight = 1.0\nplcc_weight = 1.0\nrank_weight = 0.3",
                "[train]\nbatch_size = 4\nhead_epochs = 1\nhead_lr = 0.01",
                "finetune_epochs = 1\nhead_lr_multiplier = 10.0\nwarmup_ratio = 0.1",
                "weight_decay = 0.0\nmax_grad_norm = 1.0\nhflip = false",
                '[runtime]\ndevice = "cpu"\nprecision = "fp32"\nnum_workers = 0',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    argv = ["--config", str(config), "--prefix", "tiny", "--lrs", "1e-3"]
    argv += ["--seeds", "1", "--data-root", str(tiny_job.data_root)]
    argv += ["--models-dir", str(tiny_job.models_dir)]
    argv += ["--runs-dir", str(tiny_job.runs_dir)]
    argv += ["--logs-dir", str(tmp_path / "logs")]

    assert sweep.main(argv) == 0
    assert "sweep 完成：共 1 组" in capsys.readouterr().out
