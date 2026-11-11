from __future__ import annotations

import json

import pytest
import torch
import train
from aes_head import finetune as finetune_module
from aes_head.console import echo
from aes_head.finetune import (
    CHECKPOINT_FILE,
    HISTORY_FILE,
    SUMMARY_FILE,
    TrainError,
    check_runtime,
    dry_run,
    finetune,
    load_head_model,
    warmup_cosine,
)
from aes_head.records import EpochLog, TrainSummary
from pydantic import ValidationError
from tests.helpers import NOW

from tests.conftest import TinyJob, tiny_config


def _train(job: TinyJob, run_id: str = "tiny-lr1e-3-s1", **kwargs: object) -> object:
    options: dict[str, object] = {
        "data_root": job.data_root,
        "models_dir": job.models_dir,
        "runs_dir": job.runs_dir,
        "run_id": run_id,
        "seed": 1,
        "backbone_lr": 1e-3,
        "limit": None,
        "now": NOW,
        "log": lambda _message: None,
    }
    return finetune(job.config, **{**options, **kwargs})  # type: ignore[arg-type]


def test_finetune_runs_both_stages_and_keeps_best(tiny_job: TinyJob) -> None:
    summary = _train(tiny_job)

    assert isinstance(summary, TrainSummary)
    assert [(e.stage, e.epoch) for e in summary.history] == [("head", 1), ("full", 1)]
    assert summary.best.val_srcc_mean == max(e.val_srcc_mean for e in summary.history)
    assert summary.n_train == 12 and summary.n_val == 6
    run_dir = tiny_job.runs_dir / "tiny-lr1e-3-s1"
    saved = TrainSummary.model_validate_json((run_dir / SUMMARY_FILE).read_text())
    assert saved == summary
    model, config = load_head_model(
        run_dir / CHECKPOINT_FILE, summary.checkpoint.sha256
    )
    assert config == tiny_job.config
    assert model.target_mean.shape == (2,)


def test_finetune_reports_progress_and_writes_history_live(
    tiny_job: TinyJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finetune_module, "LOG_EVERY", 2)
    job = TinyJob(
        tiny_job.data_root,
        tiny_job.models_dir,
        tiny_job.runs_dir,
        tiny_config(head_epochs=2, finetune_epochs=2),
    )
    messages: list[str] = []

    summary = _train(job, log=messages.append)

    text = "\n".join(messages)
    assert messages[0].startswith("tiny-lr1e-3-s1: train 12 张 / val 6 张")
    # Three steps per epoch: progress at step 2 and at the last step.
    assert "[head 1/2] 步 2/3" in text and "[full 2/2] 步 3/3" in text
    assert "[full 2/2] 在 val 6 张上打分" in text
    assert "新的最佳" in text
    assert "本阶段还剩 0 轮" in text
    lines = (job.runs_dir / "tiny-lr1e-3-s1" / HISTORY_FILE).read_text().splitlines()
    assert [EpochLog.model_validate_json(line) for line in lines] == list(
        summary.history  # type: ignore[attr-defined]
    )


def test_finetune_is_reproducible_with_the_same_seed(tiny_job: TinyJob) -> None:
    first = _train(tiny_job, "tiny-a")
    second = _train(tiny_job, "tiny-b")

    assert [e.train_loss for e in first.history] == pytest.approx(  # type: ignore[attr-defined]
        [e.train_loss for e in second.history]  # type: ignore[attr-defined]
    )


def test_finetune_without_head_stage(tiny_job: TinyJob) -> None:
    job = TinyJob(
        tiny_job.data_root,
        tiny_job.models_dir,
        tiny_job.runs_dir,
        tiny_config(head_epochs=0),
    )

    summary = _train(job)

    assert [e.stage for e in summary.history] == ["full"]  # type: ignore[attr-defined]


def test_finetune_refuses_existing_run_and_tiny_train_set(tiny_job: TinyJob) -> None:
    _train(tiny_job)

    with pytest.raises(TrainError, match="已存在"):
        _train(tiny_job)
    with pytest.raises(TrainError, match="batch"):
        _train(tiny_job, "tiny-small", limit=3)


def test_finetune_rejects_bad_run_id_before_training(tiny_job: TinyJob) -> None:
    with pytest.raises(ValidationError):
        _train(tiny_job, "Tiny_Run")

    assert not (tiny_job.runs_dir / "Tiny_Run").exists()


def test_check_runtime_rejects_bf16_on_cpu() -> None:
    with pytest.raises(TrainError, match="fp32"):
        check_runtime("cpu", "bf16")
    if not torch.cuda.is_available():
        with pytest.raises(TrainError, match="CUDA"):
            check_runtime("cuda", "fp32")


def test_load_head_model_checks_file(tiny_job: TinyJob) -> None:
    summary = _train(tiny_job)
    path = tiny_job.runs_dir / summary.checkpoint.path  # type: ignore[attr-defined]

    with pytest.raises(TrainError, match="哈希"):
        load_head_model(path, "0" * 64)
    with pytest.raises(TrainError, match="不存在"):
        load_head_model(path.with_name("missing.pt"), "0" * 64)


def test_echo_flushes_each_line(capsys: pytest.CaptureFixture[str]) -> None:
    echo("进度")

    assert capsys.readouterr().out == "进度\n"


def test_warmup_then_cosine_to_zero() -> None:
    factor = warmup_cosine(total_steps=10, warmup_steps=2)

    assert [factor(0), factor(1)] == [0.5, 1.0]
    assert factor(2) == pytest.approx(1.0)
    assert factor(10) == pytest.approx(0.0)


def test_dry_run_writes_nothing(tiny_job: TinyJob) -> None:
    plan = dry_run(
        tiny_job.config,
        data_root=tiny_job.data_root,
        models_dir=tiny_job.models_dir,
        runs_dir=tiny_job.runs_dir,
        run_id="tiny-dry",
        limit=None,
    )

    assert plan["steps_per_epoch"] == 3 and plan["n_val"] == 6
    assert not tiny_job.runs_dir.exists()


def test_cli_dry_run_and_smoke(
    tiny_job: TinyJob, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tiny_job.data_root.parent / "tiny.toml"
    lines = [
        'schema_version = "1.0"',
        'recipe = "tiny-ft"',
        'backbone = "tiny_cnn"',
        "[head]\nhidden_dim = 8\ndropout = 0.0",
        "[loss]\nmse_weight = 1.0\nplcc_weight = 1.0\nrank_weight = 0.3",
        "[train]\nbatch_size = 8\nhead_epochs = 2\nhead_lr = 0.01",
        "finetune_epochs = 2\nhead_lr_multiplier = 10.0\nwarmup_ratio = 0.1",
        "weight_decay = 0.0\nmax_grad_norm = 1.0\nhflip = false",
        '[runtime]\ndevice = "cuda"\nprecision = "bf16"\nnum_workers = 2',
    ]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    argv = ["--config", str(config), "--run-id", "tiny-cli", "--seed", "3"]
    argv += ["--backbone-lr", "1e-3", "--data-root", str(tiny_job.data_root)]
    argv += ["--models-dir", str(tiny_job.models_dir)]
    argv += ["--runs-dir", str(tiny_job.runs_dir), "--device", "cpu"]
    argv += ["--precision", "fp32", "--num-workers", "0", "--batch-size", "4"]
    argv += ["--head-epochs", "1", "--finetune-epochs", "1"]

    assert train.main([*argv, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["n_train"] == 12
    assert train.main(argv) == 0
    assert "最佳" in capsys.readouterr().out
