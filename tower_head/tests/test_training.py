from __future__ import annotations

import pytest
from qv_head.data import load_dataset
from qv_head.features import CachedFeatures, extract_features, load_features
from qv_head.head import load_head
from qv_head.records import EpochLog, TrainSummary
from qv_head.training import (
    HISTORY_FILE,
    SUMMARY_FILE,
    TrainError,
    plan_runs,
    run_sweep,
    train_run,
    warmup_cosine,
)
from tests.helpers import NOW

from tests.conftest import TinyJob, tiny_config


def _cache(job: TinyJob, run_id: str = "tiny-features") -> CachedFeatures:
    extract_features(
        job.config,
        data_root=job.data_root,
        models_dir=job.models_dir,
        runs_dir=job.runs_dir,
        run_id=run_id,
        limit=None,
        now=NOW,
        log=lambda _m: None,
    )
    return load_features(job.runs_dir, run_id, load_dataset(job.data_root))


def _sweep(job: TinyJob, lrs: list[str], seeds: list[int], **kwargs: object) -> list:
    options: dict[str, object] = {
        "data_root": job.data_root,
        "runs_dir": job.runs_dir,
        "features_run_id": "tiny-features",
        "prefix": "tiny",
        "lrs": lrs,
        "seeds": seeds,
        "clock": lambda: NOW,
        "log": lambda _m: None,
    }
    return run_sweep(job.config, **{**options, **kwargs})  # type: ignore[arg-type]


def test_train_run_keeps_the_best_epoch(tiny_job: TinyJob) -> None:
    cached = _cache(tiny_job)
    messages: list[str] = []

    summary = train_run(
        tiny_job.config,
        cached,
        runs_dir=tiny_job.runs_dir,
        run_id="tiny-a",
        seed=1,
        lr=1e-2,
        now=NOW,
        log=messages.append,
    )

    assert [e.epoch for e in summary.history] == [1, 2, 3]
    assert summary.best.val_srcc_mean == max(e.val_srcc_mean for e in summary.history)
    assert summary.tower_sha256 == cached.record.tower_sha256
    run_dir = tiny_job.runs_dir / "tiny-a"
    assert TrainSummary.model_validate_json((run_dir / SUMMARY_FILE).read_text()) == (
        summary
    )
    lines = (run_dir / HISTORY_FILE).read_text().splitlines()
    assert [EpochLog.model_validate_json(x) for x in lines] == list(summary.history)
    head, config, tower = load_head(run_dir / "best.pt", summary.checkpoint.sha256)
    assert config == tiny_job.config and tower == summary.tower_sha256
    assert messages[0].startswith("tiny-a: train 12 / val 8")
    assert any("新的最佳" in m for m in messages)


def test_train_run_is_reproducible_and_checks_inputs(tiny_job: TinyJob) -> None:
    cached = _cache(tiny_job)

    def run(run_id: str, **kwargs: object) -> TrainSummary:
        options: dict[str, object] = {"seed": 3, "lr": 1e-2, "now": NOW}
        options |= {"log": lambda _m: None, "runs_dir": tiny_job.runs_dir}
        return train_run(
            kwargs.pop("config", tiny_job.config),  # type: ignore[arg-type]
            cached,
            run_id=run_id,
            **{**options, **kwargs},  # type: ignore[arg-type]
        )

    first, second = run("tiny-a"), run("tiny-b")
    assert [e.train_loss for e in first.history] == pytest.approx(
        [e.train_loss for e in second.history]
    )
    with pytest.raises(TrainError, match="已存在"):
        run("tiny-a")
    with pytest.raises(TrainError, match="batch"):
        run("tiny-c", config=tiny_config(batch_size=64))
    with pytest.raises(ValueError):
        run("Tiny_C")


def test_plan_keeps_lr_text_and_rejects_duplicates() -> None:
    plan = plan_runs("p", ["1e-4", "3e-4"], [1, 2])

    assert [r for r, _, _ in plan] == [
        "p-lr1e-4-s1",
        "p-lr1e-4-s2",
        "p-lr3e-4-s1",
        "p-lr3e-4-s2",
    ]
    assert plan[2][1:] == (3e-4, 1)
    with pytest.raises(TrainError, match="重复"):
        plan_runs("p", ["1e-4", "1e-4"], [1])


def test_sweep_trains_skips_and_stops_on_partial_runs(tiny_job: TinyJob) -> None:
    _cache(tiny_job)
    messages: list[str] = []

    first = _sweep(tiny_job, ["1e-2"], [1], log=messages.append)
    again = _sweep(tiny_job, ["1e-2"], [1, 2], log=messages.append)

    assert again[0] == first[0] and len(again) == 2
    assert any(m.startswith("[1/1]") and "开始 tiny-lr1e-2-s1" in m for m in messages)
    assert any("跳过已完成的 tiny-lr1e-2-s1" in m for m in messages)
    (tiny_job.runs_dir / "tiny-lr1e-2-s3").mkdir()
    with pytest.raises(TrainError, match="没跑完"):
        _sweep(tiny_job, ["1e-2"], [3])


def test_sweep_refuses_features_from_other_tower_settings(tiny_job: TinyJob) -> None:
    _cache(tiny_job)
    other = tiny_job.config.model_copy(
        update={"tower": tiny_job.config.tower.model_copy(update={"taps": (0, 3)})}
    )

    with pytest.raises(TrainError, match="tower"):
        run_sweep(
            other,
            data_root=tiny_job.data_root,
            runs_dir=tiny_job.runs_dir,
            features_run_id="tiny-features",
            prefix="tiny",
            lrs=["1e-2"],
            seeds=[1],
            clock=lambda: NOW,
        )


def test_warmup_then_cosine_to_zero() -> None:
    factor = warmup_cosine(total_steps=10, warmup_steps=2)

    assert [factor(0), factor(1)] == [0.5, 1.0]
    assert factor(10) == pytest.approx(0.0)
