from __future__ import annotations

import json

import pytest
from qv_head import evaluation as evaluation_module
from qv_head.data import load_dataset
from qv_head.evaluation import build_report, evaluate_run
from qv_head.features import extract_features, load_features
from qv_head.records import EvaEvalReport, HeadEvalRunConfig, HeadPrediction
from qv_head.training import SUMMARY_FILE, TrainError, train_run
from tests.helpers import NOW

from tests.conftest import TinyJob


def _trained(job: TinyJob, *, limit: int | None = None) -> str:
    extract_features(
        job.config,
        data_root=job.data_root,
        models_dir=job.models_dir,
        runs_dir=job.runs_dir,
        run_id="tiny-features",
        limit=limit,
        now=NOW,
        log=lambda _m: None,
    )
    cached = load_features(job.runs_dir, "tiny-features", load_dataset(job.data_root))
    train_run(
        job.config,
        cached,
        runs_dir=job.runs_dir,
        run_id="tiny-lr1e-2-s1",
        seed=1,
        lr=1e-2,
        now=NOW,
        log=lambda _m: None,
    )
    return "tiny-lr1e-2-s1"


def _evaluate(
    job: TinyJob, train_run_id: str, split: str, run_id: str
) -> EvaEvalReport:
    return evaluate_run(
        data_root=job.data_root,
        models_dir=job.models_dir,
        runs_dir=job.runs_dir,
        train_run_id=train_run_id,
        split=split,  # type: ignore[arg-type]
        run_id=run_id,
        now=NOW,
        log=lambda _m: None,
    )


def test_val_end_to_end_matches_cached_features(tiny_job: TinyJob) -> None:
    trained = _trained(tiny_job)
    summary = json.loads((tiny_job.runs_dir / trained / SUMMARY_FILE).read_text())

    report = _evaluate(tiny_job, trained, "val", "tiny-val")

    for dim in ("aesthetic", "technical"):
        assert report.metrics[dim].srcc == pytest.approx(
            summary["best"]["val"][dim]["srcc"], abs=1e-6
        )
    out = tiny_job.runs_dir / "tiny-val"
    assert EvaEvalReport.model_validate_json((out / "report.json").read_text()) == (
        report
    )
    config = HeadEvalRunConfig.model_validate_json((out / "config.json").read_text())
    assert config.tower_precision == "fp32" and config.split == "val"
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 8


def test_eval_scores_the_held_out_split_once(tiny_job: TinyJob) -> None:
    trained = _trained(tiny_job)

    report = _evaluate(tiny_job, trained, "eval", "tiny-eval")

    assert report.n_items == 4 and report.split == "eval"
    with pytest.raises(TrainError, match="已存在"):
        _evaluate(tiny_job, trained, "eval", "tiny-eval")


def test_val_check_catches_a_different_path(
    tiny_job: TinyJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained = _trained(tiny_job)
    monkeypatch.setattr(evaluation_module, "VAL_TOLERANCE", -1.0)

    with pytest.raises(TrainError, match="不一致"):
        _evaluate(tiny_job, trained, "val", "tiny-val")
    assert not (tiny_job.runs_dir / "tiny-val").exists()


def test_refuses_bad_requests(tiny_job: TinyJob) -> None:
    with pytest.raises(TrainError, match="val 或 eval"):
        _evaluate(tiny_job, "x", "all", "tiny-all")
    with pytest.raises(TrainError, match="找不到"):
        _evaluate(tiny_job, "missing", "eval", "tiny-eval")


def test_refuses_smoke_runs_and_other_towers(tiny_job: TinyJob) -> None:
    trained = _trained(tiny_job, limit=8)
    with pytest.raises(TrainError, match="冒烟"):
        _evaluate(tiny_job, trained, "eval", "tiny-eval")

    path = tiny_job.runs_dir / trained / SUMMARY_FILE
    data = json.loads(path.read_text())
    data["limited"] = False
    data["tower_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TrainError, match="同一份权重"):
        _evaluate(tiny_job, trained, "eval", "tiny-eval")
    data["items_sha256"] = "1" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TrainError, match="数据"):
        _evaluate(tiny_job, trained, "eval", "tiny-eval")


def test_report_needs_one_prediction_per_item(tiny_job: TinyJob) -> None:
    items = load_dataset(tiny_job.data_root).split("eval")
    config = HeadEvalRunConfig(
        schema_version="1.0",
        run_id="r",
        split="eval",
        ids_fingerprint="a" * 64,
        train_run_id="t",
        checkpoint={"path": "t/best.pt", "sha256": "b" * 64},  # type: ignore[arg-type]
        tower_sha256="c" * 64,
        device="cpu",
        tower_precision="fp32",
    )
    same = [
        HeadPrediction(
            run_id="r", image_id=i.image_id, aesthetic=3.0, technical=3.0, latency_ms=1
        )
        for i in items
    ]

    report = build_report(config, same, items, now=NOW)

    assert sorted(report.metric_errors) == ["aesthetic", "technical"]
    with pytest.raises(TrainError, match="一一对应"):
        build_report(config, same[:-1], items, now=NOW)
