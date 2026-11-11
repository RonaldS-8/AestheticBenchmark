from __future__ import annotations

import json

import evaluate
import pytest
from aes_head.data import load_dataset
from aes_head.evaluation import build_report, evaluate_run
from aes_head.finetune import SUMMARY_FILE, TrainError, finetune
from aes_head.records import EvaEvalReport, HeadEvalRunConfig, HeadPrediction
from tests.helpers import NOW

from tests.conftest import TinyJob


@pytest.fixture
def trained(tiny_job: TinyJob) -> TinyJob:
    finetune(
        tiny_job.config,
        data_root=tiny_job.data_root,
        models_dir=tiny_job.models_dir,
        runs_dir=tiny_job.runs_dir,
        run_id="tiny-train",
        seed=1,
        backbone_lr=1e-3,
        limit=None,
        now=NOW,
        log=lambda _message: None,
    )
    return tiny_job


def _evaluate(job: TinyJob, split: str, run_id: str, **kwargs: object) -> object:
    options: dict[str, object] = {
        "data_root": job.data_root,
        "runs_dir": job.runs_dir,
        "train_run_id": "tiny-train",
        "split": split,
        "run_id": run_id,
        "device": "cpu",
        "precision": "fp32",
        "now": NOW,
        "log": lambda _message: None,
    }
    return evaluate_run(**{**options, **kwargs})  # type: ignore[arg-type]


def test_evaluate_writes_lora_style_report(trained: TinyJob) -> None:
    report = _evaluate(trained, "eval", "tiny-eval")

    out = trained.runs_dir / "tiny-eval"
    assert isinstance(report, EvaEvalReport)
    assert report.n_items == report.n_parsed == 4
    assert report.parse_success_rate == 1.0
    assert sorted(report.metrics) == ["aesthetic", "technical"]
    assert (
        EvaEvalReport.model_validate_json((out / "report.json").read_text()) == report
    )
    config = HeadEvalRunConfig.model_validate_json((out / "config.json").read_text())
    assert config.train_run_id == "tiny-train" and config.split == "eval"
    lines = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4


def test_evaluate_refuses_bad_requests(trained: TinyJob) -> None:
    _evaluate(trained, "val", "tiny-val")

    with pytest.raises(TrainError, match="已存在"):
        _evaluate(trained, "val", "tiny-val")
    with pytest.raises(TrainError, match="val 或 eval"):
        _evaluate(trained, "all", "tiny-all")
    with pytest.raises(TrainError, match="训练记录"):
        _evaluate(trained, "val", "tiny-x", train_run_id="missing")


def test_evaluate_refuses_other_data(trained: TinyJob) -> None:
    path = trained.runs_dir / "tiny-train" / SUMMARY_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["items_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(TrainError, match="数据"):
        _evaluate(trained, "val", "tiny-val")


def test_build_report_checks_ids_and_records_undefined_metrics(
    trained: TinyJob,
) -> None:
    items = load_dataset(trained.data_root).split("val")
    config = HeadEvalRunConfig.model_validate(
        {
            "schema_version": "1.0",
            "run_id": "tiny-val",
            "split": "val",
            "ids_fingerprint": "a" * 64,
            "train_run_id": "tiny-train",
            "checkpoint": {"path": "tiny-train/best.pt", "sha256": "b" * 64},
            "device": "cpu",
            "precision": "fp32",
        }
    )
    flat = [
        HeadPrediction(
            run_id="tiny-val",
            image_id=item.image_id,
            aesthetic=3.0,
            technical=1.0 + i,
            latency_ms=1.0,
        )
        for i, item in enumerate(items)
    ]

    report = build_report(config, flat, items, now=NOW)

    assert "aesthetic" in report.metric_errors
    assert "technical" in report.metrics
    with pytest.raises(TrainError, match="一一对应"):
        build_report(config, flat[:-1], items, now=NOW)


def test_cli_prints_metrics(
    trained: TinyJob, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--train-run", "tiny-train", "--split", "val", "--run-id", "tiny-cli"]
    argv += ["--data-root", str(trained.data_root), "--runs-dir"]
    argv += [str(trained.runs_dir), "--device", "cpu", "--precision", "fp32"]

    assert evaluate.main(argv) == 0
    assert "SRCC" in capsys.readouterr().out
