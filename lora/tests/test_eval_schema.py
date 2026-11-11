from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from lora_sft.eval.schema import (
    EVAL_RUN_ADAPTER,
    EvalReport,
    GateDecision,
    HeadEvalRunConfig,
    Method,
    Prediction,
    VlmEvalRunConfig,
    parse_prediction,
)

HEAD_RUN: dict[str, Any] = {
    "schema_version": "1.0",
    "route": "head",
    "run_id": "head-convnext-swin-human-cloud",
    "train_run_name": "head-convnext-swin-human",
    "export_id": None,
    "runtime": "torch",
    "device": "h100-80gb",
    "eval_fingerprint": "7" * 64,
}

GOOD_OUTPUT = json.dumps(
    {"aesthetic": 4.0, "technical": 3.5, "overall": 3.8, "comment": "光线柔和。"},
    ensure_ascii=False,
)


def test_run_config_fixture_is_valid(eval_run_data: dict[str, Any]) -> None:
    assert VlmEvalRunConfig.model_validate(eval_run_data).method is Method.B1_RUBRIC


@pytest.mark.parametrize(
    ("method", "k", "valid"),
    [
        ("b0_zero_shot", 0, True),
        ("b1_rubric", 3, False),
        ("b2_few_shot", 0, False),
        ("b2_few_shot", 3, True),
        ("b3_retrieval", 5, True),
        ("b3_retrieval", 9, False),
    ],
)
def test_few_shot_k_matches_method(
    eval_run_data: dict[str, Any], method: str, k: int, valid: bool
) -> None:
    eval_run_data.update(method=method, few_shot_k=k)

    if valid:
        assert VlmEvalRunConfig.model_validate(eval_run_data).few_shot_k == k
    else:
        with pytest.raises(ValidationError, match="few_shot_k"):
            VlmEvalRunConfig.model_validate(eval_run_data)


@pytest.mark.parametrize(
    ("method", "train_run", "valid"),
    [
        ("lora_sft", "qwen-4b-lora-r16", True),
        ("lora_sft", None, False),
        ("b1_rubric", "qwen-4b-lora-r16", False),
    ],
)
def test_train_run_only_for_lora(
    eval_run_data: dict[str, Any], method: str, train_run: str | None, valid: bool
) -> None:
    eval_run_data.update(method=method, train_run_name=train_run)

    if valid:
        assert (
            VlmEvalRunConfig.model_validate(eval_run_data).train_run_name == train_run
        )
    else:
        with pytest.raises(ValidationError, match="train_run_name"):
            VlmEvalRunConfig.model_validate(eval_run_data)


@pytest.mark.parametrize(
    ("runtime", "export_id", "valid"),
    [
        ("llama_cpp", "qwen-4b-lora-r16-q4-k-m", True),
        ("mlx", None, False),
        ("transformers", "qwen-4b-lora-r16-q4-k-m", False),
    ],
)
def test_on_device_runtime_needs_export(
    eval_run_data: dict[str, Any], runtime: str, export_id: str | None, valid: bool
) -> None:
    eval_run_data.update(runtime=runtime, export_id=export_id)

    if valid:
        assert VlmEvalRunConfig.model_validate(eval_run_data).export_id == export_id
    else:
        with pytest.raises(ValidationError, match="export_id"):
            VlmEvalRunConfig.model_validate(eval_run_data)


def test_parse_prediction_success() -> None:
    prediction = parse_prediction("run-a", "photo_000001", GOOD_OUTPUT, 1200.0)

    assert prediction.scores is not None and prediction.scores.overall == 3.8
    assert prediction.comment == "光线柔和。"
    assert prediction.parse_error is None


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n" + GOOD_OUTPUT + "\n```",
        GOOD_OUTPUT[:-1],
        GOOD_OUTPUT.replace('"comment"', '"note"'),
        "这张照片总体 4 分。",
    ],
)
def test_parse_prediction_records_failure(raw: str) -> None:
    prediction = parse_prediction("run-a", "photo_000001", raw, 1200.0)

    assert prediction.scores is None
    assert prediction.parse_error
    assert prediction.raw_output == raw


@pytest.mark.parametrize("both", [True, False])
def test_prediction_needs_exactly_one_outcome(both: bool) -> None:
    scores = {"aesthetic": 4.0, "technical": 3.5, "overall": 3.8} if both else None
    error = "bad" if both else None

    with pytest.raises(ValidationError, match="恰好填一个"):
        Prediction(
            run_id="run-a",
            media_id="photo_000001",
            raw_output=GOOD_OUTPUT,
            scores=scores,
            comment=None,
            parse_error=error,
            latency_ms=10.0,
        )


def test_prediction_latency_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="latency_ms"):
        parse_prediction("run-a", "photo_000001", GOOD_OUTPUT, 0.0)


def test_report_fixture_is_valid(eval_report_data: dict[str, Any]) -> None:
    report = EvalReport.model_validate(eval_report_data)

    assert report.parse_success_rate == pytest.approx(788 / 800)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"n_parsed": 801}, "n_parsed"),
        ({"latency_p50_ms": 3000.0}, "latency_p50_ms"),
        ({"pairwise_agreement": 1.01}, "pairwise_agreement"),
        ({"pairwise_min_gap": 0.0}, "pairwise_min_gap"),
        ({"n_items": 0}, "n_items"),
    ],
)
def test_report_bounds(
    eval_report_data: dict[str, Any], updates: dict[str, Any], message: str
) -> None:
    eval_report_data.update(updates)

    with pytest.raises(ValidationError, match=message):
        EvalReport.model_validate(eval_report_data)


@pytest.mark.parametrize(("key", "value"), [("srcc", 1.01), ("plcc", -1.01), ("n", 1)])
def test_correlation_bounds(
    eval_report_data: dict[str, Any], key: str, value: float
) -> None:
    eval_report_data["metrics"]["overall"][key] = value

    with pytest.raises(ValidationError, match=key):
        EvalReport.model_validate(eval_report_data)


def test_report_rejects_unknown_category(eval_report_data: dict[str, Any]) -> None:
    eval_report_data["by_category"]["selfie"] = eval_report_data["metrics"]

    with pytest.raises(ValidationError, match="selfie"):
        EvalReport.model_validate(eval_report_data)


def test_gate_fixture_is_valid(gate_data: dict[str, Any]) -> None:
    assert GateDecision.model_validate(gate_data).passed is True


def test_failed_gate_is_valid(gate_data: dict[str, Any]) -> None:
    gate_data.update(
        passed=False, failures=["overall SRCC 提升 0.01 < 0.05"], nas_run_id=None
    )

    assert GateDecision.model_validate(gate_data).failures


@pytest.mark.parametrize(
    ("passed", "failures"), [(True, ["量化掉点过多"]), (False, [])]
)
def test_gate_verdict_must_match_failures(
    gate_data: dict[str, Any], passed: bool, failures: list[str]
) -> None:
    gate_data.update(passed=passed, failures=failures)

    with pytest.raises(ValidationError, match="passed"):
        GateDecision.model_validate(gate_data)


def test_gate_cannot_pass_without_nas_eval(gate_data: dict[str, Any]) -> None:
    gate_data["nas_run_id"] = None

    with pytest.raises(ValidationError, match="NAS"):
        GateDecision.model_validate(gate_data)


def test_gate_thresholds_have_no_defaults(gate_data: dict[str, Any]) -> None:
    del gate_data["thresholds"]["max_latency_p95_ms"]

    with pytest.raises(ValidationError, match="max_latency_p95_ms"):
        GateDecision.model_validate(gate_data)


def test_run_adapter_picks_route(eval_run_data: dict[str, Any]) -> None:
    assert isinstance(EVAL_RUN_ADAPTER.validate_python(eval_run_data), VlmEvalRunConfig)
    assert isinstance(EVAL_RUN_ADAPTER.validate_python(HEAD_RUN), HeadEvalRunConfig)


@pytest.mark.parametrize(
    ("runtime", "export_id", "valid"),
    [
        ("torch", None, True),
        ("onnxruntime", "head-convnext-swin-human-fp16", True),
        ("onnxruntime", None, False),
        ("torch", "head-convnext-swin-human-fp16", False),
    ],
)
def test_head_runtime_needs_export(
    runtime: str, export_id: str | None, valid: bool
) -> None:
    data = HEAD_RUN | {"runtime": runtime, "export_id": export_id}

    if valid:
        assert HeadEvalRunConfig.model_validate(data).runtime == runtime
    else:
        with pytest.raises(ValidationError, match="export_id"):
            HeadEvalRunConfig.model_validate(data)


def test_head_run_rejects_prompt_fields() -> None:
    data = HEAD_RUN | {"prompt_id": "rubric-score@v1"}

    with pytest.raises(ValidationError, match="prompt_id"):
        HeadEvalRunConfig.model_validate(data)


def test_head_prediction_has_scores_only() -> None:
    prediction = Prediction(
        run_id="head-run",
        media_id="photo_000001",
        raw_output=None,
        scores={"aesthetic": 3.2, "technical": 4.1, "overall": 3.6},
        comment=None,
        parse_error=None,
        latency_ms=35.0,
    )

    assert prediction.scores is not None and prediction.comment is None


@pytest.mark.parametrize(
    ("scores", "comment", "error"),
    [
        (None, None, "bad"),
        ({"aesthetic": 3.2, "technical": 4.1, "overall": 3.6}, "好", None),
    ],
)
def test_prediction_without_raw_output_rules(
    scores: dict[str, float] | None, comment: str | None, error: str | None
) -> None:
    with pytest.raises(ValidationError, match="raw_output"):
        Prediction(
            run_id="head-run",
            media_id="photo_000001",
            raw_output=None,
            scores=scores,
            comment=comment,
            parse_error=error,
            latency_ms=35.0,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_prediction_scores_must_be_finite(bad: float) -> None:
    with pytest.raises(ValidationError, match="overall"):
        Prediction(
            run_id="head-run",
            media_id="photo_000001",
            raw_output=None,
            scores={"aesthetic": 0.1, "technical": 0.2, "overall": bad},
            comment=None,
            parse_error=None,
            latency_ms=120.0,
        )


@pytest.mark.parametrize("media_id", ["clip_000001", "video_1", "photo"])
def test_prediction_media_id_format(media_id: str) -> None:
    with pytest.raises(ValidationError, match="media_id"):
        parse_prediction("run-a", media_id, GOOD_OUTPUT, 10.0)


def test_report_without_annotator_ceiling(eval_report_data: dict[str, Any]) -> None:
    eval_report_data.update(annotator_ceiling=None, by_category={})

    report = EvalReport.model_validate(eval_report_data)

    assert report.annotator_ceiling is None and report.by_category == {}


GGUF = {
    "model_file": "Qwen3.5-4B-Q4_K_M.gguf",
    "model_sha256": "3" * 64,
    "mmproj_file": "mmproj-Qwen3.5-4B-f16.gguf",
    "mmproj_sha256": "4" * 64,
    "quantization": "q4_k_m",
}


def test_baseline_on_llama_cpp_with_gguf(eval_run_data: dict[str, Any]) -> None:
    eval_run_data.update(runtime="llama_cpp", gguf=GGUF)

    config = VlmEvalRunConfig.model_validate(eval_run_data)

    assert config.gguf is not None and config.gguf.quantization == "q4_k_m"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"runtime": "transformers", "gguf": GGUF}, "transformers 运行不填"),
        (
            {"runtime": "llama_cpp", "gguf": GGUF, "export_id": "some-export"},
            "之一",
        ),
        ({"runtime": "mlx", "gguf": GGUF}, "gguf 只用于 llama_cpp"),
        (
            {
                "runtime": "llama_cpp",
                "gguf": GGUF,
                "method": "lora_sft",
                "train_run_name": "qwen-4b-lora-r16",
            },
            "gguf 只用于 llama_cpp",
        ),
        ({"runtime": "llama_cpp", "gguf": GGUF | {"model_sha256": "x"}}, "sha256"),
    ],
)
def test_gguf_rules(
    eval_run_data: dict[str, Any], updates: dict[str, Any], message: str
) -> None:
    eval_run_data.update(updates)

    with pytest.raises(ValidationError, match=message):
        VlmEvalRunConfig.model_validate(eval_run_data)
