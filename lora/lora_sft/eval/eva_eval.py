"""Evaluate a Qwen model on EVA dual-score samples (stage one, ADR-0009).

The base model with the same prompt (B0) and the LoRA model are scored the
same way, so the LoRA run must beat B0 on identical inputs. Replies are parsed
strictly as ``EvaDualScoreOutput`` once format wrappers (a think block, a
code fence) are removed; a failure is recorded, never repaired.
Correlations are computed against the unrounded EVA vote means.

Each run lives in ``data/runs/<run_id>/``: ``config.json`` is written once
and must match on resume, ``predictions.jsonl`` gets one line per image and
``report.json`` is written at the end.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from lora_sft.common import ImageId, PromptId, Record, RunName, SchemaVersion, Sha256
from lora_sft.data.eva_sft import EvaDualScoreOutput, EvaSftSample
from lora_sft.eval.metrics import MetricsError, percentile, plcc, srcc
from lora_sft.train.config import BaseModelRef, ImagePreprocess

EVA_DIMENSIONS = ("aesthetic", "technical")
# "all" is the prompt-tuning set: every EVA image except the few-shot examples.
EvalSplit = Literal["val", "eval", "all"]
# Below the SRCC range [-1, 1]: an ineligible checkpoint can never be selected.
INELIGIBLE_SELECTION_SCORE = -2.0


class EvaEvalError(ValueError):
    """Raised when an EVA eval run is inconsistent with its inputs."""


class AdapterRef(Record):
    """A trained LoRA adapter, pinned by the hash of its weights file."""

    train_run_name: RunName
    path: str = Field(min_length=1)
    weights_sha256: Sha256


class EvaEvalRunConfig(Record):
    """One scoring run over one EVA split; ``adapter=None`` is the B0 baseline."""

    schema_version: SchemaVersion
    run_id: RunName
    split: EvalSplit
    ids_fingerprint: Sha256
    model: BaseModelRef
    adapter: AdapterRef | None
    prompt_id: PromptId
    chat_template_sha256: Sha256
    preprocess: ImagePreprocess
    max_new_tokens: int = Field(ge=8, le=512)
    device: str = Field(min_length=1)


class EvaPrediction(Record):
    """Output for one image; exactly one of ``scores`` or ``parse_error``."""

    run_id: RunName
    image_id: ImageId
    raw_output: str
    scores: EvaDualScoreOutput | None
    parse_error: str | None
    latency_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_outcome(self) -> EvaPrediction:
        if (self.scores is None) == (self.parse_error is None):
            raise ValueError("scores 和 parse_error 必须恰好填一个")
        return self


class DimensionMetrics(Record):
    """Agreement with EVA vote means on one dimension."""

    srcc: float = Field(ge=-1, le=1)
    plcc: float = Field(ge=-1, le=1)
    mae_1_5: float = Field(ge=0)


class EvaEvalReport(Record):
    """Summary of one EVA run; correlations use parsed predictions only.

    A dimension whose correlation is undefined (fewer than two parsed
    replies, or a constant prediction) is listed in ``metric_errors``.
    """

    schema_version: SchemaVersion
    run_id: RunName
    split: EvalSplit
    ids_fingerprint: Sha256
    created_at: AwareDatetime
    n_items: int = Field(ge=1)
    n_parsed: int = Field(ge=0)
    parse_success_rate: float = Field(ge=0, le=1)
    metrics: dict[str, DimensionMetrics]
    metric_errors: dict[str, str]
    latency_p50_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_dimensions(self) -> EvaEvalReport:
        if sorted([*self.metrics, *self.metric_errors]) != sorted(EVA_DIMENSIONS):
            raise ValueError("每个维度必须恰好有指标或无定义原因之一")
        return self

    latency_p95_ms: float = Field(gt=0)


class SelectionScore(Record):
    """Val agreement used to pick a training checkpoint.

    ``score`` is the mean SRCC over ``EVA_DIMENSIONS``, or
    ``INELIGIBLE_SELECTION_SCORE`` when ``ineligible`` lists a reason (parse
    rate below the minimum, or an SRCC that is undefined).
    """

    n_items: int = Field(ge=1)
    parse_success_rate: float = Field(ge=0, le=1)
    srcc: dict[str, float]
    ineligible: dict[str, str]
    score: float = Field(ge=INELIGIBLE_SELECTION_SCORE, le=1)

    @model_validator(mode="after")
    def _check_score(self) -> SelectionScore:
        if bool(self.ineligible) != (self.score == INELIGIBLE_SELECTION_SCORE):
            raise ValueError("只有不合格的检查点才能取不合格分数")
        return self


# Format noise around the JSON object, removed before strict parsing
# (ADR-0013): an optional leading think block, then whatever surrounds the one
# complete top-level JSON object in the reply (code fences of any style,
# inline code, prose). Zero or several objects leave the text as it is, so it
# fails; the object itself is never edited.
_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)
_JSON_DECODER = json.JSONDecoder()


def _json_objects(text: str) -> list[str]:
    """Source spans of every complete top-level JSON object in ``text``."""

    spans: list[str] = []
    start = text.find("{")
    while start != -1:
        try:
            value, end = _JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            spans.append(text[start:end])
        start = text.find("{", end)
    return spans


def unwrap_reply(raw_output: str) -> str:
    """The single JSON object in the reply, verbatim; else the reply itself."""

    text = _THINK_BLOCK.sub("", raw_output, count=1)
    objects = _json_objects(text)
    return objects[0] if len(objects) == 1 else text


def selection_score(
    replies: dict[str, str],
    samples: Sequence[EvaSftSample],
    *,
    min_parse_rate: float,
) -> SelectionScore:
    """Score one checkpoint's val replies (``image_id -> raw text``).

    Replies are parsed as strictly as in a full eval run; SRCC is against the
    unrounded vote means of the parsed images only.
    """

    by_id = {sample.image_id: sample for sample in samples}
    if replies.keys() != by_id.keys():
        missing = sorted(by_id.keys() - replies.keys())[:3]
        extra = sorted(replies.keys() - by_id.keys())[:3]
        raise EvaEvalError(f"回复与验证样本不一致: missing={missing}, extra={extra}")
    parsed: dict[str, EvaDualScoreOutput] = {}
    for image_id, text in replies.items():
        try:
            reply = unwrap_reply(text)
            parsed[image_id] = EvaDualScoreOutput.model_validate_json(reply)
        except ValidationError:
            continue
    rate = len(parsed) / len(replies)
    srccs: dict[str, float] = {}
    ineligible: dict[str, str] = {}
    if rate < min_parse_rate:
        ineligible["parse_success_rate"] = (
            f"解析成功率 {rate:.3f} 低于 {min_parse_rate}"
        )
    for dim in EVA_DIMENSIONS:
        pred = [getattr(scores, dim) for scores in parsed.values()]
        means = [getattr(by_id[i].source_labels, dim).mean for i in parsed]
        try:
            srccs[dim] = srcc(pred, means)
        except MetricsError as exc:
            ineligible[dim] = str(exc)
    score = (
        INELIGIBLE_SELECTION_SCORE
        if ineligible
        else sum(srccs.values()) / len(EVA_DIMENSIONS)
    )
    return SelectionScore(
        n_items=len(replies),
        parse_success_rate=rate,
        srcc=srccs,
        ineligible=ineligible,
        score=score,
    )


def parse_eva_reply(
    run_id: str, image_id: str, raw_output: str, latency_ms: float
) -> EvaPrediction:
    """Parse model text strictly after ``unwrap_reply``; keep any error."""

    try:
        scores = EvaDualScoreOutput.model_validate_json(unwrap_reply(raw_output))
    except ValidationError as exc:
        return EvaPrediction(
            run_id=run_id,
            image_id=image_id,
            raw_output=raw_output,
            scores=None,
            parse_error=str(exc.errors()[0]["msg"]),
            latency_ms=latency_ms,
        )
    return EvaPrediction(
        run_id=run_id,
        image_id=image_id,
        raw_output=raw_output,
        scores=scores,
        parse_error=None,
        latency_ms=latency_ms,
    )


def prepare_eva_run_dir(run_dir: Path, config: EvaEvalRunConfig) -> set[str]:
    """Create or resume a run directory; return image ids already scored."""

    config_path = run_dir / "config.json"
    predictions_path = run_dir / "predictions.jsonl"
    if config_path.exists():
        existing = EvaEvalRunConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        if existing != config:
            raise EvaEvalError(f"{config_path} 与本次配置不同，不能续跑；请换 run_id")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path.write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    done: set[str] = set()
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            prediction = EvaPrediction.model_validate_json(line)
            if prediction.run_id != config.run_id:
                raise EvaEvalError(f"{predictions_path} 混有其他运行的预测")
            done.add(prediction.image_id)
    return done


def append_prediction(run_dir: Path, prediction: EvaPrediction) -> None:
    """Append one prediction so an interrupted run resumes where it stopped."""

    with (run_dir / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(prediction.model_dump(mode="json"), ensure_ascii=False))
        f.write("\n")


def read_predictions(run_dir: Path) -> list[EvaPrediction]:
    """Load every prediction written so far."""

    path = run_dir / "predictions.jsonl"
    return [
        EvaPrediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def build_eva_report(
    config: EvaEvalRunConfig,
    predictions: Sequence[EvaPrediction],
    samples: Sequence[EvaSftSample],
    *,
    now: datetime,
) -> EvaEvalReport:
    """Aggregate predictions for every sample of the split into a report."""

    by_id = {sample.image_id: sample for sample in samples}
    predicted = {p.image_id: p for p in predictions}
    if len(predicted) != len(predictions):
        raise EvaEvalError("预测里有重复 image_id")
    if predicted.keys() != by_id.keys():
        missing = sorted(by_id.keys() - predicted.keys())[:3]
        extra = sorted(predicted.keys() - by_id.keys())[:3]
        raise EvaEvalError(f"预测与样本不一致: missing={missing}, extra={extra}")
    if any(p.run_id != config.run_id for p in predictions):
        raise EvaEvalError("预测里混有其他运行")
    parsed = [p for p in predictions if p.scores is not None]
    metrics: dict[str, DimensionMetrics] = {}
    metric_errors: dict[str, str] = {}
    for dim in EVA_DIMENSIONS:
        pred = [getattr(p.scores, dim) for p in parsed]
        means = [getattr(by_id[p.image_id].source_labels, dim).mean for p in parsed]
        targets = [getattr(by_id[p.image_id].target, dim) for p in parsed]
        try:
            metrics[dim] = DimensionMetrics(
                srcc=srcc(pred, means),
                plcc=plcc(pred, means),
                mae_1_5=sum(abs(a - b) for a, b in zip(pred, targets, strict=True))
                / len(parsed),
            )
        except MetricsError as exc:
            metric_errors[dim] = str(exc)
    latencies = [p.latency_ms for p in predictions]
    return EvaEvalReport(
        schema_version="1.0",
        run_id=config.run_id,
        split=config.split,
        ids_fingerprint=config.ids_fingerprint,
        created_at=now,
        n_items=len(predictions),
        n_parsed=len(parsed),
        parse_success_rate=len(parsed) / len(predictions),
        metrics=metrics,
        metric_errors=metric_errors,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )
