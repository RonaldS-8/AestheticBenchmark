"""Prompt tuning on EVA images, scored exactly like the LoRA model (ADR-0012).

Prompts are not trained, so tuning needs no train/test split: every baseline
(B0, B1, B2) scores the same *prompt-tuning set*, all EVA images except the
fixed few-shot examples. The best prompt is chosen on that set, and the LoRA
model is trained and evaluated with the best single-image prompt. Its gain is
measured on the SFT eval split against the same prompt without LoRA, read
from the tuning run's eval-split slice.

Replies are parsed with the same strict two-score contract and reports have
the same format as LoRA runs (``EvaPrediction`` / ``EvaEvalReport``).
Correlations are against the unrounded EVA vote means.

The model is reached through a ``complete`` callable taking our messages and
returning the reply text, so this module needs neither torch nor a server.
"""

from __future__ import annotations

import base64
import io
import math
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import AwareDatetime, Field, model_validator

from lora_sft.common import ImageId, PromptId, Record, RunName, SchemaVersion, Sha256
from lora_sft.data.eva_sft import EvaSftSample
from lora_sft.data.samples import ImagePart, Message, TextPart
from lora_sft.data.schema import eval_fingerprint
from lora_sft.eval.eva_eval import (
    EVA_DIMENSIONS,
    EvaEvalReport,
    EvaEvalRunConfig,
    EvaPrediction,
    append_prediction,
    build_eva_report,
    parse_eva_reply,
    read_predictions,
)
from lora_sft.eval.prompts import FewShotExample, PromptTemplate, build_messages
from lora_sft.eval.schema import FEW_SHOT_METHODS, GgufWeights, Method
from lora_sft.train.config import BaseModelRef, ImagePreprocess

BASELINE_METHODS = frozenset(
    {Method.B0_ZERO_SHOT, Method.B1_RUBRIC, Method.B2_FEW_SHOT}
)
PIXEL_ALIGN = 32
JPEG_QUALITY = 90

Complete = Callable[[tuple[Message, ...]], str]


class BaselineError(ValueError):
    """Raised when a baseline run is inconsistent with its inputs."""


# --- prompt-tuning set ---


class PromptTuningSet(Record):
    """Images every baseline scores: all of EVA minus the few-shot examples.

    ``example_ids`` are picked once from the SFT train split by aesthetic
    quantiles and shown by B2; excluding them everywhere keeps all baselines
    on identical images. ``eval_ids_fingerprint`` is the SFT eval split, the
    slice used to compare against the LoRA model.
    """

    schema_version: SchemaVersion
    version: str = Field(pattern=r"^v\d+$")
    created_at: AwareDatetime
    sft_manifest: str = Field(min_length=1)
    sft_manifest_sha256: Sha256
    example_strategy: Literal["aesthetic_quantiles"]
    example_ids: tuple[ImageId, ...] = Field(min_length=1, max_length=8)
    item_count: int = Field(gt=0)
    ids_fingerprint: Sha256
    eval_ids_fingerprint: Sha256

    @model_validator(mode="after")
    def _check_ids(self) -> PromptTuningSet:
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("example_ids 有重复")
        return self


def select_quantile_examples(
    pool: Sequence[EvaSftSample], k: int
) -> tuple[EvaSftSample, ...]:
    """Pick ``k`` training images spread over the aesthetic score range.

    The pool is sorted by EVA aesthetic mean and one image is taken at the
    middle of each of ``k`` equal slices.
    """

    if any(sample.split != "train" for sample in pool):
        raise BaselineError("少样本示例只能取自 train")
    if not 1 <= k <= len(pool):
        raise BaselineError(f"k={k} 必须在 1 和训练池大小 {len(pool)} 之间")
    ordered = sorted(pool, key=lambda s: (s.source_labels.aesthetic.mean, s.image_id))
    return tuple(ordered[math.floor((i + 0.5) * len(ordered) / k)] for i in range(k))


def _tuning_items(
    train: Sequence[EvaSftSample],
    val: Sequence[EvaSftSample],
    eval_: Sequence[EvaSftSample],
    example_ids: Iterable[str],
) -> tuple[EvaSftSample, ...]:
    everything = [*train, *val, *eval_]
    ids = [s.image_id for s in everything]
    if len(set(ids)) != len(ids):
        raise BaselineError("train/val/eval 之间有重复 image_id")
    excluded = set(example_ids)
    return tuple(
        sorted(
            (s for s in everything if s.image_id not in excluded),
            key=lambda s: s.image_id,
        )
    )


def build_tuning_set(
    train: Sequence[EvaSftSample],
    val: Sequence[EvaSftSample],
    eval_: Sequence[EvaSftSample],
    *,
    k: int,
    version: str,
    sft_manifest: str,
    sft_manifest_sha256: str,
    now: datetime,
) -> PromptTuningSet:
    """Fix the few-shot examples and the images every baseline scores."""

    if {s.split for s in eval_} != {"eval"}:
        raise BaselineError("eval 样本的 split 必须都是 eval")
    examples = select_quantile_examples(train, k)
    items = _tuning_items(train, val, eval_, (e.image_id for e in examples))
    return PromptTuningSet(
        schema_version="1.0",
        version=version,
        created_at=now,
        sft_manifest=sft_manifest,
        sft_manifest_sha256=sft_manifest_sha256,
        example_strategy="aesthetic_quantiles",
        example_ids=tuple(e.image_id for e in examples),
        item_count=len(items),
        ids_fingerprint=eval_fingerprint(s.image_id for s in items),
        eval_ids_fingerprint=eval_fingerprint(s.image_id for s in eval_),
    )


def tuning_samples(
    tuning_set: PromptTuningSet,
    train: Sequence[EvaSftSample],
    val: Sequence[EvaSftSample],
    eval_: Sequence[EvaSftSample],
) -> tuple[tuple[EvaSftSample, ...], tuple[EvaSftSample, ...]]:
    """Rebuild ``(items, examples)`` and prove they match the saved set."""

    by_id = {s.image_id: s for s in train}
    missing = [i for i in tuning_set.example_ids if i not in by_id]
    if missing:
        raise BaselineError(f"少样本示例不在 train 里: {missing}")
    items = _tuning_items(train, val, eval_, tuning_set.example_ids)
    if (
        len(items) != tuning_set.item_count
        or eval_fingerprint(s.image_id for s in items) != tuning_set.ids_fingerprint
        or eval_fingerprint(s.image_id for s in eval_)
        != tuning_set.eval_ids_fingerprint
    ):
        raise BaselineError("样本与提示词调优集的指纹不符，SFT 数据可能变了")
    return items, tuple(by_id[i] for i in tuning_set.example_ids)


def write_tuning_set(path: Path, tuning_set: PromptTuningSet) -> None:
    """Write a new tuning-set file; existing files are never overwritten."""

    if path.exists():
        raise BaselineError(f"{path} 已存在，不覆盖；请换版本号")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        tuning_set.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


# --- baseline runs ---


class TuningSetRef(Record):
    """The tuning-set file a run scored, pinned by hash."""

    path: str = Field(min_length=1)
    sha256: Sha256


class EvaBaselineRunConfig(Record):
    """One prompt-baseline run over the prompt-tuning set.

    ``scoring`` holds the fields shared with LoRA runs (an ``EvaEvalRunConfig``
    with split ``all`` and no adapter); the rest is what makes it a baseline.
    """

    schema_version: SchemaVersion
    scoring: EvaEvalRunConfig
    tuning_set: TuningSetRef
    method: Method
    runtime: Literal["transformers", "llama_cpp"]
    gguf: GgufWeights | None

    @model_validator(mode="after")
    def _check_baseline(self) -> EvaBaselineRunConfig:
        if self.scoring.adapter is not None:
            raise ValueError("提示词基线不加载 LoRA 适配器")
        if self.scoring.split != "all":
            raise ValueError("提示词基线只在提示词调优集（split=all）上跑")
        if self.method is Method.B3_RETRIEVAL:
            raise ValueError("B3 检索增强还没有实现")
        if self.method not in BASELINE_METHODS:
            raise ValueError(f"{self.method.value} 不是提示词基线方法")
        if (self.runtime == "llama_cpp") != (self.gguf is not None):
            raise ValueError("llama_cpp 运行必须且只能填写 gguf")
        return self

    @property
    def run_id(self) -> str:
        """The run id shared with predictions and the report."""

        return self.scoring.run_id

    @property
    def is_few_shot(self) -> bool:
        """Whether this run shows the tuning set's examples."""

        return self.method in FEW_SHOT_METHODS


def to_few_shot_examples(
    samples: Sequence[EvaSftSample],
) -> tuple[FewShotExample, ...]:
    """Examples whose replies are the samples' SFT targets, verbatim."""

    return tuple(
        FewShotExample(
            media_id=sample.image_id,
            media_path=sample.image_path,
            reply_text=sample.messages[-1].text,
        )
        for sample in samples
    )


def prepare_baseline_run_dir(run_dir: Path, config: EvaBaselineRunConfig) -> set[str]:
    """Create or resume a run directory; return image ids already scored."""

    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = EvaBaselineRunConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        if existing != config:
            raise BaselineError(f"{config_path} 与本次配置不同，不能续跑；请换 run_id")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path.write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    if not (run_dir / "predictions.jsonl").exists():
        return set()
    done: set[str] = set()
    for prediction in read_predictions(run_dir):
        if prediction.run_id != config.run_id:
            raise BaselineError(f"{run_dir} 混有其他运行的预测")
        done.add(prediction.image_id)
    return done


def run_baseline(
    config: EvaBaselineRunConfig,
    prompt: PromptTemplate,
    tuning_set: PromptTuningSet,
    samples: Sequence[EvaSftSample],
    examples: Sequence[EvaSftSample],
    *,
    run_dir: Path,
    complete: Complete,
    clock: Callable[[], float] = time.perf_counter,
    on_progress: Callable[[int, int, EvaPrediction], None] | None = None,
) -> int:
    """Score every sample not yet in ``predictions.jsonl``; return how many ran.

    ``samples`` is the tuning set's items (or a prefix for a smoke test);
    ``examples`` are its few-shot images, passed only for B2.
    """

    if prompt.prompt_id != config.scoring.prompt_id:
        raise BaselineError(
            f"提示词 {prompt.prompt_id} 与配置 {config.scoring.prompt_id} 不符"
        )
    if config.method not in prompt.methods or prompt.media != "image":
        raise BaselineError(f"{prompt.prompt_id} 不用于图片上的 {config.method.value}")
    if eval_fingerprint(s.image_id for s in samples) != config.scoring.ids_fingerprint:
        raise BaselineError("样本与配置的 ids_fingerprint 不符")
    if {s.image_id for s in samples} & set(tuning_set.example_ids):
        raise BaselineError("少样本示例图不能出现在被评的图片里")
    planned = tuning_set.example_ids if config.is_few_shot else ()
    if tuple(e.image_id for e in examples) != planned:
        raise BaselineError("少样本示例与提示词调优集的 example_ids 不符")
    shots = to_few_shot_examples(examples)
    done = prepare_baseline_run_dir(run_dir, config)
    todo = [sample for sample in samples if sample.image_id not in done]
    for number, sample in enumerate(todo, 1):
        messages = build_messages(prompt, sample.image_id, sample.image_path, shots)
        started = clock()
        text = complete(messages)
        latency_ms = (clock() - started) * 1000
        prediction = parse_eva_reply(config.run_id, sample.image_id, text, latency_ms)
        append_prediction(run_dir, prediction)
        if on_progress is not None:
            on_progress(number, len(todo), prediction)
    return len(todo)


def _write_report(path: Path, report: EvaEvalReport) -> None:
    path.write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_baseline_report(
    config: EvaBaselineRunConfig,
    samples: Sequence[EvaSftSample],
    *,
    run_dir: Path,
    now: datetime,
) -> EvaEvalReport:
    """Build ``report.json`` over the tuning set the same way as for LoRA runs."""

    report = build_eva_report(
        config.scoring, read_predictions(run_dir), samples, now=now
    )
    _write_report(run_dir / "report.json", report)
    return report


def write_eval_slice_report(
    config: EvaBaselineRunConfig,
    tuning_set: PromptTuningSet,
    eval_samples: Sequence[EvaSftSample],
    *,
    run_dir: Path,
    now: datetime,
) -> EvaEvalReport:
    """Write ``report-eval.json``: the same run restricted to the SFT eval split.

    This is "the prompt without LoRA" on exactly the images the LoRA model is
    evaluated on, so the two reports are directly comparable.
    """

    ids = {s.image_id for s in eval_samples}
    if eval_fingerprint(ids) != tuning_set.eval_ids_fingerprint:
        raise BaselineError("eval 样本与提示词调优集记录的 eval 指纹不符")
    predictions = [p for p in read_predictions(run_dir) if p.image_id in ids]
    scoring = config.scoring.model_copy(
        update={"split": "eval", "ids_fingerprint": tuning_set.eval_ids_fingerprint}
    )
    report = build_eva_report(scoring, predictions, eval_samples, now=now)
    _write_report(run_dir / "report-eval.json", report)
    return report


# --- choosing the best prompt ---


class PromptCandidate(Record):
    """One tuning run's standing; ``score`` is the mean SRCC of both dimensions."""

    run_id: RunName
    prompt_id: PromptId
    method: Method
    parse_success_rate: float = Field(ge=0, le=1)
    srcc: dict[str, float]
    ineligible: dict[str, str]
    score: float | None = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def _check_score(self) -> PromptCandidate:
        if bool(self.ineligible) != (self.score is None):
            raise ValueError("只有不合格的候选没有分数")
        return self


class PromptSelection(Record):
    """Which prompt won prompt tuning, and the one the LoRA model must use.

    ``lora_prompt_id`` is the best prompt without in-context examples, so
    LoRA training and NAS inference stay at one image per item; B2 is kept
    as a reference.
    """

    schema_version: SchemaVersion
    created_at: AwareDatetime
    tuning_set: TuningSetRef
    ids_fingerprint: Sha256
    model: BaseModelRef
    runtime: Literal["transformers", "llama_cpp"]
    min_parse_rate: float = Field(ge=0, le=1)
    candidates: tuple[PromptCandidate, ...] = Field(min_length=1)
    best_prompt_id: PromptId
    lora_prompt_id: PromptId

    @model_validator(mode="after")
    def _check_choice(self) -> PromptSelection:
        by_prompt = {c.prompt_id: c for c in self.candidates}
        for chosen in (self.best_prompt_id, self.lora_prompt_id):
            if chosen not in by_prompt or by_prompt[chosen].score is None:
                raise ValueError(f"{chosen} 不是合格的候选")
        if by_prompt[self.lora_prompt_id].method in FEW_SHOT_METHODS:
            raise ValueError("LoRA 提示词不能是少样本提示词")
        return self


def _candidate(
    config: EvaBaselineRunConfig, report: EvaEvalReport, min_parse_rate: float
) -> PromptCandidate:
    ineligible: dict[str, str] = dict(report.metric_errors)
    if report.parse_success_rate < min_parse_rate:
        ineligible["parse_success_rate"] = (
            f"解析成功率 {report.parse_success_rate:.3f} 低于 {min_parse_rate}"
        )
    srcc = {dim: metric.srcc for dim, metric in report.metrics.items()}
    return PromptCandidate(
        run_id=config.run_id,
        prompt_id=config.scoring.prompt_id,
        method=config.method,
        parse_success_rate=report.parse_success_rate,
        srcc=srcc,
        ineligible=ineligible,
        score=None if ineligible else sum(srcc.values()) / len(EVA_DIMENSIONS),
    )


def _best(candidates: Sequence[PromptCandidate], label: str) -> PromptCandidate:
    scored = [c for c in candidates if c.score is not None]
    if not scored:
        raise BaselineError(f"没有合格的{label}")
    top = max(c.score for c in scored)  # type: ignore[type-var]
    winners = [c for c in scored if c.score == top]
    if len(winners) > 1:
        raise BaselineError(
            f"{label}并列: {[c.prompt_id for c in winners]}，请人工决定"
        )
    return winners[0]


def select_best_prompt(
    runs: Sequence[tuple[EvaBaselineRunConfig, EvaEvalReport]],
    *,
    min_parse_rate: float,
    now: datetime,
) -> PromptSelection:
    """Rank tuning runs; they must differ only in prompt and method."""

    if not runs:
        raise BaselineError("没有候选运行")
    first = runs[0][0]
    shared = (
        first.tuning_set,
        first.scoring.ids_fingerprint,
        first.scoring.model,
        first.scoring.preprocess,
        first.scoring.max_new_tokens,
        first.runtime,
        first.gguf,
    )
    for config, report in runs:
        same = (
            config.tuning_set,
            config.scoring.ids_fingerprint,
            config.scoring.model,
            config.scoring.preprocess,
            config.scoring.max_new_tokens,
            config.runtime,
            config.gguf,
        )
        if same != shared:
            raise BaselineError(f"{config.run_id} 与其他运行的图片、模型或运行时不同")
        if (report.run_id, report.split, report.ids_fingerprint) != (
            config.run_id,
            "all",
            config.scoring.ids_fingerprint,
        ):
            raise BaselineError(f"{config.run_id} 的报告不是这次运行在调优集上的报告")
    prompts = [config.scoring.prompt_id for config, _ in runs]
    if len(set(prompts)) != len(prompts):
        raise BaselineError("同一个提示词出现了多次")
    candidates = tuple(_candidate(c, r, min_parse_rate) for c, r in runs)
    best = _best(candidates, "提示词")
    single = _best(
        [c for c in candidates if c.method not in FEW_SHOT_METHODS], "单图提示词"
    )
    return PromptSelection(
        schema_version="1.0",
        created_at=now,
        tuning_set=first.tuning_set,
        ids_fingerprint=first.scoring.ids_fingerprint,
        model=first.scoring.model,
        runtime=first.runtime,
        min_parse_rate=min_parse_rate,
        candidates=candidates,
        best_prompt_id=best.prompt_id,
        lora_prompt_id=single.prompt_id,
    )


# --- LoRA gain ---


class DimensionGain(Record):
    """Agreement with EVA votes without and with LoRA, and the difference."""

    base_srcc: float
    lora_srcc: float
    delta_srcc: float
    base_plcc: float
    lora_plcc: float
    delta_plcc: float


class LoraGain(Record):
    """How much LoRA adds over the same prompt on the SFT eval split."""

    schema_version: SchemaVersion
    created_at: AwareDatetime
    prompt_id: PromptId
    base_run_id: RunName
    lora_run_id: RunName
    ids_fingerprint: Sha256
    base_parse_success_rate: float = Field(ge=0, le=1)
    lora_parse_success_rate: float = Field(ge=0, le=1)
    dimensions: dict[str, DimensionGain]


def lora_gain(
    selection: PromptSelection,
    base_config: EvaBaselineRunConfig,
    base_eval: EvaEvalReport,
    lora_config: EvaEvalRunConfig,
    lora_report: EvaEvalReport,
    *,
    now: datetime,
) -> LoraGain:
    """Compare LoRA with its prompt alone; both must follow the selection."""

    prompt_id = selection.lora_prompt_id
    if base_config.scoring.prompt_id != prompt_id:
        raise BaselineError(f"基线 {base_config.run_id} 用的不是选定提示词 {prompt_id}")
    if lora_config.prompt_id != prompt_id:
        raise BaselineError(
            f"LoRA 用的是 {lora_config.prompt_id}，应为选定的 {prompt_id}"
        )
    if lora_config.adapter is None:
        raise BaselineError(f"{lora_config.run_id} 没有加载 LoRA 适配器")
    if base_config.runtime != "transformers":
        raise BaselineError("LoRA 评测在 transformers 上，基线也必须是 transformers")
    if base_config.tuning_set != selection.tuning_set:
        raise BaselineError("基线运行不属于这次提示词选择")
    if (base_config.scoring.model, base_config.scoring.preprocess) != (
        lora_config.model,
        lora_config.preprocess,
    ):
        raise BaselineError("基线与 LoRA 的基座模型或像素预算不同")
    if base_config.scoring.max_new_tokens != lora_config.max_new_tokens:
        raise BaselineError("基线与 LoRA 的 max_new_tokens 不同")
    if base_eval.run_id != base_config.run_id or lora_report.run_id != (
        lora_config.run_id
    ):
        raise BaselineError("报告与运行配置不对应")
    if (base_eval.split, lora_report.split) != ("eval", "eval") or (
        base_eval.ids_fingerprint != lora_report.ids_fingerprint
    ):
        raise BaselineError("基线与 LoRA 必须是同一份 eval 集上的报告")
    dimensions: dict[str, DimensionGain] = {}
    for dim in EVA_DIMENSIONS:
        base, lora = base_eval.metrics.get(dim), lora_report.metrics.get(dim)
        if base is None or lora is None:
            raise BaselineError(f"{dim} 的相关系数无定义，算不出提升")
        dimensions[dim] = DimensionGain(
            base_srcc=base.srcc,
            lora_srcc=lora.srcc,
            delta_srcc=lora.srcc - base.srcc,
            base_plcc=base.plcc,
            lora_plcc=lora.plcc,
            delta_plcc=lora.plcc - base.plcc,
        )
    return LoraGain(
        schema_version="1.0",
        created_at=now,
        prompt_id=prompt_id,
        base_run_id=base_config.run_id,
        lora_run_id=lora_config.run_id,
        ids_fingerprint=lora_report.ids_fingerprint,
        base_parse_success_rate=base_eval.parse_success_rate,
        lora_parse_success_rate=lora_report.parse_success_rate,
        dimensions=dimensions,
    )


# --- message conversion and summaries ---


def hf_chat_messages(
    messages: Iterable[Message],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Hugging Face chat-template messages plus image paths in order.

    Image items carry only ``type``, exactly as in training: pixels go to the
    processor separately and the template renders the vision placeholder.
    """

    converted: list[dict[str, Any]] = []
    images: list[str] = []
    for message in messages:
        content: list[dict[str, str]] = []
        for part in message.content:
            if isinstance(part, ImagePart):
                content.append({"type": "image"})
                images.append(part.image)
            elif isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                raise BaselineError("图片基线不支持视频消息")
        converted.append({"role": message.role, "content": content})
    return converted, images


def fit_size(width: int, height: int, preprocess: ImagePreprocess) -> tuple[int, int]:
    """Scale to the pixel budget, keeping aspect ratio, sides multiple of 32."""

    pixels = width * height
    scale = 1.0
    if pixels > preprocess.max_pixels:
        scale = math.sqrt(preprocess.max_pixels / pixels)
    elif pixels < preprocess.min_pixels:
        scale = math.sqrt(preprocess.min_pixels / pixels)
    new_w = max(PIXEL_ALIGN, round(width * scale / PIXEL_ALIGN) * PIXEL_ALIGN)
    new_h = max(PIXEL_ALIGN, round(height * scale / PIXEL_ALIGN) * PIXEL_ALIGN)
    while new_w * new_h > preprocess.max_pixels and max(new_w, new_h) > PIXEL_ALIGN:
        if new_w >= new_h:
            new_w -= PIXEL_ALIGN
        else:
            new_h -= PIXEL_ALIGN
    return new_w, new_h


def _image_data_uri(path: Path, preprocess: ImagePreprocess) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.resize(fit_size(rgb.width, rgb.height, preprocess)).save(
        buffer, format="JPEG", quality=JPEG_QUALITY
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


def openai_chat_messages(
    messages: Iterable[Message], data_root: Path, preprocess: ImagePreprocess
) -> list[dict[str, Any]]:
    """OpenAI chat format for llama-server, images resized to the pixel budget."""

    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role != "user":
            converted.append({"role": message.role, "content": message.text})
            continue
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                parts.append(_image_data_uri(data_root / part.image, preprocess))
            else:
                raise BaselineError("图片基线不支持视频消息")
        converted.append({"role": "user", "content": parts})
    return converted


def comparison_table(reports: Sequence[EvaEvalReport]) -> str:
    """Plain-text table of EVA runs; all runs must cover the same images."""

    if not reports:
        raise BaselineError("没有报告")
    fingerprints = {(r.split, r.ids_fingerprint) for r in reports}
    if len(fingerprints) != 1:
        raise BaselineError("这些运行评的不是同一批图片，不能放在一张表里比较")
    lines = ["run | 解析率 | 美学 SRCC/PLCC | 技术 SRCC/PLCC | p50 ms"]
    for report in reports:
        cells = []
        for dim in ("aesthetic", "technical"):
            metric = report.metrics.get(dim)
            cells.append(
                f"{metric.srcc:.3f}/{metric.plcc:.3f}"
                if metric is not None
                else f"无定义（{report.metric_errors[dim]}）"
            )
        lines.append(
            f"{report.run_id} | {report.parse_success_rate:.3f} | "
            f"{cells[0]} | {cells[1]} | {report.latency_p50_ms:.0f}"
        )
    return "\n".join(lines)
