"""Records read and written by a prompt-tuning job.

Inputs (``JobConfig``, ``PromptTuningSet``, ``TuningItem``) are exported by the
main repository (``scripts/pack_prompt_tuning.py``). Outputs
(``EvaBaselineRunConfig``, ``EvaPrediction``, ``EvaEvalReport``,
``PromptSelection``) are copies of the main repository's records, field for
field, so the main repository reads them back for the LoRA comparison;
``tests/test_prompt_tuning_sync.py`` there checks the copies stay identical.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

SCORE_MIN = 1
SCORE_MAX = 5
EVA_DIMENSIONS = ("aesthetic", "technical")


def _check_relative_posix(value: str) -> str:
    """Accept only normalized relative POSIX paths inside the data root."""

    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or ":" in value
        or ".." in path.parts
        or value != path.as_posix()
        or value == "."
    ):
        raise ValueError(f"路径必须是数据根目录下规范的相对 POSIX 路径: {value!r}")
    return value


SchemaVersion = Literal["1.0"]
RelativePath = Annotated[str, AfterValidator(_check_relative_posix)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Score = Annotated[float, Field(ge=SCORE_MIN, le=SCORE_MAX)]
ImageId = Annotated[str, Field(pattern=r"^photo_\d{6,}$")]
MediaId = Annotated[str, Field(pattern=r"^(photo|video)_\d{6,}$")]
PromptId = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*@v\d+$")]
RunName = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
# "all" is the prompt-tuning set: every EVA image except the few-shot examples.
EvalSplit = Literal["val", "eval", "all"]


class Record(BaseModel):
    """Base for records: immutable and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Method(StrEnum):
    """VLM methods: baselines from the eval plan plus the LoRA model."""

    B0_ZERO_SHOT = "b0_zero_shot"
    B1_RUBRIC = "b1_rubric"
    B2_FEW_SHOT = "b2_few_shot"
    B3_RETRIEVAL = "b3_retrieval"
    LORA_SFT = "lora_sft"


FEW_SHOT_METHODS = frozenset({Method.B2_FEW_SHOT, Method.B3_RETRIEVAL})
BASELINE_METHODS = frozenset(
    {Method.B0_ZERO_SHOT, Method.B1_RUBRIC, Method.B2_FEW_SHOT}
)


class EvaDualScoreOutput(Record):
    """The only reply the model may give: two scores, nothing else."""

    aesthetic: Score
    technical: Score


class BaseModelRef(Record):
    """A Hugging Face model pinned to one commit."""

    model_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ImagePreprocess(Record):
    """Qwen VL processor pixel budget; inference must use the same values."""

    min_pixels: int = Field(gt=0)
    max_pixels: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> ImagePreprocess:
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels 不能大于 max_pixels")
        return self


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


class GgufWeights(Record):
    """Third-party GGUF files a prompt baseline runs on, pinned by hash."""

    model_file: str = Field(min_length=1)
    model_sha256: Sha256
    mmproj_file: str = Field(min_length=1)
    mmproj_sha256: Sha256
    quantization: str = Field(pattern=r"^[a-z0-9_]+$")


class PromptTuningSet(Record):
    """Images every baseline scores: all of EVA minus the few-shot examples."""

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


class TuningSetRef(Record):
    """The tuning-set file a run scored, pinned by hash."""

    path: str = Field(min_length=1)
    sha256: Sha256


class EvaBaselineRunConfig(Record):
    """One prompt-baseline run over the prompt-tuning set."""

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
    """Which prompt won prompt tuning, and the one the LoRA model must use."""

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


# --- job inputs (exported by the main repository) ---


class TuningItem(Record):
    """One EVA image with its vote means and SFT target reply.

    ``role`` is ``example`` for the few-shot images B2 shows and ``item`` for
    every image that is scored. ``reply_text`` is the SFT assistant reply,
    verbatim; ``target`` parses it.
    """

    image_id: ImageId
    split: Literal["train", "val", "eval"]
    role: Literal["item", "example"]
    image_path: RelativePath
    image_sha256: Sha256
    aesthetic_mean: float
    technical_mean: float
    reply_text: str = Field(min_length=2)

    @model_validator(mode="after")
    def _check_reply(self) -> TuningItem:
        try:
            target = EvaDualScoreOutput.model_validate_json(self.reply_text)
        except ValidationError as exc:
            raise ValueError(
                f"{self.image_id} 的 reply_text 不是合法双分 JSON"
            ) from exc
        for name in EVA_DIMENSIONS:
            value = getattr(target, name)
            if round(value, 1) != value:
                raise ValueError(f"{self.image_id} 的目标 {name}={value} 必须一位小数")
        return self

    @property
    def target(self) -> EvaDualScoreOutput:
        """The SFT target scores."""

        return EvaDualScoreOutput.model_validate(json.loads(self.reply_text))

    def mean(self, dimension: str) -> float:
        """Unrounded EVA vote mean for ``aesthetic`` or ``technical``."""

        return float(getattr(self, f"{dimension}_mean"))


class FileRef(Record):
    """A job file, pinned by hash, relative to the job root."""

    path: RelativePath
    sha256: Sha256


class WeightFile(Record):
    """One model file the job needs, pinned by content."""

    path: RelativePath
    sha256: Sha256
    size_bytes: int = Field(gt=0)


class JobConfig(Record):
    """Everything a prompt-tuning job needs, copied from the LoRA run config.

    Model, revision, chat template, pixel budget, precision, reply length and
    the parse-rate threshold come from the LoRA config, so baselines and the
    LoRA eval differ only in prompt and weights. ``weights`` lists every model
    file with its SHA-256, taken from a copy already checked against the
    pinned revision, so weights from any download source are verified.
    """

    schema_version: SchemaVersion
    train_config: str = Field(min_length=1)
    train_config_sha256: Sha256
    model: BaseModelRef
    chat_template_sha256: Sha256
    preprocess: ImagePreprocess
    precision: Literal["bf16", "fp32"]
    max_new_tokens: int = Field(ge=8, le=512)
    min_parse_rate: float = Field(ge=0, le=1)
    tuning_set: FileRef
    items: FileRef
    weights: tuple[WeightFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_weights(self) -> JobConfig:
        paths = [w.path for w in self.weights]
        if len(set(paths)) != len(paths):
            raise ValueError("weights 有重复文件")
        template = {w.path: w for w in self.weights}.get("chat_template.jinja")
        if template is None:
            raise ValueError("weights 必须包含 chat_template.jinja")
        if template.sha256 != self.chat_template_sha256:
            raise ValueError("weights 里 chat_template.jinja 的哈希与配置不符")
        return self
