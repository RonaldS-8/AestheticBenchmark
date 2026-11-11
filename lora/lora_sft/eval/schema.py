"""Evaluation runs, predictions, reports and the deployment gate.

These are the stage-two schemas (three scores plus a comment). The same eval
set (identified by ``eval_fingerprint``) is used for the prompt-only
baselines, the LoRA SFT model, the aesthetic head and their exports on the
NAS, so all reports are directly comparable. The stage-one EVA two-score runs
use ``lora_sft.eval.eva_eval`` and ``lora_sft.eval.eva_baseline``; the video
and DOVER comparison is archived under ``archive/video-dover/``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from lora_sft.common import (
    MediaId,
    PromptId,
    RawScores,
    Record,
    RunName,
    SchemaVersion,
    Sha256,
)
from lora_sft.data.samples import COMMENT_MAX_CHARS, AestheticOutput
from lora_sft.data.schema import Category
from lora_sft.train.config import BaseModelRef, ImagePreprocess


class Method(StrEnum):
    """VLM methods: baselines from the eval plan plus the LoRA model.

    B4 (model size) is not a method: run the same method on another
    ``model`` instead. The aesthetic head uses ``HeadEvalRunConfig``.
    """

    B0_ZERO_SHOT = "b0_zero_shot"
    B1_RUBRIC = "b1_rubric"
    B2_FEW_SHOT = "b2_few_shot"
    B3_RETRIEVAL = "b3_retrieval"
    LORA_SFT = "lora_sft"


FEW_SHOT_METHODS = frozenset({Method.B2_FEW_SHOT, Method.B3_RETRIEVAL})


class GgufWeights(Record):
    """Third-party GGUF files a prompt baseline runs on, pinned by hash.

    Used for baselines on llama.cpp; our own trained models are referenced
    through ``export_id`` instead.
    """

    model_file: str = Field(min_length=1)
    model_sha256: Sha256
    mmproj_file: str = Field(min_length=1)
    mmproj_sha256: Sha256
    quantization: str = Field(pattern=r"^[a-z0-9_]+$")


class VlmEvalRunConfig(Record):
    """One VLM evaluation run of one method on the frozen eval set.

    On llama.cpp a prompt baseline names its GGUF files in ``gguf``; the LoRA
    model names its export in ``export_id``.
    """

    schema_version: SchemaVersion
    route: Literal["vlm"]
    run_id: RunName
    method: Method
    model: BaseModelRef
    train_run_name: RunName | None
    export_id: RunName | None
    gguf: GgufWeights | None
    prompt_id: PromptId
    few_shot_k: int = Field(ge=0, le=8)
    preprocess: ImagePreprocess
    chat_template_sha256: Sha256
    runtime: Literal["transformers", "llama_cpp", "mlx"]
    device: str = Field(min_length=1)
    eval_fingerprint: Sha256

    @model_validator(mode="after")
    def _check_method(self) -> VlmEvalRunConfig:
        if (self.method in FEW_SHOT_METHODS) != (self.few_shot_k > 0):
            raise ValueError(
                f"{self.method.value} 的 few_shot_k={self.few_shot_k} 不合理"
            )
        if (self.method is Method.LORA_SFT) != (self.train_run_name is not None):
            raise ValueError("只有 lora_sft 方法需要且必须填写 train_run_name")
        on_device = self.runtime != "transformers"
        if not on_device and (self.export_id or self.gguf):
            raise ValueError("transformers 运行不填 export_id 或 gguf")
        if on_device and (self.export_id is None) == (self.gguf is None):
            raise ValueError("端侧运行必须且只能填 export_id 或 gguf 之一")
        if self.gguf is not None and (
            self.runtime != "llama_cpp" or self.method is Method.LORA_SFT
        ):
            raise ValueError("gguf 只用于 llama_cpp 上的提示词基线")
        return self


class HeadEvalRunConfig(Record):
    """One aesthetic-head evaluation run; views come from the training run."""

    schema_version: SchemaVersion
    route: Literal["head"]
    run_id: RunName
    train_run_name: RunName
    export_id: RunName | None
    runtime: Literal["torch", "onnxruntime"]
    device: str = Field(min_length=1)
    eval_fingerprint: Sha256

    @model_validator(mode="after")
    def _check_runtime(self) -> HeadEvalRunConfig:
        if (self.runtime == "onnxruntime") != (self.export_id is not None):
            raise ValueError("onnxruntime 必须且只能通过 export_id 指定导出模型")
        return self


EvalRunConfig = Annotated[
    VlmEvalRunConfig | HeadEvalRunConfig, Field(discriminator="route")
]
EVAL_RUN_ADAPTER: TypeAdapter[VlmEvalRunConfig | HeadEvalRunConfig] = TypeAdapter(
    EvalRunConfig
)


class Prediction(Record):
    """Output for one eval item; a parse failure is kept, not fixed.

    VLM predictions keep ``raw_output`` and may carry a comment; head
    predictions have scores only. Scores keep the model's own scale.
    """

    run_id: RunName
    media_id: MediaId
    raw_output: str | None
    scores: RawScores | None
    comment: str | None = Field(min_length=1, max_length=COMMENT_MAX_CHARS)
    parse_error: str | None
    latency_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_outcome(self) -> Prediction:
        if (self.scores is None) == (self.parse_error is None):
            raise ValueError("scores 和 parse_error 必须恰好填一个")
        if self.raw_output is None and (
            self.parse_error is not None or self.comment is not None
        ):
            raise ValueError("解析失败或带评语的预测必须保留 raw_output")
        return self


def parse_prediction(
    run_id: str, media_id: str, raw_output: str, latency_ms: float
) -> Prediction:
    """Parse raw model text strictly as ``AestheticOutput``.

    Nothing is stripped or repaired (no code-fence removal, no partial JSON):
    the output format is a product requirement and failures are counted.
    """

    try:
        output = AestheticOutput.model_validate_json(raw_output)
    except ValidationError as exc:
        return Prediction(
            run_id=run_id,
            media_id=media_id,
            raw_output=raw_output,
            scores=None,
            comment=None,
            parse_error=str(exc),
            latency_ms=latency_ms,
        )
    return Prediction(
        run_id=run_id,
        media_id=media_id,
        raw_output=raw_output,
        scores=RawScores(
            aesthetic=output.aesthetic,
            technical=output.technical,
            overall=output.overall,
        ),
        comment=output.comment,
        parse_error=None,
        latency_ms=latency_ms,
    )


class CorrelationPair(Record):
    """Spearman and Pearson correlation against human mean scores."""

    srcc: float = Field(ge=-1, le=1)
    plcc: float = Field(ge=-1, le=1)
    n: int = Field(ge=2)


class DimensionMetrics(Record):
    """Correlations for each score dimension."""

    aesthetic: CorrelationPair
    technical: CorrelationPair
    overall: CorrelationPair


class EvalReport(Record):
    """Aggregated results of one eval run.

    Correlations use parsed predictions only, so they must always be read
    together with ``parse_success_rate``. ``annotator_ceiling`` is ``None``
    when the dataset publishes only mean scores; ``by_category`` is empty
    when it has no categories.
    """

    schema_version: SchemaVersion
    run_id: RunName
    eval_fingerprint: Sha256
    n_items: int = Field(ge=1)
    n_parsed: int = Field(ge=0)
    metrics: DimensionMetrics
    by_category: dict[Category, DimensionMetrics]
    pairwise_agreement: float = Field(ge=0, le=1)
    pairwise_min_gap: float = Field(gt=0)
    latency_p50_ms: float = Field(gt=0)
    latency_p95_ms: float = Field(gt=0)
    annotator_ceiling: DimensionMetrics | None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check_counts(self) -> EvalReport:
        if self.n_parsed > self.n_items:
            raise ValueError("n_parsed 不能大于 n_items")
        if self.latency_p50_ms > self.latency_p95_ms:
            raise ValueError("latency_p50_ms 不能大于 latency_p95_ms")
        return self

    @property
    def parse_success_rate(self) -> float:
        """Share of eval items whose output parsed."""

        return self.n_parsed / self.n_items


class GateThresholds(Record):
    """Pass criteria; values are set once annotator agreement is known."""

    min_overall_srcc_gain: float = Field(ge=0)
    min_parse_success_rate: float = Field(ge=0, le=1)
    max_nas_srcc_drop: float = Field(ge=0)
    max_latency_p95_ms: float = Field(gt=0)


class GateDecision(Record):
    """Whether a trained model (LoRA or head) may be deployed.

    ``candidate_run_id`` is the cloud eval of the trained model,
    ``nas_run_id`` the eval of its export on the NAS.
    """

    schema_version: SchemaVersion
    eval_fingerprint: Sha256
    candidate_run_id: RunName
    baseline_run_id: RunName
    nas_run_id: RunName | None
    thresholds: GateThresholds
    passed: bool
    failures: tuple[str, ...]
    decided_at: AwareDatetime

    @model_validator(mode="after")
    def _check_verdict(self) -> GateDecision:
        if self.passed == bool(self.failures):
            raise ValueError("passed 必须与 failures 是否为空一致")
        if self.passed and self.nas_run_id is None:
            raise ValueError("未在 NAS 上评测导出模型，不能判定通过")
        return self
