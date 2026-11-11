"""Training run configuration, loaded from a TOML file under ``configs/``.

Two routes share one loader, picked by ``route``: ``lora_sft`` fine-tunes the
Qwen VLM with LoRA, ``head`` trains a DOVER-style aesthetic head. Every field
is required: a run must state its model revisions, data version and
hyperparameters explicitly so it can be reproduced.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from lora_sft.common import (
    PromptId,
    Record,
    RelativePath,
    RunName,
    SchemaVersion,
    Sha256,
)
from lora_sft.data.head_samples import LabelSource


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


class DataRef(Record):
    """Which split and sample files a run trains on."""

    split_manifest: RelativePath
    split_version: str = Field(pattern=r"^split-v\d+$")
    eval_fingerprint: Sha256
    train_samples: RelativePath
    val_samples: RelativePath


class LoraSettings(Record):
    """PEFT LoRA adapter settings."""

    r: int = Field(ge=1, le=256)
    alpha: int = Field(ge=1)
    dropout: float = Field(ge=0, lt=1)
    target_modules: tuple[str, ...] = Field(min_length=1)
    freeze_vision: bool

    @model_validator(mode="after")
    def _check_modules(self) -> LoraSettings:
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules 有重复")
        return self


class KernelSettings(Record):
    """Fast kernels the run must find before it may start.

    transformers resolves Qwen3.5's Gated DeltaNet kernels when it is imported
    and silently falls back to a much slower pure-torch path if a package is
    missing or fails to import, so a required kernel is checked explicitly.
    """

    require_fla: bool
    require_causal_conv1d: bool


class SelectionSettings(Record):
    """How the kept checkpoint is chosen on the val split.

    ``srcc_mean`` greedily decodes every val image at each evaluation, parses
    the reply strictly and averages the aesthetic and technical SRCC against
    the vote means. A checkpoint whose parse rate is below ``min_parse_rate``
    or whose SRCC is undefined can never be selected.
    """

    metric: Literal["srcc_mean", "eval_loss"]
    min_parse_rate: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(ge=8, le=512)


class OptimSettings(Record):
    """Optimizer and schedule."""

    learning_rate: float = Field(gt=0, lt=1)
    epochs: int = Field(ge=1)
    per_device_batch_size: int = Field(ge=1)
    grad_accum_steps: int = Field(ge=1)
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    lr_scheduler: Literal["cosine", "linear", "constant"]
    max_grad_norm: float = Field(gt=0)


class RuntimeSettings(Record):
    """Hardware, precision and checkpointing. DeepSpeed is not supported."""

    strategy: Literal["single", "ddp", "fsdp"]
    num_gpus: int = Field(ge=1)
    precision: Literal["bf16", "fp32"]
    # "flash_attention_2" needs flash-attn (Linux wheel in requirements-train.txt).
    attn_implementation: Literal["sdpa", "flash_attention_2"]
    gradient_checkpointing: bool
    save_steps: int = Field(ge=1)
    eval_steps: int = Field(ge=1)
    output_dir: RelativePath

    @model_validator(mode="after")
    def _check_gpus(self) -> RuntimeSettings:
        if (self.strategy == "single") != (self.num_gpus == 1):
            raise ValueError(
                f"strategy={self.strategy} 与 num_gpus={self.num_gpus} 不匹配"
            )
        return self


class _RunBase(Record):
    """Fields shared by every training route."""

    schema_version: SchemaVersion
    run_name: RunName
    seed: int = Field(ge=0)
    data: DataRef
    optim: OptimSettings
    runtime: RuntimeSettings

    @property
    def global_batch_size(self) -> int:
        """Samples per optimizer step across all GPUs."""

        return (
            self.optim.per_device_batch_size
            * self.optim.grad_accum_steps
            * self.runtime.num_gpus
        )


class LoraSftConfig(_RunBase):
    """One LoRA SFT run. Loss is computed on assistant tokens only."""

    route: Literal["lora_sft"]
    base_model: BaseModelRef
    prompt_id: PromptId
    chat_template_sha256: Sha256
    preprocess: ImagePreprocess
    max_seq_len: int = Field(ge=256)
    lora: LoraSettings
    kernels: KernelSettings
    selection: SelectionSettings

    @model_validator(mode="after")
    def _check_checkpoints(self) -> LoraSftConfig:
        # Trainer only keeps a best checkpoint that was saved at the step it
        # won; an evaluation without a save would silently keep an older one.
        if self.runtime.save_steps != self.runtime.eval_steps:
            raise ValueError(
                "save_steps 必须等于 eval_steps，否则最佳检查点可能没被保存"
            )
        return self


class AestheticView(Record):
    """Whole image resized down; sees composition, colour and subject."""

    backbone: BaseModelRef
    size: int = Field(ge=32, le=1024)


class TechnicalView(Record):
    """DOVER fragments: grid x grid native-resolution patches stitched together.

    Keeping original pixels preserves noise, blur and focus that resizing
    would hide.
    """

    backbone: BaseModelRef
    grid: int = Field(ge=1, le=16)
    patch: int = Field(ge=8, le=128)

    @property
    def size(self) -> int:
        """Side length of the stitched fragment image."""

        return self.grid * self.patch


class HeadViews(Record):
    """The two input views; inference must use identical settings."""

    aesthetic: AestheticView
    technical: TechnicalView


class HeadSettings(Record):
    """MLP heads on pooled backbone features.

    Aesthetic uses the aesthetic view, technical uses the technical view and
    overall uses both views' features concatenated.
    """

    hidden_dim: int = Field(ge=1, le=4096)
    dropout: float = Field(ge=0, lt=1)


class HeadLoss(Record):
    """``plcc_rank`` is DOVER's PLCC loss plus a weighted pairwise rank loss."""

    kind: Literal["mse", "plcc_rank"]
    rank_weight: float | None = Field(gt=0)

    @model_validator(mode="after")
    def _check_weight(self) -> HeadLoss:
        if (self.kind == "plcc_rank") != (self.rank_weight is not None):
            raise ValueError("只有 plcc_rank 需要且必须填写 rank_weight")
        return self


class HeadConfig(_RunBase):
    """One aesthetic-head run.

    Weak-label pre-training on site ratings and fine-tuning on human means
    are separate runs; the second sets ``init_from_run`` to the first.
    """

    route: Literal["head"]
    views: HeadViews
    freeze_backbones: bool
    head: HeadSettings
    loss: HeadLoss
    label_sources: tuple[LabelSource, ...] = Field(min_length=1)
    init_from_run: RunName | None

    @model_validator(mode="after")
    def _check_sources(self) -> HeadConfig:
        if len(set(self.label_sources)) != len(self.label_sources):
            raise ValueError("label_sources 有重复")
        if self.init_from_run == self.run_name:
            raise ValueError("init_from_run 不能是本次运行")
        return self


TrainConfig = Annotated[LoraSftConfig | HeadConfig, Field(discriminator="route")]
TRAIN_CONFIG_ADAPTER: TypeAdapter[LoraSftConfig | HeadConfig] = TypeAdapter(TrainConfig)


def load_train_config(path: Path) -> LoraSftConfig | HeadConfig:
    """Load and validate a TOML run config of either route."""

    with path.open("rb") as f:
        return TRAIN_CONFIG_ADAPTER.validate_python(tomllib.load(f))
