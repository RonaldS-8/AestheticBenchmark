"""Records written by feature extraction, head training, selection and eval.

``EvaEvalReport`` is a field-for-field copy of the main repository's LoRA
report, so tower-head, ConvNeXt-head and LoRA results go into one table;
``tests/test_sync_with_main.py`` checks the schema and the numbers match.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from qv_head.config import JobConfig, TowerSettings
from qv_head.schema import (
    EVA_DIMENSIONS,
    DimensionMetrics,
    ImageId,
    Record,
    RelativePath,
    RunName,
    SchemaVersion,
    Sha256,
)

EvalSplit = Literal["val", "eval", "all"]


class FileRef(Record):
    """A file under the runs directory, pinned by its hash."""

    path: RelativePath
    sha256: Sha256


class FeatureSet(Record):
    """``runs/<run_id>/features.json``: cached tower features of train and val.

    The eval split is never cached; it is scored end to end, once.
    """

    schema_version: SchemaVersion
    run_id: RunName
    created_at: AwareDatetime
    items_sha256: Sha256
    tower_sha256: Sha256
    tower: TowerSettings
    layer_names: tuple[str, ...] = Field(min_length=2)
    layer_dims: tuple[int, ...] = Field(min_length=2)
    n_train: int = Field(gt=0)
    n_val: int = Field(gt=0)
    limited: bool
    latency_p50_ms: float = Field(gt=0)
    latency_p95_ms: float = Field(gt=0)
    file: FileRef

    @model_validator(mode="after")
    def _check_layers(self) -> FeatureSet:
        if len(self.layer_names) != len(self.layer_dims):
            raise ValueError("layer_names 与 layer_dims 数量不一致")
        if self.layer_names[-1] != "merger":
            raise ValueError("最后一段特征必须是 merger")
        return self

    @property
    def feature_dim(self) -> int:
        """Width of one concatenated feature row."""

        return sum(self.layer_dims)


class EpochLog(Record):
    """Train loss and val agreement after one epoch."""

    epoch: int = Field(ge=1)
    train_loss: float
    val: dict[str, DimensionMetrics]
    val_srcc_mean: float = Field(ge=-1, le=1)


class TrainSummary(Record):
    """``runs/<run_id>/train-summary.json``: one finished head-training run."""

    schema_version: SchemaVersion
    run_id: RunName
    created_at: AwareDatetime
    config: JobConfig
    seed: int
    lr: float = Field(gt=0)
    items_sha256: Sha256
    tower_sha256: Sha256
    features: FileRef
    n_train: int = Field(gt=0)
    n_val: int = Field(gt=0)
    limited: bool
    history: tuple[EpochLog, ...] = Field(min_length=1)
    best: EpochLog
    checkpoint: FileRef
    seconds: float = Field(gt=0)


class SweepGroup(Record):
    """Runs of one learning rate, differing only in seed."""

    lr: float = Field(gt=0)
    run_ids: tuple[RunName, ...] = Field(min_length=1)
    best_val_srcc_mean: tuple[float, ...] = Field(min_length=1)
    mean: float


class SweepSelection(Record):
    """The learning rate with the best seed-averaged val SRCC, and its best run."""

    schema_version: SchemaVersion
    prefix: RunName
    created_at: AwareDatetime
    recipe: RunName
    groups: tuple[SweepGroup, ...] = Field(min_length=1)
    chosen_run_id: RunName
    checkpoint: FileRef


class HeadEvalRunConfig(Record):
    """``config.json`` of one end-to-end scoring run over one split."""

    schema_version: SchemaVersion
    run_id: RunName
    split: EvalSplit
    ids_fingerprint: Sha256
    train_run_id: RunName
    checkpoint: FileRef
    tower_sha256: Sha256
    device: str = Field(min_length=1)
    tower_precision: Literal["bf16", "fp32"]


class HeadPrediction(Record):
    """Scores for one image on the 1-5 scale (not clipped)."""

    run_id: RunName
    image_id: ImageId
    aesthetic: float = Field(allow_inf_nan=False)
    technical: float = Field(allow_inf_nan=False)
    latency_ms: float = Field(gt=0)


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
