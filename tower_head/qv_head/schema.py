"""Records of the Qwen vision-tower head job.

The dataset files (``data/items.jsonl``, ``data/dataset.json``) are written
once by ``prepare_data.py`` from the main repository's EVA SFT manifest and
read only through this module afterwards, so the job never imports
``lora_sft``. ``ScoreSummary`` and the 1-5 mapping are copies of the main
repository's; ``tests/test_sync_with_main.py`` checks they stay identical.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

SCORE_MIN = 1
SCORE_MAX = 5
EVA_DIMENSIONS = ("aesthetic", "technical")
SPLITS = ("train", "val", "eval")


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
ImageId = Annotated[str, Field(pattern=r"^photo_\d{6,}$")]
RunName = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
Split = Literal["train", "val", "eval"]
Dimension = Literal["aesthetic", "technical"]
LayerSet = Literal["all", "last", "merger"]


class Record(BaseModel):
    """Base for records: immutable and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreSummary(Record):
    """Aggregate of source ratings on their original numeric scale."""

    mean: float
    standard_deviation: float = Field(ge=0)
    count: int = Field(gt=0)
    scale_min: float
    scale_max: float

    @model_validator(mode="after")
    def _check_scale(self) -> ScoreSummary:
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min 必须小于 scale_max")
        if not self.scale_min <= self.mean <= self.scale_max:
            raise ValueError(
                f"mean={self.mean} 超出评分区间 "
                f"[{self.scale_min}, {self.scale_max}]"
            )
        return self

    def to_project_scale(self) -> float:
        """Map the mean linearly onto 1-5, unrounded (the regression target)."""

        position = (self.mean - self.scale_min) / (self.scale_max - self.scale_min)
        return SCORE_MIN + (SCORE_MAX - SCORE_MIN) * position

    def to_reported_target(self) -> float:
        """The 1-5 target rounded to 0.1, as the LoRA eval reports MAE against."""

        return round(self.to_project_scale(), 1)


class HeadItem(Record):
    """One EVA image with its split, hash and source vote summaries."""

    image_id: ImageId
    split: Split
    image_path: RelativePath
    image_sha256: Sha256
    image_decode_warnings: tuple[str, ...]
    aesthetic: ScoreSummary
    technical: ScoreSummary

    def summary(self, dimension: Dimension) -> ScoreSummary:
        """Vote summary of one dimension."""

        return self.aesthetic if dimension == "aesthetic" else self.technical


class SourceRef(Record):
    """The main-repository file the dataset was copied from."""

    path: RelativePath
    sha256: Sha256


class DatasetInfo(Record):
    """``data/dataset.json``: what was copied and how to verify it."""

    schema_version: SchemaVersion
    dataset: Literal["eva"]
    created_at: AwareDatetime
    source_manifest: SourceRef
    split_version: str = Field(min_length=1)
    items_path: RelativePath
    items_sha256: Sha256
    counts: dict[Split, int]
    eval_fingerprint: Sha256

    @model_validator(mode="after")
    def _check_counts(self) -> DatasetInfo:
        if sorted(self.counts) != sorted(SPLITS) or min(self.counts.values()) < 1:
            raise ValueError("counts 必须给出 train、val、eval 三个划分且都不为空")
        return self


class TowerFile(Record):
    """One file of the extracted vision tower, pinned by its SHA-256."""

    path: RelativePath
    sha256: Sha256
    size_bytes: int = Field(gt=0)


class TowerWeights(Record):
    """``models/WEIGHTS.json``: the vision tower cut out of the base model.

    ``source_files`` are the base-model files it was cut from (config,
    preprocessor config, weight index and the shards holding the tower);
    ``files`` are what the job loads.
    """

    schema_version: SchemaVersion
    model_id: str = Field(min_length=1)
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    weight_prefix: str = Field(min_length=1)
    tensor_count: int = Field(gt=0)
    parameter_count: int = Field(gt=0)
    source_files: dict[str, Sha256]
    files: tuple[TowerFile, ...] = Field(min_length=1)

    def find(self, name: str) -> TowerFile:
        """The file called ``name`` under the tower directory."""

        for entry in self.files:
            if PurePosixPath(entry.path).name == name:
                return entry
        raise KeyError(f"WEIGHTS.json 里没有 {name}，先运行 job.py weights")


class DimensionMetrics(Record):
    """Agreement with EVA vote means on one dimension (same as the LoRA eval)."""

    srcc: float = Field(ge=-1, le=1)
    plcc: float = Field(ge=-1, le=1)
    mae_1_5: float = Field(ge=0)


class ProbeResult(Record):
    """Val agreement of a ridge probe on one layer set and score."""

    layers: LayerSet
    dimension: Dimension
    feature_dim: int = Field(gt=0)
    alpha: float = Field(gt=0)
    cv_srcc: float = Field(ge=-1, le=1)
    val: DimensionMetrics
    val_srcc_ci95: tuple[float, float]


class ProbeReport(Record):
    """``runs/<run_id>/report.json`` of a ridge probe on cached tower features."""

    schema_version: SchemaVersion
    run_id: RunName
    created_at: AwareDatetime
    features_run_id: RunName
    features_sha256: Sha256
    items_sha256: Sha256
    n_train: int = Field(gt=0)
    n_val: int = Field(gt=0)
    limited: bool
    folds: int = Field(ge=2)
    alphas: tuple[float, ...] = Field(min_length=1)
    bootstrap_samples: int = Field(ge=100)
    seed: int
    results: tuple[ProbeResult, ...] = Field(min_length=1)
