"""Prepare EVA dual-score weak supervision for Qwen3.5 multimodal SFT.

EVA publishes aesthetic and technical ratings, but no project-style overall
score or written comment.  This module therefore creates a separate first
stage contract with exactly those two targets instead of inventing labels.
"""

from __future__ import annotations

import json
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar

from PIL import Image, UnidentifiedImageError
from pydantic import AwareDatetime, Field, ValidationError, model_validator

from lora_sft.common import (
    ImageId,
    PromptId,
    Record,
    RelativePath,
    SchemaVersion,
    Score,
    Sha256,
)
from lora_sft.crawl.dataset_schema import (
    DatasetImageRecord,
    DatasetScoreRecord,
    ScoreSummary,
)
from lora_sft.crawl.http import sha256_file
from lora_sft.data.samples import Message, TextPart
from lora_sft.data.schema import SplitManifest, eval_fingerprint
from lora_sft.eval.prompts import PromptTemplate, build_messages, load_prompt

QWEN35_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN35_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN35_CHAT_TEMPLATE_SHA256 = (
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
)
EVA_PROMPT_ID = "eva-dual-score@v1"
EVA_OUTPUT_CONTRACT = "eva_dual_score_v1"
DatasetRecordT = TypeVar("DatasetRecordT", DatasetScoreRecord, DatasetImageRecord)


class EvaPreparationError(ValueError):
    """Raised when crawled EVA data cannot form trustworthy SFT samples."""


class EvaDualScoreOutput(Record):
    """The only assistant labels directly supported by EVA."""

    aesthetic: Score
    technical: Score


class EvaSourceLabels(Record):
    """Original EVA aggregates retained beside the converted targets."""

    capture_id: str = Field(min_length=1)
    aesthetic: ScoreSummary
    technical: ScoreSummary

    @model_validator(mode="after")
    def _check_counts(self) -> EvaSourceLabels:
        if self.aesthetic.count != self.technical.count:
            raise ValueError("EVA 美学与技术分的投票人数必须一致")
        return self


class EvaSftSample(Record):
    """One Qwen-compatible EVA message sequence plus source provenance."""

    schema_version: SchemaVersion
    sample_id: str = Field(pattern=r"^eva-photo_\d{6,}$")
    image_id: ImageId
    split: Literal["train", "val", "eval"]
    prompt_id: PromptId
    source_image_id: str = Field(pattern=r"^\d+$")
    image_path: RelativePath
    image_sha256: Sha256
    image_decode_warnings: tuple[str, ...]
    source_labels: EvaSourceLabels
    messages: tuple[Message, ...] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def _check_structure(self) -> EvaSftSample:
        expected_image_id = eva_image_id(self.source_image_id)
        if self.image_id != expected_image_id:
            raise ValueError(
                f"EVA source_image_id={self.source_image_id} 应映射为 "
                f"{expected_image_id}，实际 {self.image_id}"
            )
        if self.sample_id != f"eva-{self.image_id}":
            raise ValueError("EVA sample_id 必须是 eva-<image_id>")
        roles = tuple(message.role for message in self.messages)
        if roles not in (("system", "user", "assistant"), ("user", "assistant")):
            raise ValueError(f"EVA 消息顺序必须是 [system]/user/assistant: {roles}")
        user = self.messages[-2]
        if user.videos or len(user.images) != 1:
            raise ValueError("EVA user 消息必须恰好包含一张图片且不能有视频")
        if user.images[0].image != self.image_path:
            raise ValueError("EVA user 消息图片路径与 image_path 不一致")
        target = self.target
        for name in ("aesthetic", "technical"):
            value = getattr(target, name)
            if round(value, 1) != value:
                raise ValueError(f"EVA 目标分数 {name}={value} 必须保留一位小数")
        return self

    @property
    def target(self) -> EvaDualScoreOutput:
        """Strictly parse the assistant's two-score JSON reply."""

        reply = self.messages[-1]
        if len(reply.content) != 1:
            raise ValueError("EVA assistant 回复必须是单段 JSON 文本")
        try:
            return EvaDualScoreOutput.model_validate_json(reply.text)
        except ValidationError as exc:
            raise ValueError(f"EVA assistant 回复不是合法双分 JSON: {exc}") from exc


class Qwen35ImageInputSpec(Record):
    """Pinned official Qwen3.5-4B image-message and processor contract."""

    model_id: Literal["Qwen/Qwen3.5-4B"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    chat_template_sha256: Sha256
    enable_thinking: Literal[False]
    image_part_type: Literal["image"]
    image_path_key: Literal["image"]
    processor_class: Literal["Qwen3VLProcessor"]
    image_processor_type: Literal["Qwen2VLImageProcessorFast"]
    shortest_edge: int = Field(gt=0)
    longest_edge: int = Field(gt=0)
    patch_size: int = Field(gt=0)
    temporal_patch_size: int = Field(gt=0)
    merge_size: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_edges(self) -> Qwen35ImageInputSpec:
        if self.shortest_edge > self.longest_edge:
            raise ValueError("shortest_edge 不能大于 longest_edge")
        return self


class SplitRatios(Record):
    """Fractions used by the deterministic EVA split."""

    train: float = Field(gt=0, lt=1)
    val: float = Field(gt=0, lt=1)
    eval: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _check_sum(self) -> SplitRatios:
        if abs(self.train + self.val + self.eval - 1.0) > 1e-9:
            raise ValueError("train/val/eval 比例之和必须为 1")
        return self


class EvaSftManifest(Record):
    """Reproducibility record for one immutable EVA SFT preparation."""

    schema_version: SchemaVersion
    dataset: Literal["eva"]
    version: str = Field(pattern=r"^v\d+$")
    created_at: AwareDatetime
    source_scores_path: RelativePath
    source_scores_sha256: Sha256
    source_images_path: RelativePath
    source_images_sha256: Sha256
    prompt_path: RelativePath
    prompt_sha256: Sha256
    prompt_id: PromptId
    output_contract: Literal["eva_dual_score_v1"]
    assistant_loss_only: Literal[True]
    qwen_input: Qwen35ImageInputSpec
    seed: int = Field(ge=0)
    ratios: SplitRatios
    split_path: RelativePath
    train_path: RelativePath
    val_path: RelativePath
    eval_path: RelativePath
    record_count: int = Field(gt=0)
    train_count: int = Field(gt=0)
    val_count: int = Field(gt=0)
    eval_count: int = Field(gt=0)
    eval_fingerprint: Sha256
    decode_warning_count: int = Field(ge=0)
    decode_warning_image_ids: tuple[ImageId, ...]

    @model_validator(mode="after")
    def _check_counts(self) -> EvaSftManifest:
        if self.train_count + self.val_count + self.eval_count != self.record_count:
            raise ValueError("EVA train/val/eval 数量之和与总数不一致")
        if self.decode_warning_count != len(self.decode_warning_image_ids):
            raise ValueError("EVA 解码警告数量与图片 id 列表不一致")
        return self


@dataclass(frozen=True)
class EvaSftPlan:
    """Validated in-memory result returned by dry-run and write modes."""

    samples: tuple[EvaSftSample, ...]
    split: SplitManifest
    manifest: EvaSftManifest
    output_dir: Path


def eva_image_id(source_image_id: str) -> str:
    """Map an EVA/AVA numeric id to the repository photo-id namespace."""

    if not source_image_id.isdigit():
        raise EvaPreparationError(f"EVA 图片 id 不是数字: {source_image_id!r}")
    return f"photo_{int(source_image_id):06d}"


def score_to_project_scale(summary: ScoreSummary) -> float:
    """Linearly map a source mean to 1-5 and round the SFT target to 0.1."""

    position = (summary.mean - summary.scale_min) / (
        summary.scale_max - summary.scale_min
    )
    return round(1.0 + 4.0 * position, 1)


def _read_jsonl(path: Path, model: type[DatasetRecordT]) -> list[DatasetRecordT]:
    records: list[DatasetRecordT] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise EvaPreparationError(f"{path} 第 {line_number} 行为空")
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise EvaPreparationError(
                f"{path} 第 {line_number} 行不符合 schema: {exc}"
            ) from exc
    if not records:
        raise EvaPreparationError(f"{path} 没有记录")
    return records


def _index_records(
    records: Sequence[DatasetRecordT], name: str
) -> dict[str, DatasetRecordT]:
    indexed: dict[str, DatasetRecordT] = {}
    for record in records:
        if record.dataset != "eva":
            raise EvaPreparationError(f"{name} 含非 EVA 记录: {record.dataset}")
        if record.source_image_id in indexed:
            raise EvaPreparationError(
                f"{name} 的 source_image_id 重复: {record.source_image_id}"
            )
        indexed[record.source_image_id] = record
    return indexed


def _verify_image(data_root: Path, image: DatasetImageRecord) -> tuple[str, ...]:
    path = data_root / image.image_path
    if not path.is_file():
        raise EvaPreparationError(f"EVA 图片不存在: {path}")
    digest = sha256_file(path)
    if digest != image.sha256:
        raise EvaPreparationError(f"EVA 图片哈希与清单不符: {path}")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Image.open(path) as opened:
                if opened.format != "JPEG":
                    raise EvaPreparationError(f"EVA 图片不是 JPEG: {path}")
                opened.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise EvaPreparationError(f"EVA 图片无法由 Pillow 解码: {path}: {exc}") from exc
    return tuple(dict.fromkeys(str(item.message) for item in caught))


def _make_sample(
    score: DatasetScoreRecord,
    image: DatasetImageRecord,
    split: Literal["train", "val", "eval"],
    prompt: PromptTemplate,
    image_decode_warnings: tuple[str, ...],
) -> EvaSftSample:
    image_id = eva_image_id(score.source_image_id)
    target = EvaDualScoreOutput(
        aesthetic=score_to_project_scale(score.aesthetic),
        technical=score_to_project_scale(score.technical),
    )
    messages = build_messages(prompt, image_id, image.image_path)
    assistant = Message(
        role="assistant",
        content=(
            TextPart(
                type="text",
                text=json.dumps(
                    target.model_dump(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        ),
    )
    return EvaSftSample(
        schema_version="1.0",
        sample_id=f"eva-{image_id}",
        image_id=image_id,
        split=split,
        prompt_id=prompt.prompt_id,
        source_image_id=score.source_image_id,
        image_path=image.image_path,
        image_sha256=image.sha256,
        image_decode_warnings=image_decode_warnings,
        source_labels=EvaSourceLabels(
            capture_id=score.capture_id,
            aesthetic=score.aesthetic,
            technical=score.technical,
        ),
        messages=(*messages, assistant),
    )


def _split_ids(
    image_ids: list[str], seed: int, ratios: SplitRatios
) -> dict[str, tuple[str, ...]]:
    if len(image_ids) < 3:
        raise EvaPreparationError("EVA 至少需要 3 张图片才能划分 train/val/eval")
    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * ratios.val))
    eval_count = max(1, round(len(shuffled) * ratios.eval))
    train_count = len(shuffled) - val_count - eval_count
    if train_count < 1:
        raise EvaPreparationError("划分后 train 为空，请调整比例")
    return {
        "train": tuple(sorted(shuffled[:train_count])),
        "val": tuple(sorted(shuffled[train_count : train_count + val_count])),
        "eval": tuple(sorted(shuffled[train_count + val_count :])),
    }


def prepare_eva_sft(
    data_root: Path,
    prompt_path: Path,
    *,
    version: str,
    seed: int,
    ratios: SplitRatios,
    created_at: datetime,
    dry_run: bool,
) -> EvaSftPlan:
    """Validate, split and optionally write the first EVA SFT dataset."""

    scores_path = data_root / "manifests" / "eva" / "scores.jsonl"
    images_path = data_root / "manifests" / "eva" / "images.jsonl"
    scores = _read_jsonl(scores_path, DatasetScoreRecord)
    images = _read_jsonl(images_path, DatasetImageRecord)
    score_by_id = _index_records(scores, "EVA 评分清单")
    image_by_id = _index_records(images, "EVA 图片清单")
    if score_by_id.keys() != image_by_id.keys():
        missing = sorted(score_by_id.keys() - image_by_id.keys(), key=int)[:5]
        extra = sorted(image_by_id.keys() - score_by_id.keys(), key=int)[:5]
        raise EvaPreparationError(
            f"EVA 评分与图片清单不一致: missing={missing}, extra={extra}"
        )
    hashes = [record.sha256 for record in image_by_id.values()]
    if len(set(hashes)) != len(hashes):
        raise EvaPreparationError("EVA 图片清单含重复 SHA-256")
    decode_warnings = {
        source_id: _verify_image(data_root, image)
        for source_id, image in image_by_id.items()
    }

    prompt = load_prompt(prompt_path)
    # LoRA uses the best single-image prompt from prompt tuning (ADR-0012).
    if prompt.media != "image" or prompt.is_few_shot:
        raise EvaPreparationError(
            f"{prompt.prompt_id} 不能用于 EVA SFT：必须是图片、不带少样本示例的提示词"
        )

    photo_to_source = {eva_image_id(source_id): source_id for source_id in score_by_id}
    split_ids = _split_ids(list(photo_to_source), seed, ratios)
    split = SplitManifest(
        schema_version="1.0",
        split_version=f"split-{version}",
        created_at=created_at,
        seed=seed,
        train=split_ids["train"],
        val=split_ids["val"],
        eval=split_ids["eval"],
        eval_fingerprint=eval_fingerprint(split_ids["eval"]),
    )
    split_by_id = {
        image_id: name for name, ids in split_ids.items() for image_id in ids
    }
    samples = tuple(
        _make_sample(
            score_by_id[source_id],
            image_by_id[source_id],
            split_by_id[image_id],  # type: ignore[arg-type]
            prompt,
            decode_warnings[source_id],
        )
        for image_id, source_id in sorted(photo_to_source.items())
    )

    relative_root = f"manifests/eva/sft/{version}"
    output_dir = data_root / relative_root
    manifest = EvaSftManifest(
        schema_version="1.0",
        dataset="eva",
        version=version,
        created_at=created_at,
        source_scores_path="manifests/eva/scores.jsonl",
        source_scores_sha256=sha256_file(scores_path),
        source_images_path="manifests/eva/images.jsonl",
        source_images_sha256=sha256_file(images_path),
        prompt_path=prompt_path.as_posix(),
        prompt_sha256=sha256_file(prompt_path),
        prompt_id=prompt.prompt_id,
        output_contract=EVA_OUTPUT_CONTRACT,
        assistant_loss_only=True,
        qwen_input=Qwen35ImageInputSpec(
            model_id=QWEN35_MODEL_ID,
            revision=QWEN35_REVISION,
            chat_template_sha256=QWEN35_CHAT_TEMPLATE_SHA256,
            enable_thinking=False,
            image_part_type="image",
            image_path_key="image",
            processor_class="Qwen3VLProcessor",
            image_processor_type="Qwen2VLImageProcessorFast",
            shortest_edge=65_536,
            longest_edge=16_777_216,
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
        ),
        seed=seed,
        ratios=ratios,
        split_path=f"{relative_root}/split-{version}.json",
        train_path=f"{relative_root}/train.jsonl",
        val_path=f"{relative_root}/val.jsonl",
        eval_path=f"{relative_root}/eval.jsonl",
        record_count=len(samples),
        train_count=len(split.train),
        val_count=len(split.val),
        eval_count=len(split.eval),
        eval_fingerprint=split.eval_fingerprint,
        decode_warning_count=sum(bool(items) for items in decode_warnings.values()),
        decode_warning_image_ids=tuple(
            sorted(
                (
                    eva_image_id(source_id)
                    for source_id, items in decode_warnings.items()
                    if items
                )
            )
        ),
    )
    plan = EvaSftPlan(samples, split, manifest, output_dir)
    if dry_run:
        return plan
    if output_dir.exists():
        raise FileExistsError(f"EVA SFT 目标目录已存在，不覆盖: {output_dir}")
    output_dir.mkdir(parents=True)
    grouped = {
        name: [sample for sample in samples if sample.split == name]
        for name in ("train", "val", "eval")
    }
    for name, records in grouped.items():
        (output_dir / f"{name}.jsonl").write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
    (output_dir / f"split-{version}.json").write_text(
        split.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return plan
