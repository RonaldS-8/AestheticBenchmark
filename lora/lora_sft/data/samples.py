"""LoRA SFT samples and the model's output contract.

Samples use the chat format accepted by Qwen VL processors: a list of
messages whose ``content`` is a list of ``{"type": "image" | "text"}`` parts.
The assistant reply is the JSON text of ``AestheticOutput``; the same model
parses predictions at evaluation time.

Scores are the primary output and are generated first; the comment is one
short sentence, spot-checked by people but not gated.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from lora_sft.common import (
    ImageId,
    PromptId,
    Record,
    RelativePath,
    SchemaVersion,
    Score,
)
from lora_sft.data.schema import AnnotationRecord

COMMENT_MAX_CHARS = 60


class SampleMismatchError(ValueError):
    """Raised when a sample does not match its annotation record."""


class ImagePart(Record):
    """An image in a message, relative to the data root."""

    type: Literal["image"]
    image: RelativePath


class VideoPart(Record):
    """A video in a message, relative to the data root.

    Frame sampling is a run setting (``VideoSampling``), not part of the
    message, so the same messages work for any frame budget.
    """

    type: Literal["video"]
    video: RelativePath


class TextPart(Record):
    """A text segment in a message."""

    type: Literal["text"]
    text: str = Field(min_length=1)


ContentPart = Annotated[ImagePart | VideoPart | TextPart, Field(discriminator="type")]


class Message(Record):
    """One chat turn."""

    role: Literal["system", "user", "assistant"]
    content: tuple[ContentPart, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_parts(self) -> Message:
        if self.role != "user" and any(
            not isinstance(p, TextPart) for p in self.content
        ):
            raise ValueError(f"{self.role} 消息只能包含文本")
        return self

    @property
    def images(self) -> tuple[ImagePart, ...]:
        """Image parts in order."""

        return tuple(p for p in self.content if isinstance(p, ImagePart))

    @property
    def videos(self) -> tuple[VideoPart, ...]:
        """Video parts in order."""

        return tuple(p for p in self.content if isinstance(p, VideoPart))

    @property
    def text(self) -> str:
        """All text parts joined in order."""

        return "".join(p.text for p in self.content if isinstance(p, TextPart))


class AestheticOutput(Record):
    """JSON object the scoring model must emit, nothing more."""

    aesthetic: Score
    technical: Score
    overall: Score
    comment: str = Field(min_length=1, max_length=COMMENT_MAX_CHARS)


class CommentProvenance(Record):
    """Who wrote the target comment; LLM-merged comments need a reviewer."""

    method: Literal["human_written", "llm_merged"]
    author: str = Field(min_length=1)
    reviewed_by: str | None

    @model_validator(mode="after")
    def _check_review(self) -> CommentProvenance:
        if self.method == "llm_merged" and self.reviewed_by is None:
            raise ValueError("LLM 合成的评语必须经过人工复核")
        return self


class SftSample(Record):
    """One single-image, single-turn scoring sample."""

    schema_version: SchemaVersion
    sample_id: str = Field(min_length=1)
    image_id: ImageId
    prompt_id: PromptId
    messages: tuple[Message, ...] = Field(min_length=2, max_length=3)
    comment_provenance: CommentProvenance

    @model_validator(mode="after")
    def _check_structure(self) -> SftSample:
        roles = tuple(m.role for m in self.messages)
        if roles not in (("user", "assistant"), ("system", "user", "assistant")):
            raise ValueError(f"消息顺序必须是 [system] user assistant，实际 {roles}")
        if self.messages[-2].videos:
            raise ValueError("照片 SFT 样本不能包含视频")
        images = self.messages[-2].images
        if len(images) != 1:
            raise ValueError(f"user 消息必须恰好包含 1 张图片，实际 {len(images)}")
        if PurePosixPath(images[0].image).stem != self.image_id:
            raise ValueError(
                f"图片 {images[0].image!r} 与 image_id {self.image_id!r} 不一致"
            )
        target = self.target
        for name in ("aesthetic", "technical", "overall"):
            value = getattr(target, name)
            if round(value, 1) != value:
                raise ValueError(f"目标分数 {name}={value} 必须保留一位小数")
        return self

    @property
    def target(self) -> AestheticOutput:
        """Parse the assistant reply; raises if it is not valid output JSON."""

        reply = self.messages[-1]
        if len(reply.content) != 1:
            raise ValueError("assistant 回复必须是单段 JSON 文本")
        try:
            return AestheticOutput.model_validate_json(reply.text)
        except ValidationError as exc:
            raise ValueError(f"assistant 回复不是合法的输出 JSON: {exc}") from exc


def check_sample_matches_annotation(
    sample: SftSample, record: AnnotationRecord
) -> None:
    """Ensure the target scores are the annotation means rounded to 0.1."""

    if sample.image_id != record.image_id:
        raise SampleMismatchError(
            f"样本 {sample.sample_id} 对应 {sample.image_id}，"
            f"标注记录是 {record.image_id}"
        )
    target = sample.target
    for name in ("aesthetic", "technical", "overall"):
        expected = round(getattr(record.mean, name), 1)
        if getattr(target, name) != expected:
            raise SampleMismatchError(
                f"样本 {sample.sample_id} 的 {name} 应为 {expected}，"
                f"实际 {getattr(target, name)}"
            )
