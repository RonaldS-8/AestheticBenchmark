from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from lora_sft.data.samples import (
    COMMENT_MAX_CHARS,
    AestheticOutput,
    Message,
    SampleMismatchError,
    SftSample,
    check_sample_matches_annotation,
)
from lora_sft.data.schema import AnnotationRecord


def set_reply(sft_data: dict[str, Any], reply: dict[str, Any] | str) -> None:
    text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
    sft_data["messages"][-1]["content"] = [{"type": "text", "text": text}]


def valid_reply(**overrides: object) -> dict[str, Any]:
    reply: dict[str, Any] = {
        "aesthetic": 4.3,
        "technical": 2.7,
        "overall": 3.7,
        "comment": "构图干净。",
    }
    reply.update(overrides)
    return reply


def test_fixture_is_valid(sft_data: dict[str, Any]) -> None:
    sample = SftSample.model_validate(sft_data)

    assert sample.target.overall == 3.7
    assert sample.messages[1].images[0].image == "images/photo_000123.jpg"
    assert sample.messages[1].text.startswith("请从美学")


def test_json_round_trip(sft_data: dict[str, Any]) -> None:
    sample = SftSample.model_validate(sft_data)

    assert SftSample.model_validate_json(sample.model_dump_json()) == sample


def test_system_message_is_optional(sft_data: dict[str, Any]) -> None:
    sft_data["messages"] = sft_data["messages"][1:]

    assert len(SftSample.model_validate(sft_data).messages) == 2


@pytest.mark.parametrize(
    "order",
    [
        [1],
        [2, 1],
        [1, 1, 2],
        [0, 2, 1],
        [0, 0, 1, 2],
    ],
)
def test_bad_message_order_rejected(sft_data: dict[str, Any], order: list[int]) -> None:
    messages = sft_data["messages"]
    sft_data["messages"] = [messages[i] for i in order]

    with pytest.raises(ValidationError, match="messages|消息顺序|只能包含文本"):
        SftSample.model_validate(sft_data)


@pytest.mark.parametrize("n_images", [0, 2])
def test_user_needs_exactly_one_image(sft_data: dict[str, Any], n_images: int) -> None:
    image = {"type": "image", "image": "images/photo_000123.jpg"}
    text = {"type": "text", "text": "评分"}
    sft_data["messages"][1]["content"] = [image] * n_images + [text]

    with pytest.raises(ValidationError, match="恰好包含 1 张图片"):
        SftSample.model_validate(sft_data)


def test_image_must_match_id(sft_data: dict[str, Any]) -> None:
    sft_data["messages"][1]["content"][0]["image"] = "images/photo_000999.jpg"

    with pytest.raises(ValidationError, match="与 image_id"):
        SftSample.model_validate(sft_data)


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_non_user_message_cannot_hold_images(role: str) -> None:
    with pytest.raises(ValidationError, match="只能包含文本"):
        Message.model_validate(
            {"role": role, "content": [{"type": "image", "image": "images/x.jpg"}]}
        )


def test_unknown_part_type_rejected() -> None:
    with pytest.raises(ValidationError, match="type"):
        Message.model_validate(
            {"role": "user", "content": [{"type": "audio", "audio": "a.mp3"}]}
        )


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "```json\n" + json.dumps(valid_reply()) + "\n```",
        valid_reply(extra="x"),
        valid_reply(overall=5.1),
        valid_reply(overall=0.9),
        valid_reply(comment=""),
        valid_reply(comment="长" * (COMMENT_MAX_CHARS + 1)),
        {"aesthetic": 4.3, "technical": 2.7, "overall": 3.7},
    ],
)
def test_invalid_assistant_reply_rejected(
    sft_data: dict[str, Any], reply: dict[str, Any] | str
) -> None:
    set_reply(sft_data, reply)

    with pytest.raises(ValidationError, match="不是合法的输出 JSON"):
        SftSample.model_validate(sft_data)


@pytest.mark.parametrize("value", [1.0, 5.0])
def test_score_boundaries_accepted(sft_data: dict[str, Any], value: float) -> None:
    set_reply(sft_data, valid_reply(overall=value))

    assert SftSample.model_validate(sft_data).target.overall == value


def test_comment_length_boundary_accepted(sft_data: dict[str, Any]) -> None:
    set_reply(sft_data, valid_reply(comment="长" * COMMENT_MAX_CHARS))

    assert len(SftSample.model_validate(sft_data).target.comment) == COMMENT_MAX_CHARS


def test_target_needs_one_decimal(sft_data: dict[str, Any]) -> None:
    set_reply(sft_data, valid_reply(aesthetic=4.33))

    with pytest.raises(ValidationError, match="保留一位小数"):
        SftSample.model_validate(sft_data)


def test_reply_must_be_single_text_part(sft_data: dict[str, Any]) -> None:
    part = sft_data["messages"][-1]["content"][0]
    sft_data["messages"][-1]["content"] = [part, {"type": "text", "text": "补充"}]

    with pytest.raises(ValidationError, match="单段 JSON"):
        SftSample.model_validate(sft_data)


def test_llm_comment_needs_review(sft_data: dict[str, Any]) -> None:
    sft_data["comment_provenance"]["reviewed_by"] = None

    with pytest.raises(ValidationError, match="人工复核"):
        SftSample.model_validate(sft_data)


def test_human_comment_needs_no_review(sft_data: dict[str, Any]) -> None:
    sft_data["comment_provenance"] = {
        "method": "human_written",
        "author": "A02",
        "reviewed_by": None,
    }

    assert SftSample.model_validate(sft_data).comment_provenance.author == "A02"


@pytest.mark.parametrize("prompt_id", ["rubric-score", "Rubric@v1", "rubric_score@v1"])
def test_prompt_id_format(sft_data: dict[str, Any], prompt_id: str) -> None:
    sft_data["prompt_id"] = prompt_id

    with pytest.raises(ValidationError, match="prompt_id"):
        SftSample.model_validate(sft_data)


def test_aesthetic_output_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AestheticOutput.model_validate(valid_reply(extra="x"))


def test_sample_matches_annotation(
    sft_data: dict[str, Any], annotation_data: dict[str, Any]
) -> None:
    check_sample_matches_annotation(
        SftSample.model_validate(sft_data),
        AnnotationRecord.model_validate(annotation_data),
    )


def test_sample_for_other_image_rejected(
    sft_data: dict[str, Any], annotation_data: dict[str, Any]
) -> None:
    annotation_data["image_id"] = "photo_000124"
    annotation_data["image_path"] = "images/photo_000124.jpg"

    with pytest.raises(SampleMismatchError, match="标注记录是 photo_000124"):
        check_sample_matches_annotation(
            SftSample.model_validate(sft_data),
            AnnotationRecord.model_validate(annotation_data),
        )


def test_sample_score_must_equal_rounded_mean(
    sft_data: dict[str, Any], annotation_data: dict[str, Any]
) -> None:
    set_reply(sft_data, valid_reply(technical=2.6))

    with pytest.raises(SampleMismatchError, match="technical 应为 2.7"):
        check_sample_matches_annotation(
            SftSample.model_validate(sft_data),
            AnnotationRecord.model_validate(annotation_data),
        )


def test_user_message_can_hold_video() -> None:
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "videos/video_000001.mp4"},
                {"type": "text", "text": "评分"},
            ],
        }
    )

    assert message.videos[0].video == "videos/video_000001.mp4"
    assert message.images == ()


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_non_user_message_cannot_hold_video(role: str) -> None:
    with pytest.raises(ValidationError, match="只能包含文本"):
        Message.model_validate(
            {"role": role, "content": [{"type": "video", "video": "videos/v.mp4"}]}
        )


def test_photo_sft_sample_rejects_video(sft_data: dict[str, Any]) -> None:
    sft_data["messages"][1]["content"].insert(
        0, {"type": "video", "video": "videos/video_000001.mp4"}
    )

    with pytest.raises(ValidationError, match="不能包含视频"):
        SftSample.model_validate(sft_data)
