from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lora_sft.data.samples import VideoPart
from lora_sft.eval.prompts import (
    FewShotExample,
    PromptError,
    PromptTemplate,
    build_messages,
    find_prompt,
    load_prompt,
    parse_prompt,
)
from lora_sft.eval.schema import Method

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
BASELINE_FILES = sorted((PROMPTS / "baseline-v2").glob("*.prompt.md"))
ALL_FILES = sorted(PROMPTS.glob("**/*.prompt.md"))
FEW_SHOT_NOTE = (
    "对话开头标着【示例】的几轮是人工评好的照片，紧跟的 assistant 回复\n"
    "就是它的人工分数，只用来校准尺度，不需要你再评；标着【待评照片】的\n"
    "最后一张才是你要评分的。\n"
)
REPLY = '{"aesthetic": 3.5, "technical": 3.2}'

VALID = """---
id: demo-score@v1
methods: b1_rubric
media: video
---

说明文字，不发给模型。

## system

细则

## user

请评分。
"""


def write_prompt(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_example(media_id: str, path: str | None = None) -> FewShotExample:
    video = media_id.startswith("video_")
    default = f"videos/{media_id}.mp4" if video else f"images/{media_id}.jpg"
    return FewShotExample(
        media_id=media_id, media_path=path or default, reply_text=REPLY
    )


def image_prompt(tmp_path: Path, few_shot: bool = False) -> PromptTemplate:
    text = VALID.replace("media: video", "media: image")
    if few_shot:
        text = text.replace("b1_rubric", "b2_few_shot").replace(
            "## user", "## example\n\n示例 {i}\n\n## user"
        )
    write_prompt(tmp_path, "demo-score.prompt.md", text)
    return find_prompt("demo-score@v1", tmp_path)


# --- repository prompt files ---


def test_repo_has_baseline_v2_prompts() -> None:
    assert {p.name for p in BASELINE_FILES} == {
        "zero-shot-score.prompt.md",
        "rubric-score.prompt.md",
        "few-shot-score.prompt.md",
    }


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_repo_prompt_is_image_and_two_scores(path: Path) -> None:
    prompt = load_prompt(path)
    text = f"{prompt.system or ''}\n{prompt.user}"

    assert prompt.media == "image"
    for key in ("aesthetic", "technical"):
        assert f'"{key}"' in text or f"`{key}`" in text
    for key in ("overall", "comment"):
        assert key not in text
    assert "1.0–5.0" in text and "保留一位小数" in text
    assert "只输出一个 JSON 对象" in text


def test_every_method_has_exactly_one_prompt() -> None:
    covered = [m for p in ALL_FILES for m in load_prompt(p).methods]

    assert sorted(covered) == sorted(Method)


@pytest.mark.parametrize(
    "prompt_id",
    ["zero-shot-score@v2", "rubric-score@v2", "few-shot-score@v2", "eva-dual-score@v1"],
)
def test_find_repo_prompt(prompt_id: str) -> None:
    assert find_prompt(prompt_id, PROMPTS).prompt_id == prompt_id


def test_few_shot_rubric_matches_b1_rubric() -> None:
    rubric = find_prompt("rubric-score@v2", PROMPTS)
    few_shot = find_prompt("few-shot-score@v2", PROMPTS)

    assert few_shot.system is not None and FEW_SHOT_NOTE in few_shot.system
    assert few_shot.system.replace(FEW_SHOT_NOTE, "") == rubric.system


def test_few_shot_turns_are_labelled() -> None:
    few_shot = find_prompt("few-shot-score@v2", PROMPTS)

    assert few_shot.example is not None
    assert few_shot.example.startswith("【示例 {i}/{k}】")
    assert few_shot.user.startswith("【待评照片】以上 {k} 张示例到此结束")


def test_rubric_defines_both_scores() -> None:
    system = find_prompt("rubric-score@v2", PROMPTS).system

    assert system is not None
    for word in ("### 美学分", "### 技术分", "构图", "噪点", "曝光"):
        assert word in system


def test_zero_shot_has_no_rubric() -> None:
    prompt = find_prompt("zero-shot-score@v2", PROMPTS)

    assert prompt.system is None
    assert "构图" not in prompt.user


# --- parser ---


def test_parse_valid_prompt() -> None:
    prompt = parse_prompt(VALID, "demo")

    assert prompt.media == "video"
    assert prompt.system == "细则"
    assert prompt.user == "请评分。"


def test_system_section_is_optional() -> None:
    text = VALID.replace("## system\n\n细则\n\n", "")

    assert parse_prompt(text, "demo").system is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "缺少 front matter"),
        ("---\nid: demo-score@v1\n", "没有结束标记"),
        (VALID.replace("media: video\n", ""), "必须且只能有"),
        (VALID.replace("methods:", "tags: x\nmethods:"), "必须且只能有"),
        (VALID.replace("methods: b1_rubric", "methods b1_rubric"), "行无效或重复"),
        (VALID.replace("media: video", "media: video\nid: x"), "行无效或重复"),
        (VALID.replace("media: video", "media: audio"), "media"),
        (VALID.replace("## user", "## assistant"), "未知或重复的段落"),
        (VALID + "\n## user\n\n再来一次\n", "未知或重复的段落"),
        (VALID.split("## user")[0], "缺少 ## user"),
        (VALID.replace("细则", ""), "system"),
        (VALID.replace("请评分。", ""), "user"),
        (VALID.replace("b1_rubric", "b9_magic"), "methods"),
        (VALID.replace("b1_rubric", "b1_rubric, b1_rubric"), "methods 有重复"),
        (VALID.replace("b1_rubric", "b1_rubric, b2_few_shot"), "不能共用"),
        (VALID.replace("demo-score@v1", "demo-score"), "prompt_id"),
        (VALID.replace("b1_rubric", "b2_few_shot"), "必须有带 {i} 编号"),
        (
            VALID.replace("b1_rubric", "b2_few_shot").replace(
                "## user", "## example\n\n示例\n\n## user"
            ),
            "必须有带 {i} 编号",
        ),
        (VALID.replace("## user", "## example\n\n示例 {i}\n\n## user"), "不能有"),
        (VALID.replace("请评分。", "共 {k} 张"), "占位符"),
        (VALID.replace("细则", "第 {i} 条"), "占位符"),
    ],
)
def test_parse_rejects_malformed(text: str, message: str) -> None:
    with pytest.raises(PromptError, match=message):
        parse_prompt(text, "demo")


# --- file loading and lookup ---


def test_load_requires_suffix(tmp_path: Path) -> None:
    path = write_prompt(tmp_path, "demo-score.md", VALID)

    with pytest.raises(PromptError, match="结尾"):
        load_prompt(path)


def test_load_requires_matching_name(tmp_path: Path) -> None:
    path = write_prompt(tmp_path, "other-score.prompt.md", VALID)

    with pytest.raises(PromptError, match="与文件名"):
        load_prompt(path)


def test_find_picks_version_by_id(tmp_path: Path) -> None:
    write_prompt(tmp_path, "baseline-v1/demo-score.prompt.md", VALID)
    write_prompt(
        tmp_path,
        "baseline-v2/demo-score.prompt.md",
        VALID.replace("@v1", "@v2").replace("细则", "新细则"),
    )

    assert find_prompt("demo-score@v2", tmp_path).system == "新细则"
    assert find_prompt("demo-score@v1", tmp_path).system == "细则"


@pytest.mark.parametrize(
    ("files", "prompt_id"),
    [
        ([], "demo-score@v1"),
        (["a/demo-score.prompt.md"], "demo-score@v3"),
        (["a/demo-score.prompt.md", "b/demo-score.prompt.md"], "demo-score@v1"),
    ],
)
def test_find_requires_exactly_one(
    tmp_path: Path, files: list[str], prompt_id: str
) -> None:
    for rel in files:
        write_prompt(tmp_path, rel, VALID)

    with pytest.raises(PromptError, match="的提示词有"):
        find_prompt(prompt_id, tmp_path)


# --- few-shot examples ---


@pytest.mark.parametrize("reply", ["not json", "[1, 2]", "3.5"])
def test_few_shot_reply_must_be_json_object(reply: str) -> None:
    with pytest.raises(ValidationError, match="回复"):
        FewShotExample(
            media_id="photo_000010", media_path="images/eva/10.jpg", reply_text=reply
        )


def test_few_shot_example_keeps_dataset_path() -> None:
    example = make_example("photo_000010", "images/eva/10.jpg")

    assert example.media_path == "images/eva/10.jpg"


# --- message assembly ---


def test_build_rubric_image_messages() -> None:
    prompt = find_prompt("rubric-score@v2", PROMPTS)

    messages = build_messages(prompt, "photo_000001", "images/eva/1.jpg")

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].text == prompt.system
    assert messages[1].images[0].image == "images/eva/1.jpg"
    assert messages[1].videos == ()
    assert messages[1].text == prompt.user


def test_build_zero_shot_has_no_system() -> None:
    prompt = find_prompt("zero-shot-score@v2", PROMPTS)

    messages = build_messages(prompt, "photo_000001", "images/eva/1.jpg")

    assert [m.role for m in messages] == ["user"]


def test_build_few_shot_messages() -> None:
    prompt = find_prompt("few-shot-score@v2", PROMPTS)
    examples = [
        make_example("photo_000010", "images/eva/10.jpg"),
        make_example("photo_000011", "images/eva/11.jpg"),
    ]

    messages = build_messages(prompt, "photo_000001", "images/eva/1.jpg", examples)

    assert [m.role for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1].images[0].image == "images/eva/10.jpg"
    assert messages[2].text == REPLY
    assert messages[1].text.startswith("【示例 1/2】")
    assert messages[3].text.startswith("【示例 2/2】")
    assert messages[-1].text.startswith("【待评照片】以上 2 张示例到此结束")
    assert "{" not in messages[-1].text.replace('{"', "")
    assert messages[-1].images[0].image == "images/eva/1.jpg"


def test_video_prompt_inserts_video(tmp_path: Path) -> None:
    write_prompt(tmp_path, "demo-score.prompt.md", VALID)
    prompt = find_prompt("demo-score@v1", tmp_path)

    messages = build_messages(prompt, "video_000001", "videos/video_000001.mp4")

    assert messages[-1].videos == (
        VideoPart(type="video", video="videos/video_000001.mp4"),
    )


def test_few_shot_prompt_needs_examples() -> None:
    prompt = find_prompt("few-shot-score@v2", PROMPTS)

    with pytest.raises(PromptError, match="需要少样本示例"):
        build_messages(prompt, "photo_000001", "images/eva/1.jpg")


def test_non_few_shot_prompt_rejects_examples() -> None:
    prompt = find_prompt("rubric-score@v2", PROMPTS)

    with pytest.raises(PromptError, match="不接受少样本示例"):
        build_messages(
            prompt, "photo_000001", "images/eva/1.jpg", [make_example("photo_000010")]
        )


def test_image_prompt_rejects_video(tmp_path: Path) -> None:
    prompt = image_prompt(tmp_path)

    with pytest.raises(PromptError, match="不能评 video_000001"):
        build_messages(prompt, "video_000001", "videos/video_000001.mp4")


def test_image_prompt_rejects_video_example(tmp_path: Path) -> None:
    prompt = image_prompt(tmp_path, few_shot=True)

    with pytest.raises(PromptError, match="不能评 video_000010"):
        build_messages(
            prompt, "photo_000001", "images/eva/1.jpg", [make_example("video_000010")]
        )


def test_example_cannot_be_the_target() -> None:
    prompt = find_prompt("few-shot-score@v2", PROMPTS)

    with pytest.raises(PromptError, match="就是待评对象"):
        build_messages(
            prompt, "photo_000010", "images/eva/10.jpg", [make_example("photo_000010")]
        )


def test_prompt_template_is_frozen() -> None:
    prompt = PromptTemplate(
        prompt_id="demo-score@v1",
        methods=("b1_rubric",),
        media="image",
        system=None,
        example=None,
        user="评分",
    )

    with pytest.raises(ValidationError):
        prompt.user = "改"  # type: ignore[misc]
