from __future__ import annotations

from pathlib import Path

import pytest
from ptune.prompts import (
    FewShotExample,
    Message,
    PromptError,
    PromptTemplate,
    build_messages,
    find_prompt,
    load_prompt,
    parse_prompt,
)
from ptune.schema import BASELINE_METHODS
from pydantic import ValidationError

from tests.conftest import HERE

PROMPTS = HERE / "prompts"
REPLY = '{"aesthetic":3.5,"technical":3.2}'

VALID = """---
id: demo-score@v1
methods: b1_rubric
media: image
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


def example(image_id: str) -> FewShotExample:
    return FewShotExample(
        image_id=image_id, image_path=f"images/eva/{image_id[6:]}.jpg", reply_text=REPLY
    )


def test_job_prompts_cover_every_baseline_method() -> None:
    files = sorted(PROMPTS.glob("**/*.prompt.md"))
    methods = [m for path in files for m in load_prompt(path).methods]

    assert {p.name for p in files} == {
        "zero-shot-score.prompt.md",
        "rubric-score.prompt.md",
        "few-shot-score.prompt.md",
    }
    assert BASELINE_METHODS <= set(methods)


def test_parse_valid_prompt() -> None:
    prompt = parse_prompt(VALID, "demo")

    assert (prompt.system, prompt.example, prompt.user) == ("细则", None, "请评分。")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "缺少 front matter"),
        ("---\nid: demo-score@v1\n", "没有结束标记"),
        (VALID.replace("media: image\n", ""), "必须且只能有"),
        (VALID.replace("methods: b1_rubric", "methods b1_rubric"), "行无效或重复"),
        (VALID.replace("## user", "## assistant"), "未知或重复的段落"),
        (VALID.split("## user")[0], "缺少 ## user"),
        (VALID.replace("b1_rubric", "b1_rubric, b1_rubric"), "methods 有重复"),
        (VALID.replace("b1_rubric", "b1_rubric, b2_few_shot"), "不能共用"),
        (VALID.replace("b1_rubric", "b2_few_shot"), "必须有带 {i} 编号"),
        (VALID.replace("## user", "## example\n\n示例 {i}\n\n## user"), "不能有"),
        (VALID.replace("请评分。", "共 {k} 张"), "占位符"),
    ],
)
def test_parse_rejects_malformed(text: str, message: str) -> None:
    with pytest.raises(PromptError, match=message):
        parse_prompt(text, "demo")


def test_load_and_find(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="结尾"):
        load_prompt(write_prompt(tmp_path, "demo-score.md", VALID))
    with pytest.raises(PromptError, match="与文件名"):
        load_prompt(write_prompt(tmp_path, "other-score.prompt.md", VALID))
    write_prompt(tmp_path, "v1/demo-score.prompt.md", VALID)
    assert find_prompt("demo-score@v1", tmp_path).user == "请评分。"
    with pytest.raises(PromptError, match="的提示词有 0 份"):
        find_prompt("demo-score@v2", tmp_path)


def test_build_few_shot_messages_number_examples() -> None:
    prompt = find_prompt("few-shot-score@v2", PROMPTS)

    messages = build_messages(
        prompt,
        "photo_000001",
        "images/eva/1.jpg",
        [example("photo_000010"), example("photo_000011")],
    )

    assert [m.role for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1].text.startswith("【示例 1/2】")
    assert messages[3].text.startswith("【示例 2/2】")
    assert messages[2].text == REPLY
    assert messages[-1].images[0].image == "images/eva/1.jpg"
    assert messages[-1].text.startswith("【待评照片】以上 2 张示例到此结束")


def test_build_zero_shot_has_no_system() -> None:
    prompt = find_prompt("zero-shot-score@v2", PROMPTS)

    messages = build_messages(prompt, "photo_000001", "images/eva/1.jpg")

    assert [m.role for m in messages] == ["user"]


@pytest.mark.parametrize(
    ("prompt_id", "examples", "message"),
    [
        ("few-shot-score@v2", [], "需要少样本示例"),
        ("rubric-score@v2", ["photo_000010"], "不接受少样本示例"),
        ("few-shot-score@v2", ["photo_000001"], "就是待评对象"),
    ],
)
def test_build_rejects_misuse(
    prompt_id: str, examples: list[str], message: str
) -> None:
    with pytest.raises(PromptError, match=message):
        build_messages(
            find_prompt(prompt_id, PROMPTS),
            "photo_000001",
            "images/eva/1.jpg",
            [example(i) for i in examples],
        )


def test_build_rejects_video_prompt(tmp_path: Path) -> None:
    write_prompt(
        tmp_path, "demo-score.prompt.md", VALID.replace("media: image", "media: video")
    )

    with pytest.raises(PromptError, match="不是图片提示词"):
        build_messages(
            find_prompt("demo-score@v1", tmp_path), "photo_000001", "images/eva/1.jpg"
        )


def test_assistant_messages_are_text_only() -> None:
    with pytest.raises(ValidationError, match="只能包含文本"):
        Message(
            role="assistant", content=({"type": "image", "image": "images/eva/1.jpg"},)
        )


def test_prompt_template_is_frozen() -> None:
    prompt = parse_prompt(VALID, "demo")

    with pytest.raises(ValidationError):
        prompt.user = "改"  # type: ignore[misc]
    assert isinstance(prompt, PromptTemplate)
