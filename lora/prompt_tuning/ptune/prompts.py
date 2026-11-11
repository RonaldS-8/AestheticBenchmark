"""Prompt files and chat-message assembly (copy of the main repository's).

A prompt file is ``prompts/**/<name>.prompt.md``: front matter with ``id``
(``<name>@v<N>``), ``methods`` and ``media``, then a ``## system`` (optional)
and a ``## user`` section. Few-shot prompts also have an ``## example``
section: the user text of each example turn, so the model can tell examples
from the item to score; ``{i}`` becomes the example number and ``{k}`` the
number of examples. Text before the first section is documentation for
people and is never sent to the model. This job only scores images.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from ptune.schema import (
    FEW_SHOT_METHODS,
    ImageId,
    Method,
    PromptId,
    Record,
    RelativePath,
)

PROMPTS_ROOT = Path("prompts")
PROMPT_SUFFIX = ".prompt.md"
SECTION_NAMES = ("system", "example", "user")
PLACEHOLDERS = ("{i}", "{k}")
FRONT_MATTER_KEYS = frozenset({"id", "methods", "media"})


class PromptError(ValueError):
    """Raised when a prompt file is malformed, missing or misused."""


class ImagePart(Record):
    """An image in a message, relative to the data root."""

    type: Literal["image"]
    image: RelativePath


class TextPart(Record):
    """A text segment in a message."""

    type: Literal["text"]
    text: str = Field(min_length=1)


ContentPart = Annotated[ImagePart | TextPart, Field(discriminator="type")]


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
    def text(self) -> str:
        """All text parts joined in order."""

        return "".join(p.text for p in self.content if isinstance(p, TextPart))


class PromptTemplate(Record):
    """Parsed prompt file."""

    prompt_id: PromptId
    methods: tuple[Method, ...] = Field(min_length=1)
    media: Literal["image", "video"]
    system: str | None = Field(min_length=1)
    example: str | None = Field(min_length=1)
    user: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_methods(self) -> PromptTemplate:
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods 有重复")
        few_shot = {m in FEW_SHOT_METHODS for m in self.methods}
        if len(few_shot) != 1:
            raise ValueError("少样本方法和非少样本方法不能共用一份提示词")
        if self.is_few_shot:
            if self.example is None or "{i}" not in self.example:
                raise ValueError("少样本提示词必须有带 {i} 编号的 ## example 段")
        else:
            if self.example is not None:
                raise ValueError("非少样本提示词不能有 ## example 段")
            texts = (self.system or "", self.user)
            if any(p in t for p in PLACEHOLDERS for t in texts):
                raise ValueError("非少样本提示词不能用 {i} / {k} 占位符")
        return self

    @property
    def is_few_shot(self) -> bool:
        """Whether this prompt expects in-context examples."""

        return self.methods[0] in FEW_SHOT_METHODS


def parse_prompt(text: str, source: str) -> PromptTemplate:
    """Parse prompt file text; ``source`` only labels error messages."""

    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise PromptError(f"{source}: 缺少 front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise PromptError(f"{source}: front matter 没有结束标记") from exc

    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or key in meta:
            raise PromptError(f"{source}: front matter 行无效或重复: {line!r}")
        meta[key] = value.strip()
    if set(meta) != FRONT_MATTER_KEYS:
        raise PromptError(f"{source}: front matter 必须且只能有 id、methods 和 media")

    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines[end + 1 :]:
        if line.startswith("## "):
            title = line[3:].strip()
            if title not in SECTION_NAMES or title in sections:
                raise PromptError(f"{source}: 未知或重复的段落 {title!r}")
            current = sections[title] = []
        elif current is not None:
            current.append(line)
    if "user" not in sections:
        raise PromptError(f"{source}: 缺少 ## user 段落")

    def section(name: str) -> str | None:
        return "\n".join(sections[name]).strip() if name in sections else None

    try:
        return PromptTemplate(
            prompt_id=meta["id"],
            methods=tuple(m.strip() for m in meta["methods"].split(",")),
            media=meta["media"],
            system=section("system"),
            example=section("example"),
            user="\n".join(sections["user"]).strip(),
        )
    except ValidationError as exc:
        raise PromptError(f"{source}: {exc}") from exc


def load_prompt(path: Path) -> PromptTemplate:
    """Load one prompt file and check its id matches the file name."""

    if not path.name.endswith(PROMPT_SUFFIX):
        raise PromptError(f"{path}: 文件名必须以 {PROMPT_SUFFIX} 结尾")
    prompt = parse_prompt(path.read_text(encoding="utf-8"), str(path))
    name = path.name.removesuffix(PROMPT_SUFFIX)
    if prompt.prompt_id.partition("@")[0] != name:
        raise PromptError(f"{path}: id {prompt.prompt_id} 与文件名 {name} 不一致")
    return prompt


def find_prompt(prompt_id: str, root: Path = PROMPTS_ROOT) -> PromptTemplate:
    """Find the single prompt file under ``root`` with exactly this id."""

    name = prompt_id.partition("@")[0]
    matches = [
        prompt
        for path in sorted(root.glob(f"**/{name}{PROMPT_SUFFIX}"))
        if (prompt := load_prompt(path)).prompt_id == prompt_id
    ]
    if len(matches) != 1:
        raise PromptError(f"{root} 下 id 为 {prompt_id} 的提示词有 {len(matches)} 份")
    return matches[0]


class FewShotExample(Record):
    """A labelled image shown before the image to score, with its exact reply."""

    image_id: ImageId
    image_path: RelativePath
    reply_text: str = Field(min_length=2)


def _user_message(path: str, text: str) -> Message:
    return Message(
        role="user",
        content=(ImagePart(type="image", image=path), TextPart(type="text", text=text)),
    )


def build_messages(
    prompt: PromptTemplate,
    image_id: str,
    image_path: str,
    examples: Sequence[FewShotExample] = (),
) -> tuple[Message, ...]:
    """Assemble chat messages for one photo.

    Each example becomes a user turn (its image + the numbered ``## example``
    text) and an assistant turn (its reply JSON), placed before the photo to
    score, whose user turn carries the ``## user`` text.
    """

    if prompt.media != "image":
        raise PromptError(f"{prompt.prompt_id} 不是图片提示词")
    if prompt.is_few_shot != bool(examples):
        raise PromptError(
            f"{prompt.prompt_id} {'需要' if prompt.is_few_shot else '不接受'}少样本示例"
        )
    messages: list[Message] = []
    if prompt.system is not None:
        messages.append(
            Message(role="system", content=(TextPart(type="text", text=prompt.system),))
        )
    total = str(len(examples))
    for number, example in enumerate(examples, 1):
        if example.image_id == image_id:
            raise PromptError(f"示例 {example.image_id} 就是待评对象")
        text = (prompt.example or "").replace("{i}", str(number))
        messages.append(_user_message(example.image_path, text.replace("{k}", total)))
        messages.append(
            Message(
                role="assistant",
                content=(TextPart(type="text", text=example.reply_text),),
            )
        )
    user = prompt.user.replace("{k}", total) if prompt.is_few_shot else prompt.user
    messages.append(_user_message(image_path, user))
    return tuple(messages)
