"""Prompt files: loading and chat-message assembly.

A prompt file is ``prompts/**/<name>.prompt.md``: front matter with ``id``
(``<name>@v<N>``), ``methods`` and ``media`` (``image`` or ``video``), then a
``## system`` (optional) and a ``## user`` section. Few-shot prompts also
have an ``## example`` section: the user text of each example turn, so the
model can tell examples from the item to score. In few-shot prompts ``{i}``
is replaced by the example number and ``{k}`` by the number of examples.
Text before the first section is documentation for people and is never sent
to the model. Ids are
unique across ``prompts/``; a new version goes into a new folder such as
``prompts/baseline-v3/``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from lora_sft.common import MediaId, PromptId, Record, RelativePath
from lora_sft.data.samples import ImagePart, Message, TextPart, VideoPart
from lora_sft.eval.schema import FEW_SHOT_METHODS, Method

PROMPTS_ROOT = Path("prompts")
PROMPT_SUFFIX = ".prompt.md"
SECTION_NAMES = ("system", "example", "user")
PLACEHOLDERS = ("{i}", "{k}")
FRONT_MATTER_KEYS = frozenset({"id", "methods", "media"})
MEDIA_PREFIX = {"image": "photo_", "video": "video_"}

Media = Literal["image", "video"]


class PromptError(ValueError):
    """Raised when a prompt file is malformed, missing or misused."""


class PromptTemplate(Record):
    """Parsed prompt file."""

    prompt_id: PromptId
    methods: tuple[Method, ...] = Field(min_length=1)
    media: Media
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

    try:
        return PromptTemplate(
            prompt_id=meta["id"],
            methods=tuple(m.strip() for m in meta["methods"].split(",")),
            media=meta["media"],
            system=(
                "\n".join(sections["system"]).strip() if "system" in sections else None
            ),
            example=(
                "\n".join(sections["example"]).strip()
                if "example" in sections
                else None
            ),
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
    """A labelled training item shown to the model before the item to score.

    ``reply_text`` is the assistant turn exactly as the training data writes
    it, so examples and SFT targets cannot drift apart. Its keys are checked
    against the output contract by whoever builds the example.
    """

    media_id: MediaId
    media_path: RelativePath
    reply_text: str = Field(min_length=2)

    @model_validator(mode="after")
    def _check_reply(self) -> FewShotExample:
        try:
            reply = json.loads(self.reply_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"示例 {self.media_id} 的回复不是 JSON") from exc
        if not isinstance(reply, dict):
            raise ValueError(f"示例 {self.media_id} 的回复必须是 JSON 对象")
        return self


def _user_message(media: Media, path: str, text: str) -> Message:
    part = (
        ImagePart(type="image", image=path)
        if media == "image"
        else VideoPart(type="video", video=path)
    )
    return Message(role="user", content=(part, TextPart(type="text", text=text)))


def _check_media(prompt: PromptTemplate, media_id: str) -> None:
    if not media_id.startswith(MEDIA_PREFIX[prompt.media]):
        raise PromptError(f"{prompt.prompt_id} 用于 {prompt.media}，不能评 {media_id}")


def build_messages(
    prompt: PromptTemplate,
    media_id: str,
    media_path: str,
    examples: Sequence[FewShotExample] = (),
) -> tuple[Message, ...]:
    """Assemble chat messages for one photo or video.

    Each example becomes a user turn (its media + the numbered ``## example``
    text) and an assistant turn (its reply JSON), placed before the item to
    score, whose user turn carries the ``## user`` text.
    """

    if prompt.is_few_shot != bool(examples):
        raise PromptError(
            f"{prompt.prompt_id} {'需要' if prompt.is_few_shot else '不接受'}少样本示例"
        )
    _check_media(prompt, media_id)
    messages: list[Message] = []
    if prompt.system is not None:
        messages.append(
            Message(role="system", content=(TextPart(type="text", text=prompt.system),))
        )
    total = str(len(examples))
    for number, example in enumerate(examples, 1):
        _check_media(prompt, example.media_id)
        if example.media_id == media_id:
            raise PromptError(f"示例 {example.media_id} 就是待评对象")
        text = (prompt.example or "").replace("{i}", str(number))
        messages.append(
            _user_message(prompt.media, example.media_path, text.replace("{k}", total))
        )
        messages.append(
            Message(
                role="assistant",
                content=(TextPart(type="text", text=example.reply_text),),
            )
        )
    user = prompt.user.replace("{k}", total) if prompt.is_few_shot else prompt.user
    messages.append(_user_message(prompt.media, media_path, user))
    return tuple(messages)
