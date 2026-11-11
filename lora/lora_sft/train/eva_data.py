"""Load and check EVA SFT samples for a LoRA run, without importing torch.

Everything here runs on a CPU-only machine so the same checks guard both the
local dry run and the cloud job: the run config must point at the frozen EVA
split, sample files must agree with it, and the eval split must never reach
training. Label masking is also defined here so it can be tested without a
model: only the assistant reply (plus the chat template's end-of-turn) is
supervised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from lora_sft.crawl.http import sha256_file
from lora_sft.data.eva_sft import EvaSftSample
from lora_sft.data.samples import ImagePart, TextPart
from lora_sft.data.schema import SplitManifest
from lora_sft.train.config import LoraSftConfig

IGNORE_INDEX = -100
SplitName = Literal["train", "val", "eval"]


class TrainDataError(ValueError):
    """Raised when training inputs disagree with the run config or each other."""


@dataclass(frozen=True)
class EvaTrainData:
    """Validated train/val samples plus the split they were checked against."""

    split: SplitManifest
    train: tuple[EvaSftSample, ...]
    val: tuple[EvaSftSample, ...]


def load_eva_samples(path: Path, split: SplitName) -> tuple[EvaSftSample, ...]:
    """Strictly read one EVA SFT JSONL file whose records all belong to ``split``."""

    if not path.is_file():
        raise TrainDataError(f"样本文件不存在: {path}")
    samples: list[EvaSftSample] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line:
            raise TrainDataError(f"{path} 第 {number} 行为空")
        try:
            sample = EvaSftSample.model_validate_json(line)
        except ValidationError as exc:
            raise TrainDataError(f"{path} 第 {number} 行不符合 schema: {exc}") from exc
        if sample.split != split:
            raise TrainDataError(
                f"{path} 第 {number} 行 split={sample.split}，应为 {split}"
            )
        samples.append(sample)
    if not samples:
        raise TrainDataError(f"{path} 没有样本")
    ids = [sample.image_id for sample in samples]
    if len(set(ids)) != len(ids):
        raise TrainDataError(f"{path} 含重复 image_id")
    return tuple(samples)


def _check_against_split(
    samples: Sequence[EvaSftSample], expected: Sequence[str], name: str
) -> None:
    actual = {sample.image_id for sample in samples}
    if actual != set(expected):
        missing = sorted(set(expected) - actual)[:5]
        extra = sorted(actual - set(expected))[:5]
        raise TrainDataError(
            f"{name} 样本与划分文件不一致: missing={missing}, extra={extra}"
        )


def load_train_data(config: LoraSftConfig, repo_root: Path) -> EvaTrainData:
    """Load train/val samples and prove they match the frozen split.

    Fails when the split version or eval fingerprint differs from the config,
    when sample ids differ from the split, when the prompt differs from the
    config, or when any train/val image shares a file hash with an eval image.
    """

    split_path = repo_root / config.data.split_manifest
    if not split_path.is_file():
        raise TrainDataError(f"划分文件不存在: {split_path}")
    split = SplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))
    if split.split_version != config.data.split_version:
        raise TrainDataError(
            f"划分版本 {split.split_version} 与配置 {config.data.split_version} 不符"
        )
    if split.eval_fingerprint != config.data.eval_fingerprint:
        raise TrainDataError("划分文件的 eval_fingerprint 与配置不符，评测集被改动")
    train = load_eva_samples(repo_root / config.data.train_samples, "train")
    val = load_eva_samples(repo_root / config.data.val_samples, "val")
    _check_against_split(train, split.train, "train")
    _check_against_split(val, split.val, "val")
    for sample in (*train, *val):
        if sample.prompt_id != config.prompt_id:
            raise TrainDataError(
                f"{sample.sample_id} 的提示词 {sample.prompt_id} "
                f"与配置 {config.prompt_id} 不符"
            )
    return EvaTrainData(split=split, train=train, val=val)


def check_no_eval_leak(
    data: EvaTrainData, eval_samples: Sequence[EvaSftSample]
) -> None:
    """Reject any train/val image whose id or SHA-256 is in the eval split."""

    _check_against_split(eval_samples, data.split.eval, "eval")
    eval_ids = {sample.image_id for sample in eval_samples}
    eval_hashes = {sample.image_sha256: sample.image_id for sample in eval_samples}
    for sample in (*data.train, *data.val):
        if sample.image_id in eval_ids:
            raise TrainDataError(f"{sample.image_id} 出现在评测集中")
        twin = eval_hashes.get(sample.image_sha256)
        if twin is not None:
            raise TrainDataError(f"{sample.image_id} 与评测图片 {twin} 是同一文件")


def verify_images(samples: Sequence[EvaSftSample], data_root: Path) -> int:
    """Check every image exists with the recorded SHA-256; return bytes read."""

    total = 0
    for sample in samples:
        path = data_root / sample.image_path
        if not path.is_file():
            raise TrainDataError(f"图片不存在: {path}")
        if sha256_file(path) != sample.image_sha256:
            raise TrainDataError(f"图片哈希与样本不符: {path}")
        total += path.stat().st_size
    return total


def chat_messages(sample: EvaSftSample, *, include_reply: bool) -> list[dict[str, Any]]:
    """Convert a sample to the Hugging Face chat-template message format.

    The image item carries only ``type`` because pixels are passed to the
    processor separately; the template renders the vision placeholder.
    """

    messages = sample.messages if include_reply else sample.messages[:-1]
    converted: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, str]] = []
        for part in message.content:
            if isinstance(part, ImagePart):
                content.append({"type": "image"})
            elif isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                raise TrainDataError(f"{sample.sample_id} 含不支持的消息内容")
        converted.append({"role": message.role, "content": content})
    return converted


def completion_text(prompt_text: str, full_text: str, reply: str) -> str:
    """Return the supervised tail of ``full_text`` after the generation prompt.

    The rendered conversation must start with the rendered generation prompt
    and the tail must start with the assistant reply; otherwise the template
    does not produce the same prefix at train and inference time.
    """

    if not full_text.startswith(prompt_text):
        raise TrainDataError("对话模板的完整渲染不以生成提示开头，训练与推理前缀不一致")
    tail = full_text[len(prompt_text) :]
    if not tail.startswith(reply):
        raise TrainDataError(f"监督部分不以 assistant 回复开头: {tail[:80]!r}")
    if tail == reply:
        raise TrainDataError("监督部分没有结束标记，模型学不会停止")
    return tail


def mask_labels(input_ids: Sequence[int], completion_ids: Sequence[int]) -> list[int]:
    """Labels that supervise only the trailing ``completion_ids`` tokens."""

    count = len(completion_ids)
    if count == 0:
        raise TrainDataError("监督 token 为空")
    if count >= len(input_ids):
        raise TrainDataError("监督 token 不少于整条序列，提示部分丢失")
    if list(input_ids[-count:]) != list(completion_ids):
        raise TrainDataError("序列末尾与单独分词的回复不一致，无法只对回复计算损失")
    return [IGNORE_INDEX] * (len(input_ids) - count) + list(completion_ids)
