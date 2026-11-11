"""Switch LoRA training to the prompt chosen by prompt tuning (ADR-0012).

Prompt tuning writes a ``PromptSelection`` whose ``lora_prompt_id`` is the
prompt LoRA must be trained and evaluated with. Switching means: build a new
SFT sample version with that prompt on the *same* split, then point the run
config's ``prompt_id`` and ``[data]`` paths at it. Nothing else in the config
may change, so the gain over the prompt alone stays a fair comparison.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath

from lora_sft.data.eva_sft import EvaSftManifest
from lora_sft.eval.eva_baseline import PromptSelection
from lora_sft.eval.prompts import PROMPT_SUFFIX, load_prompt
from lora_sft.train.config import LoraSftConfig

# Config keys the switch rewrites; each must appear exactly once.
SWITCHED_KEYS = (
    "prompt_id",
    "split_manifest",
    "split_version",
    "train_samples",
    "val_samples",
)


class PromptSwitchError(ValueError):
    """Raised when a selection cannot be applied to a training config."""


def check_selection(selection: PromptSelection, config: LoraSftConfig) -> None:
    """The selection must come from the config's model on transformers."""

    if selection.model != config.base_model:
        raise PromptSwitchError(
            f"提示词选择用的模型 {selection.model.model_id}@"
            f"{selection.model.revision[:7]} 与训练配置不同"
        )
    if selection.runtime != "transformers":
        raise PromptSwitchError("LoRA 在 transformers 上评测，提示词选择也必须是")
    if config.prompt_id == selection.lora_prompt_id:
        raise PromptSwitchError(f"训练配置已经在用 {config.prompt_id}，不用再切换")


def prompt_file(prompt_id: str, root: Path) -> Path:
    """The single prompt file under ``root`` whose front matter has this id."""

    name = prompt_id.partition("@")[0]
    matches = [
        path
        for path in sorted(root.glob(f"**/{name}{PROMPT_SUFFIX}"))
        if load_prompt(path).prompt_id == prompt_id
    ]
    if len(matches) != 1:
        raise PromptSwitchError(
            f"{root} 下 id 为 {prompt_id} 的提示词有 {len(matches)} 份"
        )
    return matches[0]


def current_manifest(config: LoraSftConfig, repo_root: Path) -> EvaSftManifest:
    """The SFT manifest next to the config's training samples."""

    path = repo_root / PurePosixPath(config.data.train_samples).parent / "manifest.json"
    if not path.is_file():
        raise PromptSwitchError(f"找不到当前 SFT 数据的清单: {path}")
    return EvaSftManifest.model_validate_json(path.read_text(encoding="utf-8"))


def switched_config_text(
    text: str, *, prompt_id: str, sample_dir: str, split_version: str
) -> str:
    """Rewrite only the switched keys in the TOML text, keeping comments."""

    values = {
        "prompt_id": prompt_id,
        "split_manifest": f"{sample_dir}/{split_version}.json",
        "split_version": split_version,
        "train_samples": f"{sample_dir}/train.jsonl",
        "val_samples": f"{sample_dir}/val.jsonl",
    }
    for key in SWITCHED_KEYS:
        pattern = re.compile(rf'^{key} = "[^"\n]*"$', re.MULTILINE)
        text, count = pattern.subn(f'{key} = "{values[key]}"', text)
        if count != 1:
            raise PromptSwitchError(f"配置里 {key} 应恰好出现一次，实际 {count} 次")
    return text


def check_only_switched(old: LoraSftConfig, new_text: str) -> LoraSftConfig:
    """Parse the rewritten config and prove nothing else changed."""

    new = LoraSftConfig.model_validate(tomllib.loads(new_text))
    if new.data.eval_fingerprint != old.data.eval_fingerprint:
        raise PromptSwitchError("切换后 eval_fingerprint 变了")

    def rest(config: LoraSftConfig) -> dict:
        dump = config.model_dump(mode="json")
        dump.pop("prompt_id")
        for key in SWITCHED_KEYS[1:]:
            dump["data"].pop(key)
        return dump

    if rest(new) != rest(old):
        raise PromptSwitchError("切换只能改提示词和数据路径，配置其他部分不能变")
    return new
