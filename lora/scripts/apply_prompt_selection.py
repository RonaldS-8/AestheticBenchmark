"""Switch LoRA training to the prompt chosen by prompt tuning (ADR-0012).

Reads the selection, builds a new SFT sample version with its
``lora_prompt_id`` on the same seed and ratios as the current samples, checks
that train/val/eval ids are unchanged, then rewrites only ``prompt_id`` and
the ``[data]`` paths of the training config. ``--dry-run`` writes nothing;
an existing sample version is never overwritten.

Example::

    python scripts/apply_prompt_selection.py \\
        --selection data/manifests/eva/prompt-tuning/v1-selection.json --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.data.eva_sft import EvaSftPlan, prepare_eva_sft  # noqa: E402
from lora_sft.data.schema import SplitManifest  # noqa: E402
from lora_sft.eval.eva_baseline import PromptSelection  # noqa: E402
from lora_sft.train.config import LoraSftConfig, load_train_config  # noqa: E402
from lora_sft.train.prompt_switch import (  # noqa: E402
    PromptSwitchError,
    check_only_switched,
    check_selection,
    current_manifest,
    prompt_file,
    switched_config_text,
)


def check_same_split(plan: EvaSftPlan, config: LoraSftConfig, repo_root: Path) -> None:
    """The new samples must use exactly the current train/val/eval ids."""

    current = SplitManifest.model_validate_json(
        (repo_root / config.data.split_manifest).read_text(encoding="utf-8")
    )
    for name in ("train", "val", "eval"):
        if set(getattr(plan.split, name)) != set(getattr(current, name)):
            raise PromptSwitchError(f"新数据的 {name} 划分与当前不同")
    if plan.manifest.eval_fingerprint != config.data.eval_fingerprint:
        raise PromptSwitchError("新数据的 eval 指纹与配置不同")


def main(argv: list[str] | None = None) -> None:
    """Preview, then (unless --dry-run) write the samples and the config."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/eva-qwen35-4b-lora-r16.toml")
    )
    parser.add_argument("--version", default="v2")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    selection = PromptSelection.model_validate_json(
        args.selection.read_text(encoding="utf-8")
    )
    config = load_train_config(args.config)
    if not isinstance(config, LoraSftConfig):
        raise TypeError(f"{args.config} 不是 lora_sft 配置")
    check_selection(selection, config)
    prompt_path = prompt_file(selection.lora_prompt_id, REPOSITORY_ROOT / "prompts")
    base = current_manifest(config, REPOSITORY_ROOT)

    def prepare(dry_run: bool) -> EvaSftPlan:
        return prepare_eva_sft(
            args.data_root,
            prompt_path.relative_to(REPOSITORY_ROOT),
            version=args.version,
            seed=base.seed,
            ratios=base.ratios,
            created_at=datetime.now(SHANGHAI_TIMEZONE),
            dry_run=dry_run,
        )

    plan = prepare(dry_run=True)
    check_same_split(plan, config, REPOSITORY_ROOT)
    sample_dir = f"{args.data_root.as_posix()}/manifests/eva/sft/{args.version}"
    old_text = args.config.read_text(encoding="utf-8")
    new_text = switched_config_text(
        old_text,
        prompt_id=selection.lora_prompt_id,
        sample_dir=sample_dir,
        split_version=plan.split.split_version,
    )
    check_only_switched(config, new_text)
    print(
        json.dumps(
            {
                "lora_prompt_id": selection.lora_prompt_id,
                "prompt_file": prompt_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "samples": sample_dir,
                "record_count": plan.manifest.record_count,
                "eval_fingerprint": plan.manifest.eval_fingerprint,
                "config": args.config.as_posix(),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    sys.stdout.writelines(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"{args.config.as_posix()}（当前）",
            tofile=f"{args.config.as_posix()}（切换后）",
        )
    )
    if args.dry_run:
        return
    prepare(dry_run=False)
    args.config.write_text(new_text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
