"""Score an EVA split with the base model (B0) or a trained LoRA adapter.

Both use the training config's model, prompt, chat template and pixel
budget, so the comparison differs only in the adapter. Writes
``data/runs/<run_id>/`` (``config.json``, ``predictions.jsonl`` appended per
image so the same command resumes, ``report.json`` at the end).

Example::

    python scripts/eval_eva_sft.py \\
        --config configs/eva-qwen35-4b-lora-r16.toml \\
        --samples data/manifests/eva/sft/v1/val.jsonl --split val \\
        --adapter outputs/eva-qwen35-4b-lora-r16/adapter \\
        --run-id eva-qwen35-4b-lora-r16-val
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.crawl.http import sha256_file  # noqa: E402
from lora_sft.data.schema import eval_fingerprint  # noqa: E402
from lora_sft.eval.eva_eval import (  # noqa: E402
    AdapterRef,
    EvaEvalRunConfig,
    append_prediction,
    build_eva_report,
    parse_eva_reply,
    prepare_eva_run_dir,
    read_predictions,
)
from lora_sft.train.config import LoraSftConfig, load_train_config  # noqa: E402
from lora_sft.train.eva_data import load_eva_samples, verify_images  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the eval command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "eval"), required=True)
    parser.add_argument("--run-id", required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--adapter", type=Path)
    model.add_argument("--base", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="只评前 N 张，用于冒烟测试")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Score every remaining image, then write and print the report."""

    args = parse_args(argv)
    config = load_train_config(args.config)
    if not isinstance(config, LoraSftConfig):
        raise TypeError(f"{args.config} 不是 lora_sft 配置")
    samples = load_eva_samples(args.samples, args.split)
    prompts = {sample.prompt_id for sample in samples}
    if prompts != {config.prompt_id}:
        raise ValueError(
            f"样本用的提示词 {sorted(prompts)} 与配置 {config.prompt_id} 不符"
        )
    if args.limit is not None:
        samples = samples[: args.limit]
    fingerprint = eval_fingerprint(sample.image_id for sample in samples)
    if (
        args.split == "eval"
        and args.limit is None
        and fingerprint != config.data.eval_fingerprint
    ):
        raise ValueError("评测样本的指纹与配置 eval_fingerprint 不符")
    data_root = args.data_root.resolve()
    verify_images(samples, data_root)
    adapter = None
    if args.adapter is not None:
        adapter = AdapterRef(
            train_run_name=config.run_name,
            path=args.adapter.as_posix(),
            weights_sha256=sha256_file(args.adapter / "adapter_model.safetensors"),
        )
    run_config = EvaEvalRunConfig(
        schema_version="1.0",
        run_id=args.run_id,
        split=args.split,
        ids_fingerprint=fingerprint,
        model=config.base_model,
        adapter=adapter,
        prompt_id=config.prompt_id,
        chat_template_sha256=config.chat_template_sha256,
        preprocess=config.preprocess,
        # Same budget as checkpoint selection and prompt tuning (ADR-0012).
        max_new_tokens=config.selection.max_new_tokens,
        device=args.device,
    )
    run_dir = args.runs_root / args.run_id
    done = prepare_eva_run_dir(run_dir, run_config)
    todo = [sample for sample in samples if sample.image_id not in done]
    if todo:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText

        from lora_sft.train.qwen_lora import generate_reply, load_processor

        processor = load_processor(config)
        model = AutoModelForImageTextToText.from_pretrained(
            config.base_model.model_id,
            revision=config.base_model.revision,
            dtype=(
                torch.bfloat16 if config.runtime.precision == "bf16" else torch.float32
            ),
        ).to(args.device)
        if args.adapter is not None:
            model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()
        for number, sample in enumerate(todo, 1):
            text, latency_ms = generate_reply(
                model,
                processor,
                sample,
                data_root=data_root,
                preprocess=config.preprocess,
                max_new_tokens=run_config.max_new_tokens,
            )
            prediction = parse_eva_reply(args.run_id, sample.image_id, text, latency_ms)
            append_prediction(run_dir, prediction)
            if number % 50 == 0 or number == len(todo):
                print(f"{number}/{len(todo)}", file=sys.stderr, flush=True)
    report = build_eva_report(
        run_config,
        read_predictions(run_dir),
        samples,
        now=datetime.now(SHANGHAI_TIMEZONE),
    )
    (run_dir / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
