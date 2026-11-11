"""Train a LoRA adapter on EVA dual-score samples (ADR-0009).

Three modes, run in this order on a new machine:

- ``--dry-run``: data checks only, no torch needed. Validates the config,
  the frozen split, eval leakage and every image hash, then prints the plan.
- ``--preflight``: adds the pinned processor (chat-template hash), checks
  that the Gated DeltaNet kernels the config requires import (only reported,
  not enforced, on a machine without CUDA, which cannot train), and encodes
  every train/val sample, so a length or label-mask error stops the job
  before any GPU time is spent. No model weights are loaded.
- default: trains. Single GPU runs directly; multi-GPU runs under torchrun.
  The output directory must not exist unless ``--resume`` is given, which
  continues from the newest complete checkpoint (``lora_sft.train.resume``)
  with the same config and package versions. The kept
  adapter is the best checkpoint by ``selection.metric``; if no checkpoint
  qualifies, no adapter is written and the run fails.

Example::

    python scripts/train_lora_sft.py \\
        --config configs/eva-qwen35-4b-lora-r16.toml \\
        --eval-samples data/manifests/eva/sft/v1/eval.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.eval.eva_eval import INELIGIBLE_SELECTION_SCORE  # noqa: E402
from lora_sft.train.config import LoraSftConfig, load_train_config  # noqa: E402
from lora_sft.train.eva_data import (  # noqa: E402
    EvaTrainData,
    check_no_eval_leak,
    load_eva_samples,
    load_train_data,
    verify_images,
)
from lora_sft.train.resume import (  # noqa: E402
    ResumePoint,
    check_resume_environment,
    find_resume_point,
    source_version,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the training command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-samples", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def check_output_dir(
    output_dir: Path, config: LoraSftConfig, resume: bool
) -> ResumePoint | None:
    """Refuse to overwrite a run; on resume pick the checkpoint to continue.

    Resuming needs the identical config and a complete checkpoint.
    """

    info_path = output_dir / "run_info.json"
    if not resume:
        if output_dir.exists():
            raise FileExistsError(
                f"输出目录已存在，不覆盖: {output_dir}；"
                "中断后续训请加 --resume（train.sh 用 RESUME=1）"
            )
        return None
    if not info_path.is_file():
        raise FileNotFoundError(f"续跑需要已有的 {info_path}")
    stored = json.loads(info_path.read_text(encoding="utf-8"))["config"]
    if LoraSftConfig.model_validate(stored) != config:
        raise ValueError(f"{info_path} 的配置与本次不同，不能续跑")
    return find_resume_point(output_dir)


def record_resume(output_dir: Path, point: ResumePoint, environment: dict) -> None:
    """Note a resume in ``run_info.json`` and as a marker in ``metrics.jsonl``.

    Steps after the checkpoint are trained again, so ``metrics.jsonl`` may
    repeat them; entries after the last marker are the ones that count.
    """

    now = datetime.now(SHANGHAI_TIMEZONE).isoformat()
    info_path = output_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.setdefault("resumes", []).append(
        {
            "at": now,
            "from_checkpoint": point.path.name,
            "from_step": point.step,
            "skipped_incomplete": [list(item) for item in point.skipped],
            "source": source_version(REPOSITORY_ROOT),
            "environment": environment,
        }
    )
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    marker = {"event": "resume", "at": now, "from_step": point.step}
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(marker, ensure_ascii=False) + "\n")


def plan_summary(config: LoraSftConfig, data: EvaTrainData, image_bytes: int) -> dict:
    """What the run will do, printed by every mode."""

    steps_per_epoch = math.ceil(len(data.train) / config.global_batch_size)
    total_steps = steps_per_epoch * config.optim.epochs
    return {
        "run_name": config.run_name,
        "model": f"{config.base_model.model_id}@{config.base_model.revision[:7]}",
        "prompt_id": config.prompt_id,
        "split_version": data.split.split_version,
        "train_count": len(data.train),
        "val_count": len(data.val),
        "eval_count": len(data.split.eval),
        "global_batch_size": config.global_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "evaluations": total_steps // config.runtime.eval_steps,
        "selection_metric": config.selection.metric,
        "image_megabytes": round(image_bytes / 1e6, 1),
        "output_dir": config.runtime.output_dir,
    }


def preflight(config: LoraSftConfig, data: EvaTrainData, data_root: Path) -> dict:
    """Check kernels, load the processor and encode every train/val sample once."""

    from lora_sft.train.qwen_lora import (
        EvaSftCollator,
        load_processor,
        preflight_kernels,
    )

    kernels = preflight_kernels(config.kernels)
    processor = load_processor(config)
    collator = EvaSftCollator(
        processor, data_root, config.preprocess, config.max_seq_len
    )
    lengths: list[int] = []
    first: dict[str, Any] | None = None
    for sample in (*data.train, *data.val):
        item = collator.encode(sample)
        lengths.append(item["input_ids"].numel())
        if first is None:
            supervised = item["labels"][item["labels"] >= 0]
            first = {
                "sample_id": sample.sample_id,
                "supervised_text": processor.tokenizer.decode(supervised),
                "image_grid_thw": item["image_grid_thw"].tolist(),
            }
    return {
        "chat_template_sha256": config.chat_template_sha256,
        "linear_attention_kernels": kernels,
        "max_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 1),
        "max_seq_len": config.max_seq_len,
        "first_sample": first,
    }


def train(
    config: LoraSftConfig,
    data: EvaTrainData,
    *,
    data_root: Path,
    output_dir: Path,
    num_workers: int,
    resume_point: ResumePoint | None,
    summary: dict,
) -> None:
    """Load the model, train, keep the best adapter by ``selection.metric``."""

    from lora_sft.train.qwen_lora import (
        build_trainer,
        check_kernels,
        environment_info,
        load_lora_model,
        load_processor,
    )

    check_kernels(config.kernels)
    environment = environment_info()
    if resume_point is not None:
        stored = json.loads((output_dir / "run_info.json").read_text(encoding="utf-8"))
        check_resume_environment(stored["environment"], environment)
    processor = load_processor(config)
    model = load_lora_model(config)
    trainer = build_trainer(
        config,
        model,
        processor,
        data.train,
        data.val,
        data_root=data_root,
        output_dir=output_dir,
        num_workers=num_workers,
    )
    if trainer.is_world_process_zero() and resume_point is not None:
        record_resume(output_dir, resume_point, environment)
    if trainer.is_world_process_zero() and resume_point is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "config": config.model_dump(mode="json"),
            "started_at": datetime.now(SHANGHAI_TIMEZONE).isoformat(),
            "source": source_version(REPOSITORY_ROOT),
            "environment": environment,
            "plan": summary,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
        (output_dir / "run_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    # An explicit, verified checkpoint: Trainer's own ``True`` takes the newest
    # folder even when it was only half written.
    result = trainer.train(
        resume_from_checkpoint=(
            None if resume_point is None else str(resume_point.path)
        )
    )
    state = trainer.state
    check_selected(config, state.best_metric, state.best_model_checkpoint)
    trainer.save_model(str(output_dir / "adapter"))
    if trainer.is_world_process_zero():
        final = {
            "finished_at": datetime.now(SHANGHAI_TIMEZONE).isoformat(),
            "global_step": result.global_step,
            "train_loss": result.training_loss,
            "selection_metric": config.selection.metric,
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
        }
        (output_dir / "train_result.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def check_selected(
    config: LoraSftConfig, best_metric: float | None, best_checkpoint: str | None
) -> None:
    """Refuse to save an adapter when no checkpoint won the selection."""

    if best_checkpoint is None or best_metric is None:
        raise RuntimeError("训练结束但没有任何保存过的最佳检查点，不写 adapter")
    if (
        config.selection.metric == "srcc_mean"
        and best_metric <= INELIGIBLE_SELECTION_SCORE
    ):
        raise RuntimeError(
            "没有检查点达到选模条件（解析率或 SRCC 不合格），不写 adapter；"
            "原因见 metrics.jsonl 与训练日志"
        )


def main(argv: list[str] | None = None) -> None:
    """Check inputs, then dry-run, preflight or train."""

    args = parse_args(argv)
    config = load_train_config(args.config)
    if not isinstance(config, LoraSftConfig):
        raise TypeError(f"{args.config} 不是 lora_sft 配置")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != config.runtime.num_gpus:
        raise ValueError(
            f"WORLD_SIZE={world_size} 与配置 num_gpus={config.runtime.num_gpus} 不符；"
            "多卡请用 torchrun --nproc_per_node=<num_gpus>"
        )
    data_root = args.data_root.resolve()
    eval_samples_path = args.eval_samples.resolve()
    # Trainer stores checkpoint paths (best_model_checkpoint) as given; a
    # relative output dir keeps them valid when the repo moves between
    # machines or mounts before a resume.
    os.chdir(REPOSITORY_ROOT)
    output_dir = Path(config.runtime.output_dir)
    data = load_train_data(config, REPOSITORY_ROOT)
    eval_samples = load_eva_samples(eval_samples_path, "eval")
    check_no_eval_leak(data, eval_samples)
    image_bytes = verify_images((*data.train, *data.val), data_root)
    resume_point = None
    if not args.preflight and not args.dry_run:
        resume_point = check_output_dir(output_dir, config, args.resume)
    summary = plan_summary(config, data, image_bytes)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **summary}, ensure_ascii=False, indent=2))
        return
    if args.preflight:
        report = preflight(config, data, data_root)
        print(json.dumps({**summary, **report}, ensure_ascii=False, indent=2))
        return
    train(
        config,
        data,
        data_root=data_root,
        output_dir=output_dir,
        num_workers=args.num_workers,
        resume_point=resume_point,
        summary=summary,
    )


if __name__ == "__main__":
    main()
