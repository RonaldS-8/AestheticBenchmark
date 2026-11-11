"""Pick the best prompt from finished prompt-tuning runs (ADR-0012).

Each run needs ``config.json`` and ``report.json``; all runs must have scored
the same tuning set with the same model and runtime. The score is the mean
SRCC of aesthetic and technical; runs below ``--min-parse-rate`` are not
eligible. ``lora_prompt_id`` is the best prompt without few-shot examples,
which LoRA training and evaluation must use. An existing output file is
never overwritten.

Example::

    python scripts/select_best_prompt.py --min-parse-rate 0.95 \\
        --output data/manifests/eva/prompt-tuning/v1-selection.json \\
        data/runs/b0-qwen35-4b-tune data/runs/b1-qwen35-4b-tune \\
        data/runs/b2-qwen35-4b-tune
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
from lora_sft.eval.eva_baseline import (  # noqa: E402
    EvaBaselineRunConfig,
    select_best_prompt,
)
from lora_sft.eval.eva_eval import EvaEvalReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Rank the runs, print the result and write the selection file."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--min-parse-rate", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} 已存在，不覆盖")

    runs = []
    for run_dir in args.runs:
        config = EvaBaselineRunConfig.model_validate_json(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        report = EvaEvalReport.model_validate_json(
            (run_dir / "report.json").read_text(encoding="utf-8")
        )
        runs.append((config, report))
    selection = select_best_prompt(
        runs, min_parse_rate=args.min_parse_rate, now=datetime.now(SHANGHAI_TIMEZONE)
    )
    print("prompt | 方法 | 解析率 | 美学 SRCC | 技术 SRCC | 得分")
    for c in selection.candidates:
        score = (
            "不合格：" + "；".join(c.ineligible.values())
            if c.score is None
            else (f"{c.score:.3f}")
        )
        srcc = [
            f"{c.srcc[d]:.3f}" if d in c.srcc else "-"
            for d in ("aesthetic", "technical")
        ]
        print(
            f"{c.prompt_id} | {c.method.value} | {c.parse_success_rate:.3f} | "
            f"{srcc[0]} | {srcc[1]} | {score}"
        )
    print(f"最佳提示词：{selection.best_prompt_id}")
    print(f"LoRA 用的提示词（最佳单图）：{selection.lora_prompt_id}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        selection.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
