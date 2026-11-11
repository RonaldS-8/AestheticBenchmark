"""Pick the best prompt from finished tuning runs and print a comparison.

Each run needs ``config.json`` and ``report.json``; all must have scored the
whole tuning set with the same model. Score = mean SRCC of aesthetic and
technical; runs below the job's ``min_parse_rate`` (taken from the LoRA
config) are not eligible. ``lora_prompt_id`` is the best prompt without
few-shot examples, the one LoRA must be trained with. The output file is
never overwritten.

Example::

    python select_best_prompt.py --output runs/selection.json \\
        runs/b0-qwen35-4b-tune runs/b1-qwen35-4b-tune runs/b2-qwen35-4b-tune
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptune.runner import (  # noqa: E402
    comparison_table,
    load_job,
    select_best_prompt,
    write_selection,
)
from ptune.schema import EvaBaselineRunConfig, EvaEvalReport  # noqa: E402

SHANGHAI = timezone(timedelta(hours=8))


def main(argv: list[str] | None = None) -> int:
    """Rank the runs, print the tables and write the selection file."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", type=Path, default=Path("data/job.json"))
    args = parser.parse_args(argv)

    job = load_job(Path.cwd(), args.job)
    runs = []
    for run_dir in args.runs:
        runs.append(
            (
                EvaBaselineRunConfig.model_validate_json(
                    (run_dir / "config.json").read_text(encoding="utf-8")
                ),
                EvaEvalReport.model_validate_json(
                    (run_dir / "report.json").read_text(encoding="utf-8")
                ),
            )
        )
    print("全部调优图片：")
    print(comparison_table([report for _, report in runs]))
    print("其中 SFT eval 集（与 LoRA 评测同一批图）：")
    print(
        comparison_table(
            [
                EvaEvalReport.model_validate_json(
                    (run_dir / "report-eval.json").read_text(encoding="utf-8")
                )
                for run_dir in args.runs
            ]
        )
    )
    selection = select_best_prompt(
        runs, min_parse_rate=job.config.min_parse_rate, now=datetime.now(SHANGHAI)
    )
    for c in selection.candidates:
        verdict = (
            f"得分 {c.score:.3f}"
            if c.score is not None
            else "不合格：" + "；".join(c.ineligible.values())
        )
        print(f"{c.prompt_id}（{c.method.value}）：{verdict}")
    print(f"最佳提示词：{selection.best_prompt_id}")
    print(f"LoRA 用的提示词（最佳单图）：{selection.lora_prompt_id}")
    write_selection(args.output, selection)
    print(f"已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
