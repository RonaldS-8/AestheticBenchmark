"""Print one comparison table for finished EVA runs (baselines and LoRA).

Each argument is a run directory (its ``report.json``) or a report file such
as a tuning run's ``report-eval.json``; all reports must cover the same
images (same split and ``ids_fingerprint``).

Example (prompt tuning)::

    python scripts/summarize_eva_runs.py \\
        data/runs/b0-qwen35-4b-tune data/runs/b1-qwen35-4b-tune \\
        data/runs/b2-qwen35-4b-tune

Example (eval split, prompts against LoRA)::

    python scripts/summarize_eva_runs.py \\
        data/runs/b1-qwen35-4b-tune/report-eval.json \\
        data/runs/eva-qwen35-4b-lora-r16-eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.eval.eva_baseline import comparison_table  # noqa: E402
from lora_sft.eval.eva_eval import EvaEvalReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Load each run's report and print the table."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    reports = []
    for target in args.runs:
        path = target if target.suffix == ".json" else target / "report.json"
        if not path.is_file():
            raise SystemExit(f"{path} 不存在，这次运行可能还没跑完")
        reports.append(EvaEvalReport.model_validate_json(path.read_text("utf-8")))
    print(comparison_table(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
