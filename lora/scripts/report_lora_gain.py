"""Report how much LoRA adds over its prompt alone, on the SFT eval split.

The base is the prompt-tuning run of the selected LoRA prompt, read from its
``report-eval.json``; the LoRA run is ``scripts/eval_eva_sft.py`` on the eval
split with the same prompt. Writes ``<lora-run>/gain.json`` (never
overwritten).

Example::

    python scripts/report_lora_gain.py \\
        --selection data/manifests/eva/prompt-tuning/v1-selection.json \\
        --base-run data/runs/b1-qwen35-4b-tune \\
        --lora-run data/runs/eva-qwen35-4b-lora-r16-eval
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
    PromptSelection,
    lora_gain,
)
from lora_sft.eval.eva_eval import EvaEvalReport, EvaEvalRunConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Check the comparison is fair, print the gain and write it."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--lora-run", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.lora_run / "gain.json"
    if output.exists():
        raise SystemExit(f"{output} 已存在，不覆盖")

    def read(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    gain = lora_gain(
        PromptSelection.model_validate_json(read(args.selection)),
        EvaBaselineRunConfig.model_validate_json(read(args.base_run / "config.json")),
        EvaEvalReport.model_validate_json(read(args.base_run / "report-eval.json")),
        EvaEvalRunConfig.model_validate_json(read(args.lora_run / "config.json")),
        EvaEvalReport.model_validate_json(read(args.lora_run / "report.json")),
        now=datetime.now(SHANGHAI_TIMEZONE),
    )
    print(
        f"提示词 {gain.prompt_id}，eval 集，解析率 "
        f"{gain.base_parse_success_rate:.3f} → {gain.lora_parse_success_rate:.3f}"
    )
    print("维度 | SRCC 基座 → LoRA（提升） | PLCC 基座 → LoRA（提升）")
    for dim, g in gain.dimensions.items():
        print(
            f"{dim} | {g.base_srcc:.3f} → {g.lora_srcc:.3f} ({g.delta_srcc:+.3f}) | "
            f"{g.base_plcc:.3f} → {g.lora_plcc:.3f} ({g.delta_plcc:+.3f})"
        )
    output.write_text(
        gain.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
