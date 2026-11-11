"""Run one prompt baseline (B0 / B1 / B2) over the prompt-tuning set.

Run from the job root (the unpacked bundle). Writes ``runs/<run_id>/``:
``config.json``, ``predictions.jsonl`` (appended per image, so the same
command resumes after an interruption), ``report.json`` over the tuning set
and ``report-eval.json`` over its SFT eval-split images.

Example::

    python run_baseline.py --method b1_rubric --prompt-id rubric-score@v2 \\
        --run-id b1-qwen35-4b-tune --device cuda

``--limit N`` scores only the first N images (smoke test; no
``report-eval.json``, and the run cannot be used to pick a prompt).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptune.prompts import find_prompt  # noqa: E402
from ptune.runner import (  # noqa: E402
    ids_fingerprint,
    load_job,
    run_baseline,
    verify_images,
    write_reports,
)
from ptune.schema import EvaBaselineRunConfig, EvaPrediction  # noqa: E402

SHANGHAI = timezone(timedelta(hours=8))


def main(argv: list[str] | None = None) -> int:
    """Score every remaining image, then write and print the reports."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--method", choices=("b0_zero_shot", "b1_rubric", "b2_few_shot"), required=True
    )
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--job", type=Path, default=Path("data/job.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--limit", type=int, help="只评前 N 张，用于冒烟测试")
    args = parser.parse_args(argv)

    root = Path.cwd()
    job = load_job(root, args.job)
    prompt = find_prompt(args.prompt_id, root / "prompts")
    items = job.items if args.limit is None else job.items[: args.limit]
    verify_images(job, [*items, *job.examples])
    config = EvaBaselineRunConfig.model_validate(
        {
            "schema_version": "1.0",
            "scoring": {
                "schema_version": "1.0",
                "run_id": args.run_id,
                "split": "all",
                "ids_fingerprint": ids_fingerprint(i.image_id for i in items),
                "model": job.config.model,
                "adapter": None,
                "prompt_id": args.prompt_id,
                "chat_template_sha256": job.config.chat_template_sha256,
                "preprocess": job.config.preprocess,
                "max_new_tokens": job.config.max_new_tokens,
                "device": args.device,
            },
            "tuning_set": {
                "path": job.config.tuning_set.path,
                "sha256": job.config.tuning_set.sha256,
            },
            "method": args.method,
            "runtime": "transformers",
            "gguf": None,
        }
    )
    run_dir = args.runs_root / args.run_id

    def progress(done: int, total: int, prediction: EvaPrediction) -> None:
        if done % 50 == 0 or done == total or prediction.scores is None:
            status = "ok" if prediction.scores is not None else "解析失败"
            print(
                f"[{done}/{total}] {prediction.image_id} {status} "
                f"{prediction.latency_ms / 1000:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    from ptune.backend import load_complete

    complete = load_complete(job.config, root, args.device)
    count = run_baseline(
        config,
        prompt,
        job,
        items,
        run_dir=run_dir,
        complete=complete,
        on_progress=progress,
    )
    reports = write_reports(
        config, job, items, run_dir=run_dir, now=datetime.now(SHANGHAI)
    )
    print(f"本次评了 {count} 张")
    for report in reports:
        print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
