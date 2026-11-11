"""Frozen Qwen3.5-4B vision tower + score heads: every step of the job.

Steps, in order (``run.sh`` chains 2-4, and 5-6 with ``EVAL=1``)::

    python job.py weights        # 1. cut the tower out of the base model
    python job.py features --run-id qwen35-4b-tower-head-features
    python job.py probe --features qwen35-4b-tower-head-features \\
        --run-id qwen35-4b-tower-head-probe
    python job.py sweep --features qwen35-4b-tower-head-features \\
        --prefix qwen35-4b-tower-head --lrs 1e-4 3e-4 1e-3 --seeds 1 2 3
    python job.py select --prefix qwen35-4b-tower-head
    python job.py evaluate --train-run <chosen> --split val --run-id <id>
    python job.py evaluate --train-run <chosen> --split eval --run-id <id>
    python job.py pack           # local: bundle the code for the cloud

``--device cpu --precision fp32`` runs features and sweep on CPU (smoke).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qv_head.config import JobConfig, load_config, with_device  # noqa: E402
from qv_head.evaluation import evaluate_run  # noqa: E402
from qv_head.features import extract_features  # noqa: E402
from qv_head.packing import git_version, plan_pack, write_tar  # noqa: E402
from qv_head.probe_run import run_probe  # noqa: E402
from qv_head.schema import EVA_DIMENSIONS  # noqa: E402
from qv_head.selection import SelectionError, select_sweep  # noqa: E402
from qv_head.training import run_sweep  # noqa: E402
from qv_head.weights import MODEL_ID, REVISION, Fetch, extract_tower  # noqa: E402

DEFAULT_CONFIG = HERE / "configs" / "qwen35-4b-tower-head.toml"


def _now() -> datetime:
    return datetime.now().astimezone()


def _config(args: argparse.Namespace) -> JobConfig:
    config = load_config(args.config)
    if args.device is None and args.precision is None:
        return config
    return with_device(
        config,
        args.device or config.runtime.device,
        args.precision or config.tower.precision,
    )


def _hf_fetch(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MODEL_ID, name, revision=REVISION))


def _snapshot_fetch(snapshot: Path) -> Fetch:
    def fetch(name: str) -> Path:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"快照里没有 {path}")
        return path

    return fetch


def _weights(args: argparse.Namespace) -> None:
    fetch = _hf_fetch if args.snapshot is None else _snapshot_fetch(args.snapshot)
    weights = extract_tower(fetch, args.models_dir)
    print(
        f"视觉塔 {weights.tensor_count} 个张量，"
        f"{weights.parameter_count / 1e6:.1f}M 参数，已写入 {args.models_dir}",
        flush=True,
    )


def _features(args: argparse.Namespace) -> None:
    record = extract_features(
        _config(args),
        data_root=args.data_root,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        limit=args.limit,
        now=_now(),
    )
    print(
        f"特征 {record.feature_dim} 维；每张 p50 {record.latency_p50_ms:.0f} ms，"
        f"p95 {record.latency_p95_ms:.0f} ms",
        flush=True,
    )


def _probe(args: argparse.Namespace) -> None:
    run_probe(
        data_root=args.data_root,
        runs_dir=args.runs_dir,
        features_run_id=args.features,
        run_id=args.run_id,
        folds=args.folds,
        bootstrap=args.bootstrap,
        seed=args.seed,
        now=_now(),
    )


def _sweep(args: argparse.Namespace) -> None:
    summaries = run_sweep(
        _config(args),
        data_root=args.data_root,
        runs_dir=args.runs_dir,
        features_run_id=args.features,
        prefix=args.prefix,
        lrs=args.lrs,
        seeds=args.seeds,
        clock=_now,
    )
    print(f"sweep 完成：共 {len(summaries)} 组", flush=True)


def _select(args: argparse.Namespace) -> None:
    out = args.runs_dir / f"{args.prefix}-selection.json"
    if out.exists():
        raise SelectionError(f"{out} 已存在，不覆盖")
    selection = select_sweep(args.runs_dir, args.prefix, now=_now())
    out.write_text(selection.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for group in selection.groups:
        values = " ".join(f"{v:.3f}" for v in group.best_val_srcc_mean)
        print(f"lr {group.lr:g}: 均值 {group.mean:.3f}（{values}）", flush=True)
    print(f"选中 {selection.chosen_run_id}", flush=True)


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_run(
        data_root=args.data_root,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
        train_run_id=args.train_run,
        split=args.split,
        run_id=args.run_id,
        now=_now(),
    )
    for dim in EVA_DIMENSIONS:
        m = report.metrics[dim]
        print(
            f"{dim}: SRCC {m.srcc:.3f}  PLCC {m.plcc:.3f}  MAE {m.mae_1_5:.3f}",
            flush=True,
        )
    print(
        f"每张 p50 {report.latency_p50_ms:.0f} ms，p95 {report.latency_p95_ms:.0f} ms",
        flush=True,
    )


def _pack(args: argparse.Namespace) -> None:
    files = plan_pack(HERE)
    commit, dirty = git_version(HERE)
    write_tar(args.output, HERE, files, commit=commit, dirty=dirty, now=_now())
    print(
        json.dumps(
            {"output": str(args.output), "files": len(files), "git_dirty": dirty},
            ensure_ascii=False,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """All sub-commands with their options."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--models-dir", type=Path, default=HERE / "models")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    steps = parser.add_subparsers(dest="step", required=True)

    weights = steps.add_parser("weights", help="从基础模型切出视觉塔")
    weights.add_argument("--snapshot", type=Path, default=None)
    weights.set_defaults(handler=_weights)

    for name, handler in (("features", _features), ("sweep", _sweep)):
        step = steps.add_parser(name)
        step.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        step.add_argument("--device", default=None)
        step.add_argument("--precision", choices=("bf16", "fp32"), default=None)
        step.set_defaults(handler=handler)
        if name == "features":
            step.add_argument("--run-id", required=True)
            step.add_argument("--limit", type=int, default=None)
        else:
            step.add_argument("--features", required=True)
            step.add_argument("--prefix", required=True)
            step.add_argument("--lrs", nargs="+", required=True)
            step.add_argument("--seeds", nargs="+", type=int, required=True)

    probe = steps.add_parser("probe")
    probe.add_argument("--features", required=True)
    probe.add_argument("--run-id", required=True)
    probe.add_argument("--folds", type=int, default=5)
    probe.add_argument("--bootstrap", type=int, default=1000)
    probe.add_argument("--seed", type=int, default=0)
    probe.set_defaults(handler=_probe)

    select = steps.add_parser("select")
    select.add_argument("--prefix", required=True)
    select.set_defaults(handler=_select)

    evaluate = steps.add_parser("evaluate")
    evaluate.add_argument("--train-run", required=True)
    evaluate.add_argument("--split", choices=("val", "eval"), required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.set_defaults(handler=_evaluate)

    pack = steps.add_parser("pack")
    pack.add_argument(
        "--output", type=Path, default=HERE / "dist" / "qwen-vision-head.tar"
    )
    pack.set_defaults(handler=_pack)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one step."""

    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
