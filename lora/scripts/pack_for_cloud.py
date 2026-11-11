"""Bundle the repo and only the data one training run needs into a tar file.

Code is every file git tracks or would track (``git ls-files -co
--exclude-standard``) outside ``data/``. Data is the sample directory named by
the run config (manifest, split, train/val/eval JSONL) plus exactly the images
those samples reference. ``data/raw`` and other datasets are left out.
The archive also carries ``BUILD_INFO.json`` (commit, uncommitted paths, pack
time), since it has no ``.git``; training records it in ``run_info.json``.

LoRA must train on the prompt chosen by prompt tuning (ADR-0012), so packing
needs that selection file, refuses a config with any other prompt (run
``scripts/apply_prompt_selection.py`` first) and ships the selection along.

Example::

    python scripts/pack_for_cloud.py \\
        --config configs/eva-qwen35-4b-lora-r16.toml \\
        --selection data/manifests/eva/prompt-tuning/v1-selection.json \\
        --output dist/eva-qwen35-4b-lora-r16.tar --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.eval.eva_baseline import PromptSelection  # noqa: E402
from lora_sft.train.config import LoraSftConfig, load_train_config  # noqa: E402
from lora_sft.train.eva_data import load_eva_samples  # noqa: E402
from lora_sft.train.resume import BUILD_INFO, source_version  # noqa: E402

ARCHIVE_ROOT = "Lora-SFT"


def code_files(repo_root: Path) -> list[str]:
    """Tracked and untracked-but-not-ignored files outside ``data/``."""

    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    files = [name for name in listed.split("\0") if name]
    return sorted(
        name
        for name in files
        if PurePosixPath(name).parts[0] != "data" and (repo_root / name).is_file()
    )


def data_files(config: LoraSftConfig, repo_root: Path, data_root: str) -> list[str]:
    """Sample directory files plus every image the samples reference."""

    sample_dir = (repo_root / config.data.train_samples).parent
    files = {
        path.relative_to(repo_root).as_posix()
        for path in sample_dir.iterdir()
        if path.is_file()
    }
    for split in ("train", "val", "eval"):
        path = sample_dir / f"{split}.jsonl"
        for sample in load_eva_samples(path, split):
            image = f"{data_root}/{sample.image_path}"
            if not (repo_root / image).is_file():
                raise FileNotFoundError(f"样本引用的图片不存在: {image}")
            files.add(image)
    return sorted(files)


def selection_file(path: Path, config: LoraSftConfig, repo_root: Path) -> str:
    """Repo-relative path of a selection the config already follows."""

    selection = PromptSelection.model_validate_json(path.read_text(encoding="utf-8"))
    if config.prompt_id != selection.lora_prompt_id:
        raise ValueError(
            f"配置用的是 {config.prompt_id}，提示词调优选定的是 "
            f"{selection.lora_prompt_id}；先运行 scripts/apply_prompt_selection.py"
        )
    if selection.model != config.base_model:
        raise ValueError("提示词选择用的模型与训练配置不同")
    return path.resolve().relative_to(repo_root).as_posix()


def main(argv: list[str] | None = None) -> None:
    """List, size and (unless ``--dry-run``) write the archive."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    if not isinstance(config, LoraSftConfig):
        raise TypeError(f"{args.config} 不是 lora_sft 配置")
    selection = selection_file(args.selection, config, REPOSITORY_ROOT)
    code = code_files(REPOSITORY_ROOT)
    data = data_files(config, REPOSITORY_ROOT, args.data_root)
    names = [*code, *data, selection]
    total = sum((REPOSITORY_ROOT / name).stat().st_size for name in names)
    version = source_version(REPOSITORY_ROOT)
    build_info = {
        "commit": version["commit"],
        "uncommitted": version["uncommitted"],
        "packed_at": datetime.now(SHANGHAI_TIMEZONE).isoformat(),
        "config": args.config.as_posix(),
        "selection": selection,
    }
    print(
        json.dumps(
            {
                "code_files": len(code),
                "data_files": len(data),
                "megabytes": round(total / 1e6, 1),
                "commit": version["commit"],
                "uncommitted_files": len(version["uncommitted"]),
                "output": args.output.as_posix(),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )
    if version["uncommitted"]:
        print(
            f"注意：有 {len(version['uncommitted'])} 个未提交的改动会一起打包，"
            f"清单写进 {BUILD_INFO}；建议先提交再打包",
            file=sys.stderr,
        )
    if args.dry_run:
        return
    if args.output.exists():
        raise FileExistsError(f"压缩包已存在，不覆盖: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x") as archive:
        for name in names:
            archive.add(
                REPOSITORY_ROOT / name,
                arcname=f"{ARCHIVE_ROOT}/{name}",
                recursive=False,
            )
        text = json.dumps(build_info, ensure_ascii=False, indent=2) + "\n"
        payload = text.encode("utf-8")
        member = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{BUILD_INFO}")
        member.size = len(payload)
        member.mtime = int(time.time())
        archive.addfile(member, io.BytesIO(payload))


if __name__ == "__main__":
    main()
