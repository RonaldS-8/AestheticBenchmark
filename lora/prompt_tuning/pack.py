"""Pack the prompt-tuning job for the cloud (run from the main repository).

Reads the LoRA run config, the prompt-tuning set and the EVA SFT samples with
the main repository's own code, then writes one tar holding:

- this folder's code (``ptune/``, prompts, scripts, requirements), without
  ``pack.py`` and ``tests/``;
- ``data/job.json``: model, revision, chat template, pixel budget, precision,
  reply length and parse-rate threshold, all copied from the LoRA config, plus
  the SHA-256 of every model file, read from ``--weights-from`` (a local copy
  already checked against the pinned revision; the weights themselves are not
  packed, the cloud machine downloads them and checks these hashes);
- ``data/tuning-set.json`` (byte copy) and ``data/items.jsonl`` (one row per
  scored image and per few-shot example, with vote means and SFT reply);
- every image those rows reference, and ``BUILD_INFO.json``.

The prompt files here must be byte-identical to ``prompts/baseline-v2/`` in
the main repository, or packing stops. The main repository is not modified.

Example::

    python prompt_tuning/pack.py \\
        --config configs/eva-qwen35-4b-lora-r16.toml \\
        --tuning-set data/manifests/eva/prompt-tuning/v1.json \\
        --weights-from ../OCR_test/models/Qwen3.5-4B \\
        --output dist/prompt-tuning-v1.tar --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
for path in (REPOSITORY_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ptune.schema import JobConfig, TuningItem, WeightFile  # noqa: E402

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.data.eva_sft import EvaSftSample  # noqa: E402
from lora_sft.eval.eva_baseline import PromptTuningSet, tuning_samples  # noqa: E402
from lora_sft.train.config import LoraSftConfig, load_train_config  # noqa: E402
from lora_sft.train.eva_data import load_eva_samples  # noqa: E402
from lora_sft.train.resume import source_version  # noqa: E402

ARCHIVE_ROOT = "prompt-tuning"
FOLDER = HERE.name
LEFT_OUT = ("pack.py", "tests")
PROMPT_DIR = "prompts/baseline-v2"
# Shipped with a model but not read when loading it; configuration.json is
# added by ModelScope only.
NOT_WEIGHTS = frozenset({"LICENSE", "README.md", "configuration.json"})


class PackError(ValueError):
    """Raised when the job cannot be packed consistently."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_row(sample: EvaSftSample, role: str) -> TuningItem:
    """One job row from an SFT sample: vote means and the verbatim reply."""

    return TuningItem(
        image_id=sample.image_id,
        split=sample.split,
        role=role,  # type: ignore[arg-type]
        image_path=sample.image_path,
        image_sha256=sample.image_sha256,
        aesthetic_mean=sample.source_labels.aesthetic.mean,
        technical_mean=sample.source_labels.technical.mean,
        reply_text=sample.messages[-1].text,
    )


def weight_files(folder: Path) -> tuple[WeightFile, ...]:
    """Hash every file a model load reads from a local copy of the weights."""

    weights = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if (
            not path.is_file()
            or path.name in NOT_WEIGHTS
            or any(part.startswith(".") for part in relative.parts)
            or path.suffix == ".part"
            or path.name == "DOWNLOAD.json"
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        weights.append(
            WeightFile(
                path=relative.as_posix(),
                sha256=digest.hexdigest(),
                size_bytes=path.stat().st_size,
            )
        )
    if not weights:
        raise PackError(f"{folder} 里没有模型文件")
    return tuple(weights)


def job_files(
    config: LoraSftConfig,
    config_path: str,
    config_bytes: bytes,
    tuning_set_bytes: bytes,
    data_root: Path,
    weights: tuple[WeightFile, ...],
) -> tuple[dict[str, bytes], list[TuningItem]]:
    """``data/`` files of the job (name -> bytes) and the rows they list."""

    tuning_set = PromptTuningSet.model_validate_json(tuning_set_bytes)
    manifest = data_root / tuning_set.sft_manifest
    if _sha(manifest.read_bytes()) != tuning_set.sft_manifest_sha256:
        raise PackError(f"{manifest} 与提示词调优集记录的哈希不符")
    splits = {
        name: load_eva_samples(manifest.parent / f"{name}.jsonl", name)
        for name in ("train", "val", "eval")
    }
    items, examples = tuning_samples(
        tuning_set, splits["train"], splits["val"], splits["eval"]
    )
    rows = [to_row(s, "item") for s in items] + [to_row(s, "example") for s in examples]
    items_bytes = "".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")
    job = JobConfig(
        schema_version="1.0",
        train_config=config_path,
        train_config_sha256=_sha(config_bytes),
        model=config.base_model.model_dump(),
        chat_template_sha256=config.chat_template_sha256,
        preprocess=config.preprocess.model_dump(),
        precision=config.runtime.precision,
        max_new_tokens=config.selection.max_new_tokens,
        min_parse_rate=config.selection.min_parse_rate,
        tuning_set={"path": "data/tuning-set.json", "sha256": _sha(tuning_set_bytes)},
        items={"path": "data/items.jsonl", "sha256": _sha(items_bytes)},
        weights=weights,
    )
    files = {
        "data/job.json": (job.model_dump_json(indent=2) + "\n").encode("utf-8"),
        "data/tuning-set.json": tuning_set_bytes,
        "data/items.jsonl": items_bytes,
    }
    return files, rows


def check_prompts(repo_root: Path, folder: Path) -> None:
    """The job's prompt files must equal the main repository's, byte for byte."""

    source = sorted((repo_root / PROMPT_DIR).glob("*.prompt.md"))
    copies = sorted((folder / PROMPT_DIR).glob("*.prompt.md"))
    if [p.name for p in source] != [p.name for p in copies]:
        raise PackError(f"{FOLDER}/{PROMPT_DIR} 与 {PROMPT_DIR} 的提示词文件不同")
    for original, copy in zip(source, copies, strict=True):
        if original.read_bytes() != copy.read_bytes():
            raise PackError(f"{copy.name} 与主仓库的副本内容不同，请先同步")


def code_files(repo_root: Path) -> list[str]:
    """Files of this folder git tracks or would track, minus the left-out ones."""

    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z", FOLDER],
        cwd=repo_root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    names = []
    for name in listed.split("\0"):
        if not name or not (repo_root / name).is_file():
            continue
        relative = PurePosixPath(name).relative_to(FOLDER)
        if relative.parts[0] in LEFT_OUT or "__pycache__" in relative.parts:
            continue
        names.append(relative.as_posix())
    return sorted(names)


def main(argv: list[str] | None = None) -> None:
    """List, size and (unless ``--dry-run``) write the archive."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tuning-set", type=Path, required=True)
    parser.add_argument(
        "--weights-from",
        type=Path,
        required=True,
        help="已核对过的本地权重副本，只用来算每个文件的 sha256，不打包",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    if not isinstance(config, LoraSftConfig):
        raise TypeError(f"{args.config} 不是 lora_sft 配置")
    check_prompts(REPOSITORY_ROOT, HERE)
    print(f"计算 {args.weights_from} 里每个文件的 sha256……", file=sys.stderr)
    weights = weight_files(args.weights_from)
    files, rows = job_files(
        config,
        args.config.as_posix(),
        args.config.read_bytes(),
        args.tuning_set.read_bytes(),
        args.data_root,
        weights,
    )
    images = sorted({row.image_path for row in rows})
    for image in images:
        if not (args.data_root / image).is_file():
            raise FileNotFoundError(f"图片不存在: {args.data_root / image}")
    code = code_files(REPOSITORY_ROOT)
    version = source_version(REPOSITORY_ROOT)
    build_info = {
        "commit": version["commit"],
        "uncommitted": version["uncommitted"],
        "packed_at": datetime.now(SHANGHAI_TIMEZONE).isoformat(),
        "config": args.config.as_posix(),
        "tuning_set": args.tuning_set.as_posix(),
    }
    total = sum((HERE / name).stat().st_size for name in code)
    total += sum(len(data) for data in files.values())
    total += sum((args.data_root / image).stat().st_size for image in images)
    print(
        json.dumps(
            {
                "code_files": len(code),
                "scored_images": sum(row.role == "item" for row in rows),
                "example_images": sum(row.role == "example" for row in rows),
                "weight_files": len(weights),
                "weight_gigabytes": round(sum(w.size_bytes for w in weights) / 1e9, 2),
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
            f"注意：有 {len(version['uncommitted'])} 个未提交的改动，"
            "清单写进 BUILD_INFO.json；建议先提交再打包",
            file=sys.stderr,
        )
    if args.dry_run:
        return
    if args.output.exists():
        raise FileExistsError(f"压缩包已存在，不覆盖: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
        member = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{name}")
        member.size = len(payload)
        member.mtime = int(time.time())
        archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(args.output, "x") as archive:
        for name in code:
            archive.add(HERE / name, arcname=f"{ARCHIVE_ROOT}/{name}", recursive=False)
        for name, payload in files.items():
            add_bytes(archive, name, payload)
        for image in images:
            archive.add(
                args.data_root / image,
                arcname=f"{ARCHIVE_ROOT}/data/{image}",
                recursive=False,
            )
        info = json.dumps(build_info, ensure_ascii=False, indent=2) + "\n"
        add_bytes(archive, "BUILD_INFO.json", info.encode("utf-8"))
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
