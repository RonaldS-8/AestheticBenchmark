"""Bundle the job for the cloud: code, the dataset and one recipe's weights.

Every image and checkpoint is hash-checked while packing. Tests, docs, runs,
the probe and the packer itself stay behind. ``BUILD_INFO.json`` records the
git commit the code came from and whether the tree had local changes.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aes_head.config import load_config
from aes_head.data import DATASET_FILE, load_dataset, sha256_file
from aes_head.schema import WeightsIndex
from aes_head.weights import INDEX_FILE, load_index

ARCHIVE_ROOT = "aesthetic-head"
CODE_PATTERNS = (
    "aes_head/*.py",
    "configs/*.toml",
    "train.py",
    "evaluate.py",
    "select_run.py",
    "setup.sh",
    "run.sh",
    "sweep.py",
    "task.sh",
    "requirements.txt",
    "README.md",
)


class PackError(ValueError):
    """Raised for missing or changed files, or an existing output."""


@dataclass(frozen=True)
class Entry:
    """One archive member, from a file on disk or from bytes."""

    name: str
    source: Path | None = None
    data: bytes | None = None

    def size(self) -> int:
        """Size in bytes."""

        return self.source.stat().st_size if self.source else len(self.data or b"")


def plan_pack(root: Path, config_path: Path) -> list[Entry]:
    """List and verify everything the cloud job needs."""

    config = load_config(config_path)
    entries: list[Entry] = []
    for pattern in CODE_PATTERNS:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise PackError(f"缺少 {pattern}")
        entries += [Entry(p.relative_to(root).as_posix(), p) for p in matches]
    data_root = root / "data"
    dataset = load_dataset(data_root)
    entries.append(Entry(f"data/{DATASET_FILE}", data_root / DATASET_FILE))
    entries.append(Entry(f"data/{dataset.info.items_path}", data_root / "items.jsonl"))
    for item in dataset.items:
        path = data_root / item.image_path
        if not path.is_file() or sha256_file(path) != item.image_sha256:
            raise PackError(f"图片缺失或哈希不符: {path}")
        entries.append(Entry(f"data/{item.image_path}", path))
    weights = load_index(root / "models").find(config.backbone)
    path = root / "models" / weights.path
    if not path.is_file() or sha256_file(path) != weights.sha256:
        raise PackError(f"权重缺失或哈希不符: {path}")
    entries.append(Entry(f"models/{weights.path}", path))
    index = WeightsIndex(schema_version="1.0", files=(weights,))
    entries.append(
        Entry(
            f"models/{INDEX_FILE}",
            data=(index.model_dump_json(indent=2) + "\n").encode(),
        )
    )
    return entries


def git_version(root: Path) -> tuple[str, bool]:
    """Commit hash of ``root``'s checkout and whether this folder has changes."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise PackError(f"git {' '.join(args)} 失败: {result.stderr.strip()}")
        return result.stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", "."))


def write_tar(
    output: Path, entries: list[Entry], *, commit: str, dirty: bool, now: datetime
) -> None:
    """Write the archive under ``aesthetic-head/``; never overwrites."""

    if output.exists():
        raise PackError(f"{output} 已存在，不覆盖")
    info = {
        "created_at": now.isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "files": len(entries),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    members = [*entries, Entry("BUILD_INFO.json", data=json.dumps(info).encode())]
    with tarfile.open(output, "w") as archive:
        for entry in members:
            name = f"{ARCHIVE_ROOT}/{entry.name}"
            if entry.source is not None:
                archive.add(entry.source, arcname=name, recursive=False)
                continue
            member = tarfile.TarInfo(name)
            member.size = entry.size()
            member.mtime = int(now.timestamp())
            archive.addfile(member, io.BytesIO(entry.data or b""))
