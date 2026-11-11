"""Bundle the job's code for the cloud.

Data and weights are not packed: the cloud machine already holds the main
repository's EVA data and the Qwen3.5-4B checkpoint, and ``prepare_data.py``
and ``job.py weights`` copy them in there with every hash checked.
``BUILD_INFO.json`` records the git commit and whether this folder had local
changes.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

ARCHIVE_ROOT = "qwen-vision-head"
CODE_PATTERNS = (
    "qv_head/*.py",
    "configs/*.toml",
    "job.py",
    "prepare_data.py",
    "setup.sh",
    "run.sh",
    "task.sh",
    "requirements.txt",
    "README.md",
)


class PackError(ValueError):
    """Raised for missing files or an existing output."""


def plan_pack(root: Path) -> list[Path]:
    """Every code file the cloud job needs."""

    files: list[Path] = []
    for pattern in CODE_PATTERNS:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise PackError(f"缺少 {pattern}")
        files += matches
    return files


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
    output: Path,
    root: Path,
    files: list[Path],
    *,
    commit: str,
    dirty: bool,
    now: datetime,
) -> None:
    """Write the archive under ``qwen-vision-head/``; never overwrites."""

    if output.exists():
        raise PackError(f"{output} 已存在，不覆盖")
    info = json.dumps(
        {
            "created_at": now.isoformat(),
            "git_commit": commit,
            "git_dirty": dirty,
            "files": len(files),
        }
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as archive:
        for path in files:
            name = f"{ARCHIVE_ROOT}/{path.relative_to(root).as_posix()}"
            archive.add(path, arcname=name, recursive=False)
        member = tarfile.TarInfo(f"{ARCHIVE_ROOT}/BUILD_INFO.json")
        member.size = len(info)
        member.mtime = int(now.timestamp())
        archive.addfile(member, io.BytesIO(info))
