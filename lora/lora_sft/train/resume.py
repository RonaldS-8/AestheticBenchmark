"""Decide where an interrupted training run resumes, without importing torch.

Trainer writes a checkpoint in place, adapter first and ``trainer_state.json``
last, and its own ``resume_from_checkpoint=True`` simply takes the highest
``checkpoint-N``. A run killed mid-save would then load step-N weights but
restart at step 0, because the state file is missing. Here only complete
checkpoints count, and the step recorded inside must match the folder name.

The code version of a run is read from git when the checkout has ``.git``,
otherwise from the ``BUILD_INFO.json`` that ``scripts/pack_for_cloud.py``
writes into the upload archive.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
TRAINER_STATE = "trainer_state.json"
# Written by Trainer before trainer_state.json; all must exist to resume.
CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    TRAINER_STATE,
)
BUILD_INFO = "BUILD_INFO.json"
# Packages whose version change could alter numerics after a resume.
PINNED_PACKAGES = ("torch", "transformers", "peft", "accelerate")


class ResumeError(RuntimeError):
    """Raised when a run cannot resume exactly where it stopped."""


@dataclass(frozen=True)
class ResumePoint:
    """The checkpoint to resume from and the incomplete ones passed over."""

    path: Path
    step: int
    skipped: tuple[tuple[str, str], ...]


def checkpoint_problem(path: Path, step: int) -> str | None:
    """Why ``path`` cannot be resumed from, or None when it is complete."""

    missing = [name for name in CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        return f"缺少 {', '.join(missing)}"
    try:
        state = json.loads((path / TRAINER_STATE).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"{TRAINER_STATE} 无法解析: {exc}"
    if state.get("global_step") != step:
        return f"{TRAINER_STATE} 记录的步数 {state.get('global_step')} 与目录名不符"
    return None


def find_resume_point(output_dir: Path) -> ResumePoint:
    """Return the newest complete checkpoint; fail if there is none."""

    checkpoints: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            match = CHECKPOINT_PATTERN.match(path.name)
            if match and path.is_dir():
                checkpoints.append((int(match.group(1)), path))
    skipped: list[tuple[str, str]] = []
    for step, path in sorted(checkpoints, reverse=True):
        problem = checkpoint_problem(path, step)
        if problem is None:
            return ResumePoint(path=path, step=step, skipped=tuple(skipped))
        skipped.append((path.name, problem))
    detail = "；".join(f"{name}: {why}" for name, why in skipped) or "没有检查点目录"
    raise ResumeError(
        f"{output_dir} 里没有完整的检查点（{detail}），无法续训。"
        f"第一个检查点之前中断的运行没有可恢复的进度：确认后删除 {output_dir}，"
        "再不带 --resume / RESUME=1 重新开始"
    )


def check_resume_environment(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Refuse to resume when a pinned package version changed."""

    changed = [
        f"{name} {stored.get(name)} → {current.get(name)}"
        for name in PINNED_PACKAGES
        if stored.get(name) != current.get(name)
    ]
    if changed:
        raise ResumeError(
            f"续训环境与首次训练不同：{'，'.join(changed)}；"
            "按 requirements-train.txt 重装环境后再续训"
        )


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout


def source_version(repo_root: Path) -> dict[str, Any]:
    """Commit and uncommitted paths of the code being run.

    Uses git when ``repo_root/.git`` exists, else ``BUILD_INFO.json`` from the
    upload archive. Fails when neither is available.
    """

    if (repo_root / ".git").exists():
        changes = _git(repo_root, "status", "--porcelain").splitlines()
        return {
            "source": "git",
            "commit": _git(repo_root, "rev-parse", "HEAD").strip(),
            "uncommitted": sorted(line[3:] for line in changes if line),
        }
    info_path = repo_root / BUILD_INFO
    if not info_path.is_file():
        raise ResumeError(
            f"{repo_root} 既没有 .git 也没有 {BUILD_INFO}，无法记录代码版本；"
            "请用 scripts/pack_for_cloud.py 打包上传"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return {
        "source": "archive",
        "commit": info["commit"],
        "uncommitted": info["uncommitted"],
    }
