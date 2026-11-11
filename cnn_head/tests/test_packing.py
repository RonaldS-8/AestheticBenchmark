from __future__ import annotations

import json
import re
import shutil
import tarfile
from pathlib import Path

import pytest
from aes_head.packing import (
    ARCHIVE_ROOT,
    CODE_PATTERNS,
    PackError,
    git_version,
    plan_pack,
    write_tar,
)
from tests.helpers import HERE, NOW

from tests.conftest import TinyJob, write_tiny_weights


@pytest.fixture
def job_root(tiny_job: TinyJob, tmp_path: Path) -> Path:
    """A job root with the real code files and the tiny dataset and weights."""

    root = tmp_path / "job"
    for pattern in CODE_PATTERNS:
        for path in HERE.glob(pattern):
            target = root / path.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copytree(tiny_job.data_root, root / "data")
    write_tiny_weights(root / "models")
    config = root / "configs" / "convnext-t-ft.toml"
    text = config.read_text(encoding="utf-8")
    config.write_text(text.replace('"convnext_tiny"', '"tiny_cnn"'), encoding="utf-8")
    return root


def test_pack_contains_code_data_and_one_checkpoint(job_root: Path) -> None:
    entries = plan_pack(job_root, job_root / "configs" / "convnext-t-ft.toml")
    output = job_root.parent / "dist" / "job.tar"

    write_tar(output, entries, commit="d" * 40, dirty=True, now=NOW)

    with tarfile.open(output) as archive:
        names = archive.getnames()
        info = json.load(archive.extractfile(f"{ARCHIVE_ROOT}/BUILD_INFO.json"))  # type: ignore[arg-type]
        index = json.load(archive.extractfile(f"{ARCHIVE_ROOT}/models/WEIGHTS.json"))  # type: ignore[arg-type]
    assert f"{ARCHIVE_ROOT}/train.py" in names
    assert f"{ARCHIVE_ROOT}/models/tiny.pth" in names
    assert sum("/data/images/" in n for n in names) == 22
    assert not any("/tests/" in n or n.endswith("pack.py") for n in names)
    assert info["git_dirty"] is True and info["files"] == len(entries)
    assert [f["backbone"] for f in index["files"]] == ["tiny_cnn"]
    with pytest.raises(PackError, match="不覆盖"):
        write_tar(output, entries, commit="d" * 40, dirty=False, now=NOW)


def test_pack_rejects_changed_image_or_weights(job_root: Path) -> None:
    config = job_root / "configs" / "convnext-t-ft.toml"
    image = sorted((job_root / "data" / "images" / "eva").iterdir())[0]
    original = image.read_bytes()
    image.write_bytes(b"changed")
    with pytest.raises(PackError, match="图片"):
        plan_pack(job_root, config)
    image.write_bytes(original)
    (job_root / "models" / "tiny.pth").write_bytes(b"changed")
    with pytest.raises(PackError, match="权重"):
        plan_pack(job_root, config)


def test_pack_requires_every_code_file(job_root: Path) -> None:
    (job_root / "run.sh").unlink()

    with pytest.raises(PackError, match="run.sh"):
        plan_pack(job_root, job_root / "configs" / "convnext-t-ft.toml")


def test_git_version_reads_this_checkout(tmp_path: Path) -> None:
    commit, dirty = git_version(HERE)

    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert isinstance(dirty, bool)
    with pytest.raises(PackError, match="git"):
        git_version(tmp_path)
