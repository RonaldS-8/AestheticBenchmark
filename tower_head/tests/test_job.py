from __future__ import annotations

import json
import tarfile
from pathlib import Path

import job
import pytest
from qv_head.packing import (
    ARCHIVE_ROOT,
    CODE_PATTERNS,
    PackError,
    git_version,
    plan_pack,
    write_tar,
)
from qv_head.selection import SelectionError
from tests.helpers import NOW, make_rows, write_dataset

from tests.conftest import write_snapshot

TINY_TOML = """
schema_version = "1.0"
recipe = "tiny-head"
[tower]
taps = [1, 3]
min_pixels = 1024
max_pixels = 16384
precision = "bf16"
[head]
hidden_dim = 8
dropout = 0.0
[loss]
mse_weight = 1.0
plcc_weight = 1.0
rank_weight = 0.3
[train]
batch_size = 4
epochs = 2
warmup_ratio = 0.1
weight_decay = 0.0
max_grad_norm = 1.0
[runtime]
device = "cuda"
"""


def test_every_step_runs_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (tmp_path / "data").mkdir()
    write_dataset(tmp_path / "data", make_rows(12, 8, 4))
    config = tmp_path / "tiny.toml"
    config.write_text(TINY_TOML, encoding="utf-8")
    roots = ["--data-root", str(tmp_path / "data")]
    roots += ["--models-dir", str(tmp_path / "models")]
    roots += ["--runs-dir", str(tmp_path / "runs")]
    cpu = ["--config", str(config), "--device", "cpu", "--precision", "fp32"]

    assert job.main([*roots, "weights", "--snapshot", str(snapshot)]) == 0
    assert job.main([*roots, "features", *cpu, "--run-id", "t-features"]) == 0
    probe = ["--features", "t-features", "--run-id", "t-probe", "--folds", "2"]
    assert job.main([*roots, "probe", *probe, "--bootstrap", "100"]) == 0
    sweep = ["--features", "t-features", "--prefix", "t", "--lrs", "1e-2"]
    assert job.main([*roots, "sweep", *cpu, *sweep, "--seeds", "1", "2"]) == 0
    assert job.main([*roots, "select", "--prefix", "t"]) == 0
    chosen = json.loads((tmp_path / "runs" / "t-selection.json").read_text())
    evaluate = ["evaluate", "--train-run", chosen["chosen_run_id"]]
    assert job.main([*roots, *evaluate, "--split", "val", "--run-id", "t-val"]) == 0
    assert job.main([*roots, *evaluate, "--split", "eval", "--run-id", "t-eval"]) == 0

    out = capsys.readouterr().out
    assert "视觉塔" in out and "sweep 完成：共 2 组" in out
    assert "选中 t-lr1e-2-s" in out and "technical: SRCC" in out
    with pytest.raises(SelectionError, match="不覆盖"):
        job.main([*roots, "select", "--prefix", "t"])


def test_snapshot_fetch_requires_the_file(tmp_path: Path) -> None:
    fetch = job._snapshot_fetch(tmp_path)

    with pytest.raises(FileNotFoundError, match="快照"):
        fetch("config.json")


def test_pack_bundles_code_only(tmp_path: Path) -> None:
    files = plan_pack(job.HERE)
    commit, dirty = git_version(job.HERE)
    output = tmp_path / "bundle.tar"

    write_tar(output, job.HERE, files, commit=commit, dirty=dirty, now=NOW)

    with tarfile.open(output) as archive:
        names = archive.getnames()
    assert f"{ARCHIVE_ROOT}/job.py" in names
    assert f"{ARCHIVE_ROOT}/BUILD_INFO.json" in names
    assert not any("/tests/" in n or "/data/" in n for n in names)
    assert len(commit) == 40
    with pytest.raises(PackError, match="不覆盖"):
        write_tar(output, job.HERE, files, commit=commit, dirty=dirty, now=NOW)


def test_pack_fails_on_missing_files_and_git_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(PackError, match="缺少"):
        plan_pack(tmp_path)
    monkeypatch.setattr(
        "qv_head.packing.CODE_PATTERNS", (*CODE_PATTERNS[:1], "missing.txt")
    )
    with pytest.raises(PackError, match="missing.txt"):
        plan_pack(job.HERE)
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    with pytest.raises(PackError, match="git"):
        git_version(outside)


def test_cli_pack_writes_the_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "q.tar"

    assert job.main(["pack", "--output", str(output)]) == 0

    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    assert output.is_file()
