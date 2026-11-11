from __future__ import annotations

import json
from pathlib import Path

import prepare_data
import pytest
from aes_head.data import load_dataset
from aes_head.prepare import PrepareError, plan_copy, write_copy
from tests.helpers import NOW, make_rows, write_source


def _edit_json(path: Path, **changes: object) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "main-data"
    return root, write_source(root, make_rows(6, 2, 2))


def test_plan_copy_reads_and_checks_source(source: tuple[Path, str]) -> None:
    root, manifest = source

    plan = plan_copy(root, manifest, now=NOW)

    assert plan.info.counts == {"train": 6, "val": 2, "eval": 2}
    assert plan.info.source_manifest.path == manifest
    assert plan.image_bytes > 0
    assert [i.split for i in plan.items] == ["train"] * 6 + ["val"] * 2 + ["eval"] * 2


def test_write_copy_produces_a_loadable_dataset(
    source: tuple[Path, str], tmp_path: Path
) -> None:
    root, manifest = source
    plan = plan_copy(root, manifest, now=NOW)
    out = tmp_path / "copy"

    write_copy(plan, root, out)
    dataset = load_dataset(out)

    assert dataset.items == plan.items
    assert dataset.info == plan.info
    with pytest.raises(PrepareError, match="不覆盖"):
        write_copy(plan, root, out)


def test_plan_copy_rejects_wrong_split_label(source: tuple[Path, str]) -> None:
    root, manifest = source
    path = root / "manifests/eva/sft/v1/val.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace('"split": "val"', '"split": "eval"'),
        encoding="utf-8",
    )

    with pytest.raises(PrepareError, match="划分的样本"):
        plan_copy(root, manifest, now=NOW)


def test_plan_copy_rejects_count_mismatch(source: tuple[Path, str]) -> None:
    root, manifest = source
    _edit_json(root / manifest, train_count=5)

    with pytest.raises(PrepareError, match="数量"):
        plan_copy(root, manifest, now=NOW)


def test_plan_copy_rejects_duplicate_ids(source: tuple[Path, str]) -> None:
    root, manifest = source
    folder = root / "manifests/eva/sft/v1"
    first = (folder / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with (folder / "val.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(first.replace('"split": "train"', '"split": "val"') + "\n")
    split = json.loads((folder / "split-v1.json").read_text(encoding="utf-8"))
    _edit_json(
        folder / "split-v1.json", val=[*split["val"], json.loads(first)["image_id"]]
    )
    _edit_json(root / manifest, val_count=3)

    with pytest.raises(PrepareError, match="重复"):
        plan_copy(root, manifest, now=NOW)


def test_plan_copy_rejects_wrong_eval_fingerprint(source: tuple[Path, str]) -> None:
    root, manifest = source
    _edit_json(root / manifest, eval_fingerprint="0" * 64)

    with pytest.raises(PrepareError, match="指纹"):
        plan_copy(root, manifest, now=NOW)


def test_plan_copy_rejects_other_datasets(source: tuple[Path, str]) -> None:
    root, manifest = source
    _edit_json(root / manifest, dataset="ava")

    with pytest.raises(PrepareError, match="只支持 EVA"):
        plan_copy(root, manifest, now=NOW)


def test_plan_copy_rejects_changed_or_missing_image(source: tuple[Path, str]) -> None:
    root, manifest = source
    images = sorted((root / "images/eva").iterdir())
    images[0].write_bytes(b"changed")

    with pytest.raises(PrepareError, match="哈希"):
        plan_copy(root, manifest, now=NOW)
    images[0].unlink()
    with pytest.raises(PrepareError, match="不存在"):
        plan_copy(root, manifest, now=NOW)


def test_cli_dry_run_writes_nothing(
    source: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, manifest = source
    out = tmp_path / "copy"
    common = ["--source-root", str(root), "--manifest", manifest, "--output", str(out)]

    assert prepare_data.main([*common, "--dry-run"]) == 0
    assert not out.exists()
    assert '"dry_run": true' in capsys.readouterr().out
    assert prepare_data.main(common) == 0
    assert (out / "dataset.json").is_file()
