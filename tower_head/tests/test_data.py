from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from qv_head.data import (
    DATASET_FILE,
    DatasetError,
    items_jsonl_bytes,
    load_dataset,
    load_rgb,
    sha256_bytes,
)
from qv_head.schema import (
    DatasetInfo,
    HeadItem,
    ScoreSummary,
    TowerFile,
    TowerWeights,
)
from tests.helpers import make_rows, summary, write_dataset


def _rewrite_info(root: Path, **changes: object) -> None:
    path = root / DATASET_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_dataset_keeps_split_order(tmp_path: Path) -> None:
    rows = make_rows(5, 2, 2)
    write_dataset(tmp_path, rows)

    dataset = load_dataset(tmp_path)

    assert [i.image_id for i in dataset.split("train")] == list(rows)[:5]
    assert len(dataset.split("val")) == 2
    assert len(dataset.split("eval")) == 2


def test_load_dataset_requires_dataset_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="prepare_data"):
        load_dataset(tmp_path)


def test_load_dataset_rejects_changed_items(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(3, 1, 1))
    path = tmp_path / "items.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"count":30', b'"count":31', 1))

    with pytest.raises(DatasetError, match="哈希"):
        load_dataset(tmp_path)


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(3, 1, 1))
    lines = (tmp_path / "items.jsonl").read_bytes().splitlines(keepends=True)
    raw = b"".join([*lines, lines[0]])
    (tmp_path / "items.jsonl").write_bytes(raw)
    _rewrite_info(tmp_path, items_sha256=sha256_bytes(raw))

    with pytest.raises(DatasetError, match="重复"):
        load_dataset(tmp_path)


def test_load_dataset_rejects_wrong_counts(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(3, 1, 1))
    _rewrite_info(tmp_path, counts={"train": 2, "val": 2, "eval": 1})

    with pytest.raises(DatasetError, match="数量"):
        load_dataset(tmp_path)


def test_load_dataset_rejects_wrong_eval_fingerprint(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(3, 1, 1))
    _rewrite_info(tmp_path, eval_fingerprint="0" * 64)

    with pytest.raises(DatasetError, match="指纹"):
        load_dataset(tmp_path)


def test_load_rgb_decodes_at_native_size(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(2, 1, 1))
    item = load_dataset(tmp_path).items[1]

    image = load_rgb(tmp_path, item)

    assert image.mode == "RGB"
    assert image.size == (41, 32)


def test_load_rgb_rejects_changed_or_missing_image(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(2, 1, 1))
    first, second = load_dataset(tmp_path).items[:2]
    (tmp_path / first.image_path).write_bytes(b"not a jpeg")
    (tmp_path / second.image_path).unlink()

    with pytest.raises(DatasetError, match="哈希"):
        load_rgb(tmp_path, first)
    with pytest.raises(DatasetError, match="不存在"):
        load_rgb(tmp_path, second)


def test_score_summary_maps_to_project_scale() -> None:
    aesthetic = summary(4.6875, 0, 10)
    technical = summary(2.75, 1, 4)

    assert aesthetic.to_project_scale() == pytest.approx(2.875)
    assert aesthetic.to_reported_target() == 2.9
    assert technical.to_reported_target() == 3.3


@pytest.mark.parametrize(
    ("mean", "low", "high", "message"),
    [(5.0, 4.0, 4.0, "scale_min"), (11.0, 0.0, 10.0, "超出")],
)
def test_score_summary_rejects_bad_scale(
    mean: float, low: float, high: float, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        summary(mean, low, high)


@pytest.mark.parametrize("path", ["../x.jpg", "C:/x.jpg", "a\\b.jpg", "/x.jpg", "."])
def test_item_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="相对 POSIX"):
        HeadItem(
            image_id="photo_000001",
            split="train",
            image_path=path,
            image_sha256="a" * 64,
            image_decode_warnings=(),
            aesthetic=summary(5, 0, 10),
            technical=summary(2, 1, 4),
        )


def test_item_summary_picks_dimension(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(2, 1, 1))
    item = load_dataset(tmp_path).items[0]

    assert item.summary("aesthetic") is item.aesthetic
    assert item.summary("technical") is item.technical


def test_dataset_info_needs_all_splits(tmp_path: Path) -> None:
    info = write_dataset(tmp_path, make_rows(2, 1, 1))
    data = info.model_dump()
    data["counts"] = {"train": 2, "val": 1}

    with pytest.raises(ValidationError, match="三个划分"):
        DatasetInfo.model_validate(data)


def test_items_jsonl_is_one_line_per_item(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_rows(2, 1, 1))
    items = load_dataset(tmp_path).items

    assert items_jsonl_bytes(items).count(b"\n") == len(items)


def test_tower_weights_find() -> None:
    entry = TowerFile(
        path="qwen35-4b-visual/model.safetensors", sha256="b" * 64, size_bytes=10
    )
    index = TowerWeights(
        schema_version="1.0",
        model_id="Qwen/Qwen3.5-4B",
        revision="0" * 40,
        weight_prefix="model.visual.",
        tensor_count=1,
        parameter_count=1,
        source_files={"config.json": "a" * 64},
        files=(entry,),
    )

    assert index.find("model.safetensors") == entry
    with pytest.raises(KeyError, match="job.py weights"):
        index.find("config.json")


def test_records_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ScoreSummary.model_validate(
            {
                "mean": 1.0,
                "standard_deviation": 0.0,
                "count": 1,
                "scale_min": 0.0,
                "scale_max": 2.0,
                "extra": 1,
            }
        )
