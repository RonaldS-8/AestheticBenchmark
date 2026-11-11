from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from qv_head.data import sha256_file
from qv_head.weights import (
    INDEX_FILE,
    TOWER_DIR,
    WeightsError,
    extract_tower,
    load_index,
    verify_tower,
)
from safetensors import safe_open

from tests.conftest import TINY_VISION, snapshot_fetch, write_snapshot


def test_extract_cuts_out_only_the_tower(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")

    weights = extract_tower(
        snapshot_fetch(snapshot), tmp_path / "models", revision="1" * 40
    )

    assert weights.revision == "1" * 40
    assert {f.path.split("/")[-1] for f in weights.files} == {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    }
    assert "model.safetensors-00002-of-00002.safetensors" in weights.source_files
    assert "model.safetensors-00001-of-00002.safetensors" not in weights.source_files
    tower_file = tmp_path / "models" / TOWER_DIR / "model.safetensors"
    with safe_open(str(tower_file), framework="pt") as handle:
        keys = list(handle.keys())
    assert keys and not any(k.startswith("model.") for k in keys)
    assert weights.tensor_count == len(keys)
    saved = json.loads((tmp_path / "models" / TOWER_DIR / "config.json").read_text())
    assert saved["depth"] == TINY_VISION["depth"]
    assert load_index(tmp_path / "models") == weights
    assert verify_tower(tmp_path / "models") == weights


def test_extract_refuses_existing_output(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    extract_tower(snapshot_fetch(snapshot), tmp_path / "models", revision="1" * 40)

    with pytest.raises(WeightsError, match="不覆盖"):
        extract_tower(snapshot_fetch(snapshot), tmp_path / "models", revision="1" * 40)


def test_extract_rejects_missing_prefix_and_mixed_dtypes(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    with pytest.raises(WeightsError, match="没有以"):
        extract_tower(
            snapshot_fetch(snapshot),
            tmp_path / "a",
            revision="1" * 40,
            prefix="model.audio.",
        )

    mixed = write_snapshot(tmp_path / "mixed", extra_dtype=torch.bfloat16)
    with pytest.raises(WeightsError, match="精度不统一"):
        extract_tower(snapshot_fetch(mixed), tmp_path / "b", revision="1" * 40)


def test_verify_tower_catches_changed_and_missing_files(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    models = tmp_path / "models"
    extract_tower(snapshot_fetch(snapshot), models, revision="1" * 40)
    config = models / TOWER_DIR / "config.json"
    config.write_text(config.read_text() + " ", encoding="utf-8")

    with pytest.raises(WeightsError, match="哈希"):
        verify_tower(models)
    config.unlink()
    with pytest.raises(WeightsError, match="不存在"):
        verify_tower(models)
    (models / INDEX_FILE).unlink()
    with pytest.raises(WeightsError, match="job.py weights"):
        load_index(models)


def test_index_pins_source_hashes(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")

    weights = extract_tower(snapshot_fetch(snapshot), tmp_path / "m", revision="1" * 40)

    assert weights.source_files["config.json"] == sha256_file(snapshot / "config.json")
