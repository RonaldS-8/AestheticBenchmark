from __future__ import annotations

import json
import shutil
import tomllib
import uuid
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
WORKSPACE_TMP = Path(__file__).parents[1] / "data" / "test-tmp"
RECORD_ONLY_KEYS = ("image_id", "image", "downloaded_at")


@pytest.fixture
def tmp_path() -> Path:
    """Keep test files in the writable data tree, not the locked system temp."""

    WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
    path = WORKSPACE_TMP / f"pytest-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def record_data() -> dict[str, Any]:
    return load_fixture("photo_record.json")


@pytest.fixture
def candidate_data() -> dict[str, Any]:
    data = load_fixture("photo_record.json")
    for key in RECORD_ONLY_KEYS:
        del data[key]
    return data


@pytest.fixture
def capture_data() -> dict[str, Any]:
    return load_fixture("captured_response.json")


@pytest.fixture
def annotation_data() -> dict[str, Any]:
    return load_fixture("annotation_record.json")


@pytest.fixture
def sft_data() -> dict[str, Any]:
    return load_fixture("sft_sample.json")


@pytest.fixture
def train_config_data() -> dict[str, Any]:
    with (FIXTURES / "lora_sft.example.toml").open("rb") as f:
        return tomllib.load(f)


@pytest.fixture
def eval_run_data() -> dict[str, Any]:
    return load_fixture("eval_run_config.json")


@pytest.fixture
def eval_report_data() -> dict[str, Any]:
    return load_fixture("eval_report.json")


@pytest.fixture
def gate_data() -> dict[str, Any]:
    return load_fixture("gate_decision.json")


@pytest.fixture
def export_data() -> dict[str, Any]:
    return load_fixture("export_manifest.json")


@pytest.fixture
def head_export_data() -> dict[str, Any]:
    return load_fixture("head_export_manifest.json")
