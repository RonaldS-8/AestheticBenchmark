from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from lora_sft.export.schema import (
    EXPORT_MANIFEST_ADAPTER,
    HeadExportManifest,
    LoraExportManifest,
)

WEIGHTS = {"path": "model.gguf", "sha256": "1" * 64, "role": "weights"}
PROJECTOR = {"path": "mmproj.gguf", "sha256": "2" * 64, "role": "vision_projector"}


def test_fixture_is_valid(export_data: dict[str, Any]) -> None:
    manifest = LoraExportManifest.model_validate(export_data)

    assert [f.role for f in manifest.files] == ["weights", "vision_projector"]


@pytest.mark.parametrize(("fmt", "runtime"), [("gguf", "mlx"), ("mlx", "llama_cpp")])
def test_format_must_match_runtime(
    export_data: dict[str, Any], fmt: str, runtime: str
) -> None:
    export_data.update(format=fmt, runtime=runtime)

    with pytest.raises(ValidationError, match="格式不能用"):
        LoraExportManifest.model_validate(export_data)


def test_mlx_export_needs_no_projector(export_data: dict[str, Any]) -> None:
    weights = WEIGHTS | {"path": "model.safetensors"}
    export_data.update(
        format="mlx", runtime="mlx", quantization="4bit", files=[weights]
    )

    assert LoraExportManifest.model_validate(export_data).format == "mlx"


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ([WEIGHTS], "视觉投影文件"),
        ([WEIGHTS, PROJECTOR, PROJECTOR | {"path": "mmproj2.gguf"}], "视觉投影文件"),
        ([PROJECTOR], "缺少 weights"),
        ([WEIGHTS, WEIGHTS, PROJECTOR], "重复路径"),
        ([], "files"),
    ],
)
def test_file_list_rules(
    export_data: dict[str, Any], files: list[dict[str, str]], message: str
) -> None:
    export_data["files"] = files

    with pytest.raises(ValidationError, match=message):
        LoraExportManifest.model_validate(export_data)


@pytest.mark.parametrize("quantization", ["Q4_K_M", "q4-k-m", ""])
def test_quantization_label_format(
    export_data: dict[str, Any], quantization: str
) -> None:
    export_data["quantization"] = quantization

    with pytest.raises(ValidationError, match="quantization"):
        LoraExportManifest.model_validate(export_data)


def test_head_fixture_is_valid(head_export_data: dict[str, Any]) -> None:
    manifest = EXPORT_MANIFEST_ADAPTER.validate_python(head_export_data)

    assert isinstance(manifest, HeadExportManifest)
    assert manifest.views.technical.size == 224


def test_adapter_picks_lora_route(export_data: dict[str, Any]) -> None:
    manifest = EXPORT_MANIFEST_ADAPTER.validate_python(export_data)

    assert isinstance(manifest, LoraExportManifest)


@pytest.mark.parametrize("role", ["vision_projector", "tokenizer"])
def test_head_export_rejects_vlm_files(
    head_export_data: dict[str, Any], role: str
) -> None:
    head_export_data["files"].append(
        {"path": "extra.bin", "sha256": "3" * 64, "role": role}
    )

    with pytest.raises(ValidationError, match="美学头导出"):
        HeadExportManifest.model_validate(head_export_data)


@pytest.mark.parametrize(
    ("key", "value"), [("format", "gguf"), ("runtime", "llama_cpp")]
)
def test_head_export_is_onnx_only(
    head_export_data: dict[str, Any], key: str, value: str
) -> None:
    head_export_data[key] = value

    with pytest.raises(ValidationError, match=key):
        HeadExportManifest.model_validate(head_export_data)


def test_head_export_needs_weights(head_export_data: dict[str, Any]) -> None:
    head_export_data["files"][0]["role"] = "config"

    with pytest.raises(ValidationError, match="缺少 weights"):
        HeadExportManifest.model_validate(head_export_data)
