"""Manifests of on-device exports, one per training route.

``LoraExportManifest``: LoRA merged into the base, converted to GGUF or MLX
and quantized. ``HeadExportManifest``: backbones plus head exported to ONNX.
File paths are relative to the export bundle root, where the manifest sits.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from lora_sft.common import Record, RelativePath, RunName, SchemaVersion, Sha256
from lora_sft.train.config import BaseModelRef, HeadViews, ImagePreprocess

FORMAT_RUNTIME = {"gguf": "llama_cpp", "mlx": "mlx"}


class ExportedFile(Record):
    """One file shipped to the NAS."""

    path: RelativePath
    sha256: Sha256
    role: Literal["weights", "vision_projector", "tokenizer", "config"]


class _ExportBase(Record):
    """Fields shared by every export."""

    schema_version: SchemaVersion
    export_id: RunName
    train_run_name: RunName
    checkpoint_sha256: Sha256
    quantization: str = Field(pattern=r"^[a-z0-9_]+$")
    runtime_version: str = Field(min_length=1)
    files: tuple[ExportedFile, ...] = Field(min_length=1)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check_files(self) -> _ExportBase:
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("files 中有重复路径")
        roles = [f.role for f in self.files]
        if "weights" not in roles:
            raise ValueError("缺少 weights 文件")
        return self


class LoraExportManifest(_ExportBase):
    """Merged LoRA model; ``checkpoint_sha256`` is the adapter file's hash."""

    route: Literal["lora_sft"]
    base_model: BaseModelRef
    format: Literal["gguf", "mlx"]
    runtime: Literal["llama_cpp", "mlx"]
    chat_template_sha256: Sha256
    preprocess: ImagePreprocess

    @model_validator(mode="after")
    def _check_format(self) -> LoraExportManifest:
        if FORMAT_RUNTIME[self.format] != self.runtime:
            raise ValueError(f"{self.format} 格式不能用 {self.runtime} 运行")
        projectors = [f for f in self.files if f.role == "vision_projector"]
        if self.format == "gguf" and len(projectors) != 1:
            raise ValueError("GGUF 导出必须恰好包含一个视觉投影文件 (mmproj)")
        return self


class HeadExportManifest(_ExportBase):
    """Aesthetic head in ONNX; ``checkpoint_sha256`` is the head checkpoint."""

    route: Literal["head"]
    format: Literal["onnx"]
    runtime: Literal["onnxruntime"]
    views: HeadViews

    @model_validator(mode="after")
    def _check_roles(self) -> HeadExportManifest:
        if any(f.role in ("vision_projector", "tokenizer") for f in self.files):
            raise ValueError("美学头导出不应包含视觉投影或分词器文件")
        return self


ExportManifest = Annotated[
    LoraExportManifest | HeadExportManifest, Field(discriminator="route")
]
EXPORT_MANIFEST_ADAPTER: TypeAdapter[LoraExportManifest | HeadExportManifest] = (
    TypeAdapter(ExportManifest)
)
