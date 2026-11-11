"""Cut the vision tower out of the pinned Qwen3.5-4B checkpoint into ``models/``.

Only the tower's tensors (``model.visual.*``) are read, from the shards that
hold them, and loaded strictly into ``Qwen3_5VisionModel``: a missing or
unexpected tensor is an error. The tower is saved in the checkpoint's own
dtype with ``save_pretrained`` next to the base model's image-processor
config, and ``models/WEIGHTS.json`` pins every file by SHA-256. Nothing is
overwritten.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from qv_head.data import sha256_file
from qv_head.schema import TowerFile, TowerWeights

MODEL_ID = "Qwen/Qwen3.5-4B"
# Same revision as configs/eva-qwen35-4b-lora-r16.toml in the main repository.
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
WEIGHT_PREFIX = "model.visual."
TOWER_DIR = "qwen35-4b-visual"
INDEX_FILE = "WEIGHTS.json"
CONFIG_FILE = "config.json"
PREPROCESSOR_FILE = "preprocessor_config.json"
WEIGHT_INDEX_FILE = "model.safetensors.index.json"
TOWER_WEIGHTS_FILE = "model.safetensors"

# A file name in the base-model repository at the pinned revision -> local path.
Fetch = Callable[[str], Path]


class WeightsError(ValueError):
    """Raised for a bad checkpoint, a hash mismatch or an existing output."""


def _tower_state(
    fetch: Fetch, prefix: str
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The tower's tensors (prefix removed) and hashes of the files read."""

    index_path = fetch(WEIGHT_INDEX_FILE)
    weight_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))[
        "weight_map"
    ]
    keys = sorted(k for k in weight_map if k.startswith(prefix))
    if not keys:
        raise WeightsError(f"权重索引里没有以 {prefix} 开头的张量")
    sources = {WEIGHT_INDEX_FILE: sha256_file(index_path)}
    state: dict[str, torch.Tensor] = {}
    for shard in sorted({weight_map[k] for k in keys}):
        path = fetch(shard)
        sources[shard] = sha256_file(path)
        with safe_open(str(path), framework="pt") as handle:
            for key in keys:
                if weight_map[key] == shard:
                    state[key[len(prefix) :]] = handle.get_tensor(key)
    return state, sources


def extract_tower(
    fetch: Fetch,
    models_dir: Path,
    *,
    model_id: str = MODEL_ID,
    revision: str = REVISION,
    prefix: str = WEIGHT_PREFIX,
) -> TowerWeights:
    """Write ``models/<TOWER_DIR>/`` and ``models/WEIGHTS.json``."""

    out_dir = models_dir / TOWER_DIR
    if (models_dir / INDEX_FILE).exists() or out_dir.exists():
        raise WeightsError(f"{models_dir} 里已有视觉塔，不覆盖；确认后删除再重跑")
    config_path = fetch(CONFIG_FILE)
    preprocessor_path = fetch(PREPROCESSOR_FILE)
    full_config = Qwen3_5Config.from_json_file(str(config_path))
    state, sources = _tower_state(fetch, prefix)
    dtypes = {tensor.dtype for tensor in state.values()}
    if len(dtypes) != 1:
        raise WeightsError(f"视觉塔张量的精度不统一: {sorted(map(str, dtypes))}")
    model = Qwen3_5VisionModel(full_config.vision_config).to(dtypes.pop())
    # strict: every tower parameter must come from the checkpoint, nothing else.
    model.load_state_dict(state, strict=True)
    out_dir.mkdir(parents=True)
    model.save_pretrained(out_dir)
    shutil.copyfile(preprocessor_path, out_dir / PREPROCESSOR_FILE)
    sources[CONFIG_FILE] = sha256_file(config_path)
    sources[PREPROCESSOR_FILE] = sha256_file(preprocessor_path)
    files = tuple(
        TowerFile(
            path=f"{TOWER_DIR}/{path.name}",
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(out_dir.iterdir())
    )
    weights = TowerWeights(
        schema_version="1.0",
        model_id=model_id,
        revision=revision,
        weight_prefix=prefix,
        tensor_count=len(state),
        parameter_count=sum(p.numel() for p in model.parameters()),
        source_files=dict(sorted(sources.items())),
        files=files,
    )
    (models_dir / INDEX_FILE).write_text(
        weights.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return weights


def load_index(models_dir: Path) -> TowerWeights:
    """Read ``models/WEIGHTS.json``."""

    path = models_dir / INDEX_FILE
    if not path.is_file():
        raise WeightsError(f"找不到 {path}，先运行 job.py weights")
    return TowerWeights.model_validate_json(path.read_text(encoding="utf-8"))


def verify_tower(models_dir: Path) -> TowerWeights:
    """Check every tower file against ``WEIGHTS.json``."""

    weights = load_index(models_dir)
    for entry in weights.files:
        path = models_dir / entry.path
        if not path.is_file():
            raise WeightsError(f"视觉塔文件不存在: {path}")
        if sha256_file(path) != entry.sha256:
            raise WeightsError(f"视觉塔文件哈希与 WEIGHTS.json 不符: {path}")
    return weights
