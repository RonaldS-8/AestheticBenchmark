from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lora_sft.data.head_samples import LabelSource
from lora_sft.train.config import (
    TRAIN_CONFIG_ADAPTER,
    HeadConfig,
    LoraSftConfig,
    load_train_config,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "lora_sft.example.toml"
HEAD_FIXTURE = FIXTURES / "head.example.toml"


def test_load_example_config() -> None:
    config = load_train_config(FIXTURE)

    assert isinstance(config, LoraSftConfig)
    assert config.base_model.model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert config.lora.freeze_vision is True
    assert config.global_batch_size == 16


def test_load_rejects_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('schema_version = "1.0"\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="route"):
        load_train_config(path)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "seed"),
        ("optim", "max_grad_norm"),
        ("lora", "freeze_vision"),
        ("runtime", "precision"),
        ("kernels", "require_fla"),
        ("selection", "metric"),
    ],
)
def test_no_field_has_a_default(
    train_config_data: dict[str, Any], section: str | None, key: str
) -> None:
    target = train_config_data if section is None else train_config_data[section]
    del target[key]

    with pytest.raises(ValidationError, match=key):
        LoraSftConfig.model_validate(train_config_data)


def test_unknown_key_rejected(train_config_data: dict[str, Any]) -> None:
    train_config_data["lora"]["use_dora"] = True

    with pytest.raises(ValidationError, match="use_dora"):
        LoraSftConfig.model_validate(train_config_data)


@pytest.mark.parametrize(
    ("key", "value"),
    [("revision", "main"), ("revision", "abc123"), ("model_id", "qwen-4b")],
)
def test_base_model_must_be_pinned(
    train_config_data: dict[str, Any], key: str, value: str
) -> None:
    train_config_data["base_model"][key] = value

    with pytest.raises(ValidationError, match=key):
        LoraSftConfig.model_validate(train_config_data)


@pytest.mark.parametrize(
    ("min_pixels", "max_pixels", "valid"),
    [(1000, 1000, True), (1001, 1000, False), (0, 1000, False)],
)
def test_pixel_budget(
    train_config_data: dict[str, Any], min_pixels: int, max_pixels: int, valid: bool
) -> None:
    train_config_data["preprocess"] = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

    if valid:
        assert LoraSftConfig.model_validate(train_config_data).preprocess.max_pixels
    else:
        with pytest.raises(ValidationError, match="min_pixels"):
            LoraSftConfig.model_validate(train_config_data)


@pytest.mark.parametrize(
    ("key", "value", "valid"),
    [
        ("r", 1, True),
        ("r", 256, True),
        ("r", 0, False),
        ("r", 257, False),
        ("dropout", 0.0, True),
        ("dropout", 1.0, False),
    ],
)
def test_lora_boundaries(
    train_config_data: dict[str, Any], key: str, value: float, valid: bool
) -> None:
    train_config_data["lora"][key] = value

    if valid:
        assert (
            getattr(LoraSftConfig.model_validate(train_config_data).lora, key) == value
        )
    else:
        with pytest.raises(ValidationError, match=key):
            LoraSftConfig.model_validate(train_config_data)


@pytest.mark.parametrize(
    ("modules", "message"),
    [([], "target_modules"), (["q_proj", "q_proj"], "target_modules 有重复")],
)
def test_target_modules(
    train_config_data: dict[str, Any], modules: list[str], message: str
) -> None:
    train_config_data["lora"]["target_modules"] = modules

    with pytest.raises(ValidationError, match=message):
        LoraSftConfig.model_validate(train_config_data)


@pytest.mark.parametrize(
    ("strategy", "gpus", "valid"),
    [
        ("single", 1, True),
        ("single", 2, False),
        ("ddp", 1, False),
        ("ddp", 8, True),
        ("fsdp", 8, True),
        ("deepspeed", 8, False),
    ],
)
def test_strategy_matches_gpus(
    train_config_data: dict[str, Any], strategy: str, gpus: int, valid: bool
) -> None:
    train_config_data["runtime"].update(strategy=strategy, num_gpus=gpus)

    if valid:
        config = LoraSftConfig.model_validate(train_config_data)
        assert config.global_batch_size == 16 * gpus
    else:
        with pytest.raises(ValidationError, match="strategy"):
            LoraSftConfig.model_validate(train_config_data)


@pytest.mark.parametrize(
    ("key", "value"),
    [("learning_rate", 0.0), ("epochs", 0), ("warmup_ratio", 1.0)],
)
def test_optim_boundaries(
    train_config_data: dict[str, Any], key: str, value: float
) -> None:
    train_config_data["optim"][key] = value

    with pytest.raises(ValidationError, match=key):
        LoraSftConfig.model_validate(train_config_data)


def test_output_dir_must_be_relative(train_config_data: dict[str, Any]) -> None:
    train_config_data["runtime"]["output_dir"] = "D:/outputs/run"

    with pytest.raises(ValidationError, match="相对 POSIX 路径"):
        LoraSftConfig.model_validate(train_config_data)


def head_data() -> dict[str, Any]:
    with HEAD_FIXTURE.open("rb") as f:
        return tomllib.load(f)


def test_load_head_example_config() -> None:
    config = load_train_config(HEAD_FIXTURE)

    assert isinstance(config, HeadConfig)
    assert config.views.technical.size == 224
    assert config.label_sources == (LabelSource.HUMAN_MEAN,)
    assert config.global_batch_size == 32


def test_unknown_route_rejected(train_config_data: dict[str, Any]) -> None:
    train_config_data["route"] = "dpo"

    with pytest.raises(ValidationError, match="route"):
        TRAIN_CONFIG_ADAPTER.validate_python(train_config_data)


def test_head_route_rejects_lora_fields(train_config_data: dict[str, Any]) -> None:
    train_config_data["route"] = "head"

    with pytest.raises(ValidationError, match="views"):
        TRAIN_CONFIG_ADAPTER.validate_python(train_config_data)


@pytest.mark.parametrize(
    ("kind", "weight", "valid"),
    [
        ("plcc_rank", 0.3, True),
        ("plcc_rank", None, False),
        ("mse", None, True),
        ("mse", 0.3, False),
        ("plcc_rank", 0.0, False),
    ],
)
def test_head_loss_rank_weight(kind: str, weight: float | None, valid: bool) -> None:
    data = head_data()
    data["loss"] = {"kind": kind, "rank_weight": weight}

    if valid:
        assert HeadConfig.model_validate(data).loss.kind == kind
    else:
        with pytest.raises(ValidationError, match="rank_weight"):
            HeadConfig.model_validate(data)


@pytest.mark.parametrize(
    ("view", "key", "value"),
    [
        ("aesthetic", "size", 31),
        ("aesthetic", "size", 1025),
        ("technical", "grid", 0),
        ("technical", "grid", 17),
        ("technical", "patch", 7),
    ],
)
def test_head_view_bounds(view: str, key: str, value: int) -> None:
    data = head_data()
    data["views"][view][key] = value

    with pytest.raises(ValidationError, match=key):
        HeadConfig.model_validate(data)


def test_head_backbone_must_be_pinned() -> None:
    data = head_data()
    data["views"]["technical"]["backbone"]["revision"] = "main"

    with pytest.raises(ValidationError, match="revision"):
        HeadConfig.model_validate(data)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"label_sources": []}, "label_sources"),
        ({"label_sources": ["human_mean", "human_mean"]}, "label_sources 有重复"),
        ({"label_sources": ["user_votes"]}, "label_sources"),
        ({"init_from_run": "head-convnext-swin-human"}, "不能是本次运行"),
    ],
)
def test_head_sources_and_init(updates: dict[str, Any], message: str) -> None:
    data = head_data() | updates

    with pytest.raises(ValidationError, match=message):
        HeadConfig.model_validate(data)


def test_head_stage_one_needs_no_init() -> None:
    data = head_data() | {"label_sources": ["site_rating"], "init_from_run": None}

    assert HeadConfig.model_validate(data).init_from_run is None


def test_save_steps_must_equal_eval_steps(
    train_config_data: dict[str, Any],
) -> None:
    train_config_data["runtime"].update(save_steps=400, eval_steps=200)

    with pytest.raises(ValidationError, match="save_steps"):
        LoraSftConfig.model_validate(train_config_data)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("metric", "val_mae"),
        ("min_parse_rate", 0.0),
        ("min_parse_rate", 1.01),
        ("max_new_tokens", 4),
    ],
)
def test_selection_boundaries(
    train_config_data: dict[str, Any], key: str, value: object
) -> None:
    train_config_data["selection"][key] = value

    with pytest.raises(ValidationError, match=key):
        LoraSftConfig.model_validate(train_config_data)


def test_selection_and_kernels_load(train_config_data: dict[str, Any]) -> None:
    train_config_data["selection"].update(metric="srcc_mean", min_parse_rate=1.0)
    train_config_data["kernels"]["require_fla"] = True

    config = LoraSftConfig.model_validate(train_config_data)

    assert config.selection.metric == "srcc_mean"
    assert config.kernels.require_fla is True
