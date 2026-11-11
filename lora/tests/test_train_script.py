"""Torch-free parts of ``scripts/train_lora_sft.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from lora_sft.eval.eva_eval import INELIGIBLE_SELECTION_SCORE
from lora_sft.train.config import LoraSftConfig, load_train_config
from scripts.train_lora_sft import check_selected

ROOT = Path(__file__).resolve().parents[1]
EVA_CONFIG = ROOT / "configs" / "eva-qwen35-4b-lora-r16.toml"
INELIGIBLE = INELIGIBLE_SELECTION_SCORE


def eva_config(metric: str = "srcc_mean") -> LoraSftConfig:
    config = load_train_config(EVA_CONFIG)
    assert isinstance(config, LoraSftConfig)
    selection = config.selection.model_copy(update={"metric": metric})
    return config.model_copy(update={"selection": selection})


def test_eva_config_selects_by_srcc_and_needs_fla() -> None:
    config = load_train_config(EVA_CONFIG)

    assert isinstance(config, LoraSftConfig)
    assert config.selection.metric == "srcc_mean"
    assert config.kernels.require_fla is True
    assert config.runtime.save_steps == config.runtime.eval_steps


def test_check_selected_accepts_eligible_best() -> None:
    check_selected(eva_config(), 0.71, "outputs/run/checkpoint-306")
    check_selected(eva_config("eval_loss"), 0.42, "outputs/run/checkpoint-102")


@pytest.mark.parametrize(
    ("metric", "best", "checkpoint", "message"),
    [
        ("srcc_mean", INELIGIBLE, "outputs/run/checkpoint-102", "选模条件"),
        ("srcc_mean", None, None, "最佳检查点"),
        ("eval_loss", 0.42, None, "最佳检查点"),
    ],
)
def test_check_selected_refuses_without_a_winner(
    metric: str, best: float | None, checkpoint: str | None, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        check_selected(eva_config(metric), best, checkpoint)
