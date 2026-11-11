from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

from lora_sft.eval.eva_baseline import PromptSelection
from lora_sft.train.config import LoraSftConfig, load_train_config
from lora_sft.train.prompt_switch import (
    PromptSwitchError,
    check_only_switched,
    check_selection,
    current_manifest,
    prompt_file,
    switched_config_text,
)
from scripts.eval_eva_sft import main as eval_main
from scripts.pack_for_cloud import selection_file
from tests.test_train_eva_data import SAMPLES, repo  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
EVA_CONFIG = ROOT / "configs" / "eva-qwen35-4b-lora-r16.toml"
V2 = "data/manifests/eva/sft/v2"


def eva_config() -> LoraSftConfig:
    config = load_train_config(EVA_CONFIG)
    assert isinstance(config, LoraSftConfig)
    return config


def selection_data(lora: str = "rubric-score@v2", **overrides: object) -> dict:
    def candidate(prompt_id: str, method: str, score: float) -> dict:
        return {
            "run_id": prompt_id.partition("@")[0],
            "prompt_id": prompt_id,
            "method": method,
            "parse_success_rate": 1.0,
            "srcc": {"aesthetic": score, "technical": score},
            "ineligible": {},
            "score": score,
        }

    config = eva_config()
    return {
        "schema_version": "1.0",
        "created_at": "2026-11-06T10:00:00+08:00",
        "tuning_set": {"path": "data/tuning-set.json", "sha256": "1" * 64},
        "ids_fingerprint": "2" * 64,
        "model": config.base_model.model_dump(),
        "runtime": "transformers",
        "min_parse_rate": 0.95,
        "candidates": [
            candidate("zero-shot-score@v2", "b0_zero_shot", 0.5),
            candidate("rubric-score@v2", "b1_rubric", 0.6),
            candidate("few-shot-score@v2", "b2_few_shot", 0.7),
        ],
        "best_prompt_id": "few-shot-score@v2",
        "lora_prompt_id": lora,
        **overrides,
    }


def selection(lora: str = "rubric-score@v2", **overrides: object) -> PromptSelection:
    return PromptSelection.model_validate(selection_data(lora, **overrides))


def switched_text(prompt_id: str = "rubric-score@v2") -> str:
    return switched_config_text(
        EVA_CONFIG.read_text(encoding="utf-8"),
        prompt_id=prompt_id,
        sample_dir=V2,
        split_version="split-v2",
    )


def test_selection_must_match_model_runtime_and_differ(tmp_path: Path) -> None:
    config = eva_config()
    check_selection(selection(), config)

    other_model = {"model_id": "Qwen/Qwen3.5-9B", "revision": "a" * 40}
    with pytest.raises(PromptSwitchError, match="模型"):
        check_selection(selection(model=other_model), config)
    with pytest.raises(PromptSwitchError, match="transformers"):
        check_selection(selection(runtime="llama_cpp"), config)
    same = config.model_copy(update={"prompt_id": "rubric-score@v2"})
    with pytest.raises(PromptSwitchError, match="已经在用"):
        check_selection(selection(), same)


def test_prompt_file_finds_exactly_one(tmp_path: Path) -> None:
    path = prompt_file("rubric-score@v2", ROOT / "prompts")

    assert path.relative_to(ROOT).as_posix() == (
        "prompts/baseline-v2/rubric-score.prompt.md"
    )
    with pytest.raises(PromptSwitchError, match="有 0 份"):
        prompt_file("rubric-score@v9", ROOT / "prompts")


def test_switch_rewrites_only_prompt_and_data_paths() -> None:
    old = eva_config()

    new = check_only_switched(old, switched_text())

    assert new.prompt_id == "rubric-score@v2"
    assert new.data.split_manifest == f"{V2}/split-v2.json"
    assert new.data.split_version == "split-v2"
    assert (new.data.train_samples, new.data.val_samples) == (
        f"{V2}/train.jsonl",
        f"{V2}/val.jsonl",
    )
    assert new.data.eval_fingerprint == old.data.eval_fingerprint
    assert "# SHA-256 of chat_template.jinja" in switched_text()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda t: t.replace("r = 16", "r = 8"), "其他部分不能变"),
        (lambda t: t.replace("max_new_tokens = 64", "max_new_tokens = 128"), "其他"),
        (
            lambda t: t.replace(tomllib.loads(t)["data"]["eval_fingerprint"], "0" * 64),
            "eval_fingerprint",
        ),
    ],
)
def test_switch_rejects_any_other_change(
    edit: Callable[[str], str], message: str
) -> None:
    with pytest.raises(PromptSwitchError, match=message):
        check_only_switched(eva_config(), edit(switched_text()))


def test_switch_needs_each_key_exactly_once() -> None:
    text = EVA_CONFIG.read_text(encoding="utf-8")
    kwargs = {"prompt_id": "rubric-score@v2", "sample_dir": V2}
    current = f'split_version = "{eva_config().data.split_version}"\n'

    with pytest.raises(PromptSwitchError, match="split_version 应恰好出现一次"):
        switched_config_text(
            text.replace(current, ""),
            split_version="split-v2",
            **kwargs,
        )
    with pytest.raises(PromptSwitchError, match="prompt_id 应恰好出现一次，实际 2"):
        switched_config_text(
            text + 'prompt_id = "x@v1"\n', split_version="split-v2", **kwargs
        )


def config_for_repo(repo: Path, prompt_id: str) -> Path:  # noqa: F811
    data = tomllib.loads(EVA_CONFIG.read_text(encoding="utf-8"))
    split = json.loads((repo / SAMPLES / "split-v1.json").read_text(encoding="utf-8"))
    # Point the config at the v1 fixture, whatever the real config uses.
    text = switched_config_text(
        EVA_CONFIG.read_text(encoding="utf-8"),
        prompt_id=prompt_id,
        sample_dir=SAMPLES,
        split_version="split-v1",
    )
    text = text.replace(data["data"]["eval_fingerprint"], split["eval_fingerprint"])
    path = repo / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_current_manifest_reads_sample_manifest(repo: Path) -> None:  # noqa: F811
    config = load_train_config(config_for_repo(repo, "eva-dual-score@v1"))
    assert isinstance(config, LoraSftConfig)

    assert current_manifest(config, repo).seed == 42
    with pytest.raises(PromptSwitchError, match="找不到"):
        current_manifest(config, repo / "elsewhere")


def test_pack_requires_config_on_selected_prompt(tmp_path: Path) -> None:
    path = ROOT / "data" / "test-tmp" / f"selection-{tmp_path.name}.json"
    path.write_text(json.dumps(selection_data()), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="apply_prompt_selection"):
            selection_file(path, eva_config(), ROOT)
        switched = check_only_switched(eva_config(), switched_text())
        assert selection_file(path, switched, ROOT) == (
            f"data/test-tmp/selection-{tmp_path.name}.json"
        )
        other = switched.model_copy(
            update={
                "base_model": selection(
                    model={"model_id": "a/b", "revision": "c" * 40}
                ).model
            }
        )
        with pytest.raises(ValueError, match="模型"):
            selection_file(path, other, ROOT)
    finally:
        path.unlink()


def test_eval_rejects_samples_on_another_prompt(repo: Path) -> None:  # noqa: F811
    config = config_for_repo(repo, "rubric-score@v2")

    with pytest.raises(ValueError, match="与配置 rubric-score@v2 不符"):
        eval_main(
            [
                "--config",
                str(config),
                "--samples",
                str(repo / SAMPLES / "val.jsonl"),
                "--split",
                "val",
                "--run-id",
                "eva-check",
                "--base",
                "--runs-root",
                str(repo / "runs"),
            ]
        )
    assert not (repo / "runs").exists()
