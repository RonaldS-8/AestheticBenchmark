from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from lora_sft.data.eva_sft import (
    EVA_PROMPT_ID,
    QWEN35_CHAT_TEMPLATE_SHA256,
    QWEN35_MODEL_ID,
    QWEN35_REVISION,
    SplitRatios,
    prepare_eva_sft,
)
from lora_sft.train.config import LoraSftConfig, load_train_config
from lora_sft.train.eva_data import (
    IGNORE_INDEX,
    EvaTrainData,
    TrainDataError,
    chat_messages,
    check_no_eval_leak,
    completion_text,
    load_eva_samples,
    load_train_data,
    mask_labels,
    verify_images,
)
from lora_sft.train.prompt_switch import prompt_file
from tests.test_data_eva_sft import NOW, PROMPT, write_sources

ROOT = Path(__file__).resolve().parents[1]
EVA_CONFIG = ROOT / "configs" / "eva-qwen35-4b-lora-r16.toml"
SAMPLES = "data/manifests/eva/sft/v1"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake repo root holding a prepared 10-image EVA dataset."""

    data_root = tmp_path / "data"
    write_sources(data_root, tuple(str(i) for i in range(1, 11)))
    prepare_eva_sft(
        data_root,
        PROMPT,
        version="v1",
        seed=42,
        ratios=SplitRatios(train=0.6, val=0.2, eval=0.2),
        created_at=NOW,
        dry_run=False,
    )
    return tmp_path


def config_for(repo: Path, **overrides: object) -> LoraSftConfig:
    data = tomllib.loads(EVA_CONFIG.read_text(encoding="utf-8"))
    split = json.loads((repo / SAMPLES / "split-v1.json").read_text(encoding="utf-8"))
    # The fixture is prepared on the v1 prompt, whatever the real config uses.
    data["prompt_id"] = EVA_PROMPT_ID
    data["data"].update(
        split_manifest=f"{SAMPLES}/split-v1.json",
        split_version="split-v1",
        eval_fingerprint=split["eval_fingerprint"],
        train_samples=f"{SAMPLES}/train.jsonl",
        val_samples=f"{SAMPLES}/val.jsonl",
    )
    data.update(overrides)
    return LoraSftConfig.model_validate(data)


def test_eva_config_matches_prepared_dataset_contract() -> None:
    config = load_train_config(EVA_CONFIG)

    assert isinstance(config, LoraSftConfig)
    assert config.base_model.model_id == QWEN35_MODEL_ID
    assert config.base_model.revision == QWEN35_REVISION
    assert config.chat_template_sha256 == QWEN35_CHAT_TEMPLATE_SHA256
    assert prompt_file(config.prompt_id, ROOT / "prompts").is_file()
    assert config.data.split_manifest.endswith(f"/{config.data.split_version}.json")
    assert {"in_proj_qkv", "in_proj_z", "out_proj"} <= set(config.lora.target_modules)
    assert config.runtime.save_steps % config.runtime.eval_steps == 0


def test_load_train_data_happy_path(repo: Path) -> None:
    config = config_for(repo)

    data = load_train_data(config, repo)
    eval_samples = load_eva_samples(repo / SAMPLES / "eval.jsonl", "eval")
    check_no_eval_leak(data, eval_samples)

    assert (len(data.train), len(data.val), len(eval_samples)) == (6, 2, 2)
    assert verify_images((*data.train, *data.val), repo / "data") > 0


def test_split_version_mismatch_rejected(repo: Path) -> None:
    config = config_for(repo)
    data = config.data.model_copy(update={"split_version": "split-v2"})

    with pytest.raises(TrainDataError, match="划分版本"):
        load_train_data(config.model_copy(update={"data": data}), repo)


def test_eval_fingerprint_mismatch_rejected(repo: Path) -> None:
    config = config_for(repo)
    data = config.data.model_copy(update={"eval_fingerprint": "0" * 64})

    with pytest.raises(TrainDataError, match="eval_fingerprint"):
        load_train_data(config.model_copy(update={"data": data}), repo)


def test_missing_split_file_rejected(repo: Path) -> None:
    config = config_for(repo)
    (repo / SAMPLES / "split-v1.json").unlink()

    with pytest.raises(TrainDataError, match="划分文件不存在"):
        load_train_data(config, repo)


def test_prompt_mismatch_rejected(repo: Path) -> None:
    config = config_for(repo, prompt_id="rubric-score@v1")

    with pytest.raises(TrainDataError, match="提示词"):
        load_train_data(config, repo)


def test_samples_differing_from_split_rejected(repo: Path) -> None:
    path = repo / SAMPLES / "train.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[1:]), encoding="utf-8")

    with pytest.raises(TrainDataError, match="train 样本与划分文件不一致"):
        load_train_data(config_for(repo), repo)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "没有样本"),
        ("\n", "第 1 行为空"),
        ("{}\n", "不符合 schema"),
    ],
)
def test_load_samples_rejects_bad_files(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(TrainDataError, match=message):
        load_eva_samples(path, "train")


def test_load_samples_rejects_missing_wrong_split_and_duplicates(repo: Path) -> None:
    path = repo / SAMPLES / "train.jsonl"

    with pytest.raises(TrainDataError, match="不存在"):
        load_eva_samples(repo / "nope.jsonl", "train")
    with pytest.raises(TrainDataError, match="应为 val"):
        load_eva_samples(path, "val")
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(f"{first}\n{first}\n", encoding="utf-8")
    with pytest.raises(TrainDataError, match="重复 image_id"):
        load_eva_samples(path, "train")


def test_eval_leak_by_id_and_by_hash_rejected(repo: Path) -> None:
    data = load_train_data(config_for(repo), repo)
    eval_samples = load_eva_samples(repo / SAMPLES / "eval.jsonl", "eval")

    by_id = EvaTrainData(data.split, (*data.train, eval_samples[0]), data.val)
    with pytest.raises(TrainDataError, match="出现在评测集中"):
        check_no_eval_leak(by_id, eval_samples)

    twin = data.train[0].model_copy(
        update={"image_sha256": eval_samples[0].image_sha256}
    )
    by_hash = EvaTrainData(data.split, (twin, *data.train[1:]), data.val)
    with pytest.raises(TrainDataError, match="是同一文件"):
        check_no_eval_leak(by_hash, eval_samples)


def test_verify_images_rejects_missing_and_changed_files(repo: Path) -> None:
    sample = load_eva_samples(repo / SAMPLES / "train.jsonl", "train")[0]
    image = repo / "data" / sample.image_path

    image.write_bytes(image.read_bytes() + b"x")
    with pytest.raises(TrainDataError, match="哈希"):
        verify_images([sample], repo / "data")
    image.unlink()
    with pytest.raises(TrainDataError, match="图片不存在"):
        verify_images([sample], repo / "data")


def test_chat_messages_keep_image_placeholder_only(repo: Path) -> None:
    sample = load_eva_samples(repo / SAMPLES / "train.jsonl", "train")[0]

    prompt = chat_messages(sample, include_reply=False)
    full = chat_messages(sample, include_reply=True)

    assert [m["role"] for m in prompt] == ["system", "user"]
    assert prompt[1]["content"][0] == {"type": "image"}
    assert full[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": sample.messages[-1].text}],
    }


REPLY = '{"aesthetic":3.1,"technical":2.9}'


def test_completion_text_returns_reply_and_end_marker() -> None:
    prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    tail = completion_text(prompt, prompt + REPLY + "<|im_end|>\n", REPLY)

    assert tail == REPLY + "<|im_end|>\n"


@pytest.mark.parametrize(
    ("full", "message"),
    [
        ("other" + REPLY + "<|im_end|>", "前缀不一致"),
        ("P" + "<think></think>" + REPLY + "<|im_end|>", "不以 assistant 回复开头"),
        ("P" + REPLY, "没有结束标记"),
    ],
)
def test_completion_text_rejects_template_drift(full: str, message: str) -> None:
    with pytest.raises(TrainDataError, match=message):
        completion_text("P", full, REPLY)


def test_mask_labels_supervises_only_the_tail() -> None:
    labels = mask_labels([1, 2, 3, 4, 5], [4, 5])

    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]


@pytest.mark.parametrize(
    ("input_ids", "completion_ids", "message"),
    [
        ([1, 2], [], "为空"),
        ([1, 2], [1, 2], "提示部分丢失"),
        ([1, 2, 3], [9], "不一致"),
    ],
)
def test_mask_labels_rejects_bad_boundaries(
    input_ids: list[int], completion_ids: list[int], message: str
) -> None:
    with pytest.raises(TrainDataError, match=message):
        mask_labels(input_ids, completion_ids)
