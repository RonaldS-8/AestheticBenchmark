from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from lora_sft.crawl.dataset_schema import (
    DatasetImageRecord,
    DatasetScoreRecord,
    ScoreSummary,
)
from lora_sft.crawl.http import sha256_file
from lora_sft.data.eva_sft import (
    QWEN35_CHAT_TEMPLATE_SHA256,
    QWEN35_MODEL_ID,
    QWEN35_REVISION,
    EvaPreparationError,
    EvaSftPlan,
    EvaSftSample,
    SplitRatios,
    eva_image_id,
    prepare_eva_sft,
    score_to_project_scale,
)

NOW = datetime.fromisoformat("2026-11-03T10:00:00+08:00")
PROMPT = Path("prompts/eva-sft-v1/eva-dual-score.prompt.md")


def summary(mean: float, scale_min: float, scale_max: float) -> ScoreSummary:
    return ScoreSummary(
        mean=mean,
        standard_deviation=0.5,
        count=20,
        scale_min=scale_min,
        scale_max=scale_max,
    )


def write_sources(
    data_root: Path, ids: tuple[str, ...] = ("1", "2", "3", "4")
) -> tuple[list[DatasetScoreRecord], list[DatasetImageRecord]]:
    image_root = data_root / "images" / "eva"
    manifest_root = data_root / "manifests" / "eva"
    image_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)
    scores: list[DatasetScoreRecord] = []
    images: list[DatasetImageRecord] = []
    for offset, source_id in enumerate(ids):
        path = image_root / f"{source_id}.jpg"
        Image.new("RGB", (8 + offset, 8), (offset, 0, 0)).save(path, "JPEG")
        scores.append(
            DatasetScoreRecord(
                schema_version="1.0",
                capture_id="eva-test-ratings",
                dataset="eva",
                source_image_id=source_id,
                image_member=f"{source_id}.jpg",
                aesthetic=summary(5 + (offset % 5), 0, 10),
                technical=summary(2.5, 1, 4),
                overall=None,
            )
        )
        images.append(
            DatasetImageRecord(
                schema_version="1.0",
                dataset="eva",
                source_image_id=source_id,
                image_path=f"images/eva/{source_id}.jpg",
                sha256=sha256_file(path),
                format="jpeg",
                size_bytes=path.stat().st_size,
                has_eoi_marker=True,
            )
        )
    (manifest_root / "scores.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in scores),
        encoding="utf-8",
    )
    (manifest_root / "images.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in images),
        encoding="utf-8",
    )
    return scores, images


def make_plan(data_root: Path, *, dry_run: bool = True) -> EvaSftPlan:
    return prepare_eva_sft(
        data_root,
        PROMPT,
        version="v1",
        seed=42,
        ratios=SplitRatios(train=0.5, val=0.25, eval=0.25),
        created_at=NOW,
        dry_run=dry_run,
    )


@pytest.mark.parametrize(
    ("mean", "scale_min", "scale_max", "expected"),
    [(0, 0, 10, 1.0), (5, 0, 10, 3.0), (10, 0, 10, 5.0), (2.5, 1, 4, 3.0)],
)
def test_score_to_project_scale(
    mean: float, scale_min: float, scale_max: float, expected: float
) -> None:
    assert score_to_project_scale(summary(mean, scale_min, scale_max)) == expected


def test_eva_image_id_is_stable() -> None:
    assert eva_image_id("71") == "photo_000071"
    assert eva_image_id("1000000") == "photo_1000000"
    with pytest.raises(EvaPreparationError, match="不是数字"):
        eva_image_id("x71")


def test_prepare_eva_sft_dry_run_builds_qwen_messages(tmp_path: Path) -> None:
    write_sources(tmp_path)

    plan = make_plan(tmp_path)

    assert not plan.output_dir.exists()
    assert len(plan.samples) == 4
    split_counts = (len(plan.split.train), len(plan.split.val), len(plan.split.eval))
    assert split_counts == (2, 1, 1)
    sample = next(item for item in plan.samples if item.source_image_id == "1")
    assert tuple(message.role for message in sample.messages) == (
        "system",
        "user",
        "assistant",
    )
    assert sample.messages[1].content[0].model_dump() == {
        "type": "image",
        "image": "images/eva/1.jpg",
    }
    assert sample.target.model_dump() == {"aesthetic": 3.0, "technical": 3.0}
    assert sample.source_labels.aesthetic.mean == 5
    assert sample.image_decode_warnings == ()
    assert plan.manifest.qwen_input.model_id == QWEN35_MODEL_ID
    assert plan.manifest.qwen_input.revision == QWEN35_REVISION
    assert plan.manifest.qwen_input.chat_template_sha256 == QWEN35_CHAT_TEMPLATE_SHA256
    assert plan.manifest.qwen_input.enable_thinking is False
    assert plan.manifest.assistant_loss_only is True


def test_prepare_eva_sft_writes_once(tmp_path: Path) -> None:
    write_sources(tmp_path)

    plan = make_plan(tmp_path, dry_run=False)

    assert (
        len((plan.output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines())
        == 2
    )
    assert (
        len((plan.output_dir / "val.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )
    assert (
        len((plan.output_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )
    manifest = json.loads(
        (plan.output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["output_contract"] == "eva_dual_score_v1"
    assert manifest["eval_fingerprint"] == plan.split.eval_fingerprint
    with pytest.raises(FileExistsError, match="不覆盖"):
        make_plan(tmp_path, dry_run=False)


def test_split_is_reproducible(tmp_path: Path) -> None:
    write_sources(tmp_path, tuple(str(index) for index in range(1, 21)))

    first = make_plan(tmp_path)
    second = make_plan(tmp_path)

    assert first.split == second.split
    assert first.manifest.eval_fingerprint == second.manifest.eval_fingerprint
    all_ids = set(first.split.train) | set(first.split.val) | set(first.split.eval)
    assert len(all_ids) == 20


def test_prepare_rejects_missing_image_record(tmp_path: Path) -> None:
    _, images = write_sources(tmp_path)
    path = tmp_path / "manifests" / "eva" / "images.jsonl"
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in images[:-1]),
        encoding="utf-8",
    )

    with pytest.raises(EvaPreparationError, match="清单不一致"):
        make_plan(tmp_path)


def test_prepare_rejects_changed_image(tmp_path: Path) -> None:
    write_sources(tmp_path)
    (tmp_path / "images" / "eva" / "1.jpg").write_bytes(b"changed")

    with pytest.raises(EvaPreparationError, match="哈希"):
        make_plan(tmp_path)


def test_eva_sample_rejects_extra_target_key(tmp_path: Path) -> None:
    write_sources(tmp_path)
    sample = make_plan(tmp_path).samples[0]
    data: dict[str, Any] = sample.model_dump()
    data["messages"][-1]["content"][0][
        "text"
    ] = '{"aesthetic":3.0,"technical":3.0,"overall":3.0}'

    with pytest.raises(ValidationError, match="合法双分 JSON"):
        EvaSftSample.model_validate(data)


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.8, "val": 0.1, "eval": 0.2},
        {"train": 1.0, "val": 0.0, "eval": 0.0},
    ],
)
def test_split_ratios_are_strict(ratios: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SplitRatios.model_validate(ratios)
