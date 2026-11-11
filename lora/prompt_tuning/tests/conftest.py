from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image
from ptune.schema import JobConfig, PromptTuningSet, TuningItem

HERE = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
# image id -> (split, role, aesthetic mean 0-10, technical mean 1-4)
ROWS: dict[str, tuple[str, str, float, float]] = {
    "photo_000001": ("train", "item", 4.0, 2.0),
    "photo_000002": ("train", "item", 5.0, 2.5),
    "photo_000003": ("val", "item", 6.0, 2.2),
    "photo_000004": ("val", "item", 7.0, 3.1),
    "photo_000005": ("eval", "item", 5.5, 2.4),
    "photo_000006": ("eval", "item", 8.0, 3.4),
    "photo_000007": ("eval", "item", 3.0, 1.8),
    "photo_000101": ("train", "example", 4.5, 2.6),
    "photo_000102": ("train", "example", 7.5, 3.2),
}


def fingerprint(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def target(aesthetic: float, technical: float) -> str:
    a = round(1 + 4 * aesthetic / 10, 1)
    t = round(1 + 4 * (technical - 1) / 3, 1)
    return json.dumps({"aesthetic": a, "technical": t}, separators=(",", ":"))


@dataclass(frozen=True)
class JobFiles:
    root: Path
    job: Path


def write_job(root: Path, rows: dict[str, tuple[str, str, float, float]]) -> JobFiles:
    """Lay out a job root like an unpacked bundle."""

    data = root / "data"
    (data / "images" / "eva").mkdir(parents=True)
    shutil.copytree(HERE / "prompts", root / "prompts")
    items = []
    for offset, (image_id, (split, role, aesthetic, technical)) in enumerate(
        rows.items()
    ):
        path = data / "images" / "eva" / f"{int(image_id[6:])}.jpg"
        Image.new("RGB", (16 + offset, 16), (offset * 20, 0, 0)).save(path, "JPEG")
        items.append(
            TuningItem(
                image_id=image_id,
                split=split,  # type: ignore[arg-type]
                role=role,  # type: ignore[arg-type]
                image_path=path.relative_to(data).as_posix(),
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                aesthetic_mean=aesthetic,
                technical_mean=technical,
                reply_text=target(aesthetic, technical),
            )
        )
    scored = [i.image_id for i in items if i.role == "item"]
    tuning_set = PromptTuningSet(
        schema_version="1.0",
        version="v1",
        created_at=NOW,
        sft_manifest="manifests/eva/sft/v1/manifest.json",
        sft_manifest_sha256="d" * 64,
        example_strategy="aesthetic_quantiles",
        example_ids=tuple(i.image_id for i in items if i.role == "example"),
        item_count=len(scored),
        ids_fingerprint=fingerprint(scored),
        eval_ids_fingerprint=fingerprint(
            [i.image_id for i in items if i.role == "item" and i.split == "eval"]
        ),
    )
    tuning_bytes = (tuning_set.model_dump_json(indent=2) + "\n").encode("utf-8")
    items_bytes = "".join(
        json.dumps(i.model_dump(mode="json"), ensure_ascii=False) + "\n" for i in items
    ).encode("utf-8")
    (data / "tuning-set.json").write_bytes(tuning_bytes)
    (data / "items.jsonl").write_bytes(items_bytes)
    job = JobConfig(
        schema_version="1.0",
        train_config="configs/eva-qwen35-4b-lora-r16.toml",
        train_config_sha256="e" * 64,
        model={"model_id": "Qwen/Qwen3.5-4B", "revision": "a" * 40},
        chat_template_sha256="b" * 64,
        preprocess={"min_pixels": 4096, "max_pixels": 65536},
        precision="bf16",
        max_new_tokens=32,
        min_parse_rate=0.95,
        tuning_set={
            "path": "data/tuning-set.json",
            "sha256": hashlib.sha256(tuning_bytes).hexdigest(),
        },
        items={
            "path": "data/items.jsonl",
            "sha256": hashlib.sha256(items_bytes).hexdigest(),
        },
        weights=[{"path": "chat_template.jinja", "sha256": "b" * 64, "size_bytes": 10}],
    )
    job_path = data / "job.json"
    job_path.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return JobFiles(root, job_path)


@pytest.fixture
def job_files(tmp_path: Path) -> JobFiles:
    return write_job(tmp_path, ROWS)
