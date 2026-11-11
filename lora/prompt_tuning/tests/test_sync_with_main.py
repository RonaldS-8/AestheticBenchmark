"""The copies in this folder must behave exactly like the main repository.

Runs only inside the main repository (skipped in the uploaded bundle, which
has no ``lora_sft``). Checks that shared records have identical schemas, the
prompt files are byte-identical, messages are assembled the same way and a
report built here equals the one the main repository builds.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from ptune import metrics as job_metrics
from ptune import prompts as job_prompts
from ptune import runner as job_runner
from ptune import schema as job_schema

from tests.conftest import HERE, NOW

REPO = HERE.parent
if not (REPO / "lora_sft").is_dir():
    pytest.skip("不在主仓库里，跳过一致性检查", allow_module_level=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lora_sft.data import eva_sft as main_eva_sft  # noqa: E402
from lora_sft.eval import eva_baseline as main_baseline  # noqa: E402
from lora_sft.eval import eva_eval as main_eval  # noqa: E402
from lora_sft.eval import metrics as main_metrics  # noqa: E402
from lora_sft.eval import prompts as main_prompts  # noqa: E402
from lora_sft.eval import schema as main_schema  # noqa: E402
from lora_sft.train import config as main_config  # noqa: E402

SHARED = [
    (job_schema.EvaDualScoreOutput, main_eva_sft.EvaDualScoreOutput),
    (job_schema.BaseModelRef, main_config.BaseModelRef),
    (job_schema.ImagePreprocess, main_config.ImagePreprocess),
    (job_schema.AdapterRef, main_eval.AdapterRef),
    (job_schema.EvaEvalRunConfig, main_eval.EvaEvalRunConfig),
    (job_schema.EvaPrediction, main_eval.EvaPrediction),
    (job_schema.DimensionMetrics, main_eval.DimensionMetrics),
    (job_schema.EvaEvalReport, main_eval.EvaEvalReport),
    (job_schema.GgufWeights, main_schema.GgufWeights),
    (job_schema.PromptTuningSet, main_baseline.PromptTuningSet),
    (job_schema.TuningSetRef, main_baseline.TuningSetRef),
    (job_schema.EvaBaselineRunConfig, main_baseline.EvaBaselineRunConfig),
    (job_schema.PromptCandidate, main_baseline.PromptCandidate),
    (job_schema.PromptSelection, main_baseline.PromptSelection),
]


def _strip(node: object) -> object:
    """Drop docstring-derived descriptions; everything else must match."""

    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


@pytest.mark.parametrize("pair", SHARED, ids=lambda p: p[0].__name__)
def test_shared_record_schemas_are_identical(pair: tuple[Any, Any]) -> None:
    copy, original = pair

    assert _strip(copy.model_json_schema()) == _strip(original.model_json_schema())


def test_method_enum_is_identical() -> None:
    assert [m.value for m in job_schema.Method] == [m.value for m in main_schema.Method]
    assert {m.value for m in job_schema.FEW_SHOT_METHODS} == {
        m.value for m in main_schema.FEW_SHOT_METHODS
    }


def test_prompt_files_are_byte_identical() -> None:
    source = sorted((REPO / "prompts" / "baseline-v2").glob("*.prompt.md"))
    copies = sorted((HERE / "prompts" / "baseline-v2").glob("*.prompt.md"))

    assert [p.name for p in copies] == [p.name for p in source]
    for original, copy in zip(source, copies, strict=True):
        assert copy.read_bytes() == original.read_bytes(), copy.name


@pytest.mark.parametrize(
    "prompt_id", ["zero-shot-score@v2", "rubric-score@v2", "few-shot-score@v2"]
)
def test_messages_are_assembled_identically(prompt_id: str) -> None:
    copy = job_prompts.find_prompt(prompt_id, HERE / "prompts")
    original = main_prompts.find_prompt(prompt_id, REPO / "prompts")
    ids = ["photo_000010", "photo_000011"] if copy.is_few_shot else []
    reply = '{"aesthetic":3.5,"technical":3.2}'

    built = job_prompts.build_messages(
        copy,
        "photo_000001",
        "images/eva/1.jpg",
        [
            job_prompts.FewShotExample(
                image_id=i, image_path=f"images/eva/{i[6:]}.jpg", reply_text=reply
            )
            for i in ids
        ],
    )
    expected = main_prompts.build_messages(
        original,
        "photo_000001",
        "images/eva/1.jpg",
        [
            main_prompts.FewShotExample(
                media_id=i, media_path=f"images/eva/{i[6:]}.jpg", reply_text=reply
            )
            for i in ids
        ],
    )

    assert [m.model_dump() for m in built] == [m.model_dump() for m in expected]


def test_metrics_and_parser_match() -> None:
    rng = random.Random(0)
    x = [rng.uniform(1, 5) for _ in range(50)]
    y = [v + rng.gauss(0, 0.5) for v in x]
    replies = [
        '{"aesthetic":3.2,"technical":2.0}',
        "```json\n{}\n```",
        "x",
        '<think>\n</think>\n```json\n{"aesthetic":3.2,"technical":2.0}\n```',
        'ok ```json\n{"aesthetic":3.2,"technical":2.0}\n```',
    ]

    assert job_metrics.srcc(x, y) == main_metrics.srcc(x, y)
    assert job_metrics.plcc(x, y) == main_metrics.plcc(x, y)
    assert job_metrics.percentile(x, 95) == main_metrics.percentile(x, 95)
    for reply in replies:
        copy = job_runner.parse_reply("b1-tune", "photo_000001", reply, 5)
        original = main_eval.parse_eva_reply("b1-tune", "photo_000001", reply, 5)
        assert copy.model_dump() == original.model_dump()


def _load_main_test_module() -> ModuleType:
    path = REPO / "tests" / "test_data_eva_sft.py"
    spec = importlib.util.spec_from_file_location("main_test_data_eva_sft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packed_job_reports_like_the_main_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pack import job_files

    helpers = _load_main_test_module()
    monkeypatch.chdir(REPO)
    data_root = tmp_path / "data"
    helpers.write_sources(data_root, tuple(str(i) for i in range(1, 21)))
    plan = main_eva_sft.prepare_eva_sft(
        data_root,
        helpers.PROMPT,
        version="v1",
        seed=42,
        ratios=main_eva_sft.SplitRatios(train=0.6, val=0.2, eval=0.2),
        created_at=helpers.NOW,
        dry_run=False,
    )
    by_split = {
        name: tuple(s for s in plan.samples if s.split == name)
        for name in ("train", "val", "eval")
    }
    manifest = data_root / "manifests" / "eva" / "sft" / "v1" / "manifest.json"
    tuning_set = main_baseline.build_tuning_set(
        by_split["train"],
        by_split["val"],
        by_split["eval"],
        k=2,
        version="v1",
        sft_manifest="manifests/eva/sft/v1/manifest.json",
        sft_manifest_sha256=job_runner.sha256_file(manifest),
        now=NOW,
    )
    config_path = REPO / "configs" / "eva-qwen35-4b-lora-r16.toml"
    config = main_config.load_train_config(config_path)
    files, rows = job_files(
        config,
        "configs/eva-qwen35-4b-lora-r16.toml",
        config_path.read_bytes(),
        (tuning_set.model_dump_json(indent=2) + "\n").encode("utf-8"),
        data_root,
        (
            job_schema.WeightFile(
                path="chat_template.jinja",
                sha256=config.chat_template_sha256,
                size_bytes=1,
            ),
        ),
    )
    job = job_schema.JobConfig.model_validate_json(files["data/job.json"])
    items = [r for r in rows if r.role == "item"]
    samples, _ = main_baseline.tuning_samples(
        tuning_set, by_split["train"], by_split["val"], by_split["eval"]
    )
    rng = random.Random(1)
    replies = {
        r.image_id: json.dumps(
            {
                "aesthetic": round(rng.uniform(1, 5), 1),
                "technical": round(rng.uniform(1, 5), 1),
            }
        )
        for r in items
    }
    replies[items[0].image_id] = "解析不了"
    scoring = {
        "schema_version": "1.0",
        "run_id": "b1-tune",
        "split": "all",
        "ids_fingerprint": tuning_set.ids_fingerprint,
        "model": job.model.model_dump(),
        "adapter": None,
        "prompt_id": "rubric-score@v2",
        "chat_template_sha256": job.chat_template_sha256,
        "preprocess": job.preprocess.model_dump(),
        "max_new_tokens": job.max_new_tokens,
        "device": "cuda",
    }

    built = job_runner.build_report(
        job_schema.EvaEvalRunConfig.model_validate(scoring),
        [job_runner.parse_reply("b1-tune", i, t, 10.0) for i, t in replies.items()],
        items,
        now=NOW,
    )
    expected = main_eval.build_eva_report(
        main_eval.EvaEvalRunConfig.model_validate(scoring),
        [main_eval.parse_eva_reply("b1-tune", i, t, 10.0) for i, t in replies.items()],
        samples,
        now=NOW,
    )

    assert built.model_dump() == expected.model_dump()
    assert job.model.model_dump() == config.base_model.model_dump()
    assert job.min_parse_rate == config.selection.min_parse_rate
    assert job.max_new_tokens == config.selection.max_new_tokens
    assert json.loads(files["data/tuning-set.json"]) == json.loads(
        tuning_set.model_dump_json()
    )
