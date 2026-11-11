"""The copies in this folder must behave exactly like the main repository.

Runs only inside the main repository (skipped once this folder is used on its
own). Checks the copied metrics, score summary and report schema, that a head
report equals the LoRA report built from the same scores, and that the dataset
copy keeps the main repository's targets, splits and eval fingerprint.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest
from aes_head import metrics as head_metrics
from aes_head.data import load_dataset
from aes_head.evaluation import build_report
from aes_head.prepare import plan_copy
from aes_head.records import EvaEvalReport, HeadEvalRunConfig, HeadPrediction
from aes_head.schema import DimensionMetrics, ScoreSummary
from tests.helpers import HERE, NOW

REPO = HERE.parent / "lora"
if not (REPO / "lora_sft").is_dir():
    pytest.skip("不在主仓库里，跳过一致性检查", allow_module_level=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lora_sft.crawl import dataset_schema as main_dataset_schema  # noqa: E402
from lora_sft.data import eva_sft as main_eva_sft  # noqa: E402
from lora_sft.data.schema import eval_fingerprint  # noqa: E402
from lora_sft.eval import eva_eval as main_eva_eval  # noqa: E402
from lora_sft.eval import metrics as main_metrics  # noqa: E402
from lora_sft.train import config as main_config  # noqa: E402


def _strip(node: object) -> object:
    """Drop docstring-derived descriptions; everything else must match."""

    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def test_score_summary_schema_is_identical() -> None:
    copy = ScoreSummary.model_json_schema()
    original = main_dataset_schema.ScoreSummary.model_json_schema()

    assert _strip(copy) == _strip(original)


def test_metrics_match() -> None:
    rng = random.Random(0)
    x = [rng.uniform(1, 5) for _ in range(60)]
    y = [round(v + rng.gauss(0, 0.5), 1) for v in x]

    assert head_metrics.srcc(x, y) == main_metrics.srcc(x, y)
    assert head_metrics.plcc(x, y) == main_metrics.plcc(x, y)
    assert head_metrics.percentile(x, 95) == main_metrics.percentile(x, 95)


def _load_main_test_module() -> ModuleType:
    path = REPO / "tests" / "test_data_eva_sft.py"
    spec = importlib.util.spec_from_file_location("main_test_data_eva_sft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main_plan(data_root: Path) -> main_eva_sft.EvaSftPlan:
    """EVA SFT samples written by the main repository's own code."""

    helpers = _load_main_test_module()
    helpers.write_sources(data_root, tuple(str(i) for i in range(1, 21)))
    return main_eva_sft.prepare_eva_sft(
        data_root,
        helpers.PROMPT,
        version="v1",
        seed=42,
        ratios=main_eva_sft.SplitRatios(train=0.6, val=0.2, eval=0.2),
        created_at=helpers.NOW,
        dry_run=False,
    )


@pytest.mark.parametrize(
    "pair",
    [
        (EvaEvalReport, main_eva_eval.EvaEvalReport),
        (DimensionMetrics, main_eva_eval.DimensionMetrics),
    ],
    ids=lambda p: p[0].__name__,
)
def test_report_schemas_are_identical(pair: tuple[type, type]) -> None:
    copy, original = pair

    assert _strip(copy.model_json_schema()) == _strip(original.model_json_schema())  # type: ignore[attr-defined]


def test_head_report_equals_lora_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    data_root = tmp_path / "data"
    plan = _main_plan(data_root)
    copy = plan_copy(data_root, "manifests/eva/sft/v1/manifest.json", now=NOW)
    samples = [s for s in plan.samples if s.split == "val"]
    items = [i for i in copy.items if i.split == "val"]
    rng = random.Random(3)
    scores = {
        s.image_id: (round(rng.uniform(1, 5), 2), round(rng.uniform(1, 5), 2))
        for s in samples
    }
    fingerprint = eval_fingerprint(scores)
    lora = main_config.load_train_config(
        REPO / "configs" / "eva-qwen35-4b-lora-r16.toml"
    )
    main_config_json = {
        "schema_version": "1.0",
        "run_id": "head-val",
        "split": "val",
        "ids_fingerprint": fingerprint,
        "model": lora.base_model.model_dump(),
        "adapter": None,
        "prompt_id": "zero-shot-score@v2",
        "chat_template_sha256": lora.chat_template_sha256,
        "preprocess": lora.preprocess.model_dump(),
        "max_new_tokens": 64,
        "device": "cuda",
    }
    expected = main_eva_eval.build_eva_report(
        main_eva_eval.EvaEvalRunConfig.model_validate(main_config_json),
        [
            main_eva_eval.EvaPrediction(
                run_id="head-val",
                image_id=image_id,
                raw_output="",
                scores=main_eva_sft.EvaDualScoreOutput(aesthetic=a, technical=t),
                parse_error=None,
                latency_ms=5.0 + k,
            )
            for k, (image_id, (a, t)) in enumerate(scores.items())
        ],
        samples,
        now=NOW,
    )
    head_config = HeadEvalRunConfig.model_validate(
        {
            "schema_version": "1.0",
            "run_id": "head-val",
            "split": "val",
            "ids_fingerprint": fingerprint,
            "train_run_id": "head-train",
            "checkpoint": {"path": "head-train/best.pt", "sha256": "c" * 64},
            "device": "cuda",
            "precision": "bf16",
        }
    )

    built = build_report(
        head_config,
        [
            HeadPrediction(
                run_id="head-val",
                image_id=image_id,
                aesthetic=a,
                technical=t,
                latency_ms=5.0 + k,
            )
            for k, (image_id, (a, t)) in enumerate(scores.items())
        ],
        items,
        now=NOW,
    )

    assert built.model_dump() == expected.model_dump()


def test_copy_keeps_main_targets_and_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    data_root = tmp_path / "data"
    plan = _main_plan(data_root)

    copy = plan_copy(data_root, "manifests/eva/sft/v1/manifest.json", now=NOW)

    by_id = {item.image_id: item for item in copy.items}
    assert sorted(by_id) == sorted(s.image_id for s in plan.samples)
    for sample in plan.samples:
        item = by_id[sample.image_id]
        assert item.split == sample.split
        for dim in ("aesthetic", "technical"):
            assert item.summary(dim).to_reported_target() == getattr(sample.target, dim)
    eval_ids = [s.image_id for s in plan.samples if s.split == "eval"]
    assert copy.info.eval_fingerprint == eval_fingerprint(eval_ids)


def test_local_copy_matches_main_data() -> None:
    folder = REPO / "data" / "manifests" / "eva" / "sft" / "v1"
    if not (folder.is_dir() and (HERE / "data" / "dataset.json").is_file()):
        pytest.skip("本机没有 EVA 数据或还没复制")
    dataset = load_dataset(HERE / "data")
    by_id = {item.image_id: item for item in dataset.items}
    for name in ("train", "val", "eval"):
        lines = (folder / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        for line in lines:
            sample = main_eva_sft.EvaSftSample.model_validate_json(line)
            item = by_id.pop(sample.image_id)
            assert item.split == sample.split
            assert item.image_sha256 == sample.image_sha256
            assert item.aesthetic.to_reported_target() == sample.target.aesthetic
            assert item.technical.to_reported_target() == sample.target.technical
    assert not by_id
