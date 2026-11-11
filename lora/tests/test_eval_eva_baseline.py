from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from lora_sft.data.eva_sft import (
    EvaPreparationError,
    EvaSftSample,
    SplitRatios,
    prepare_eva_sft,
)
from lora_sft.data.samples import Message, TextPart, VideoPart
from lora_sft.data.schema import eval_fingerprint
from lora_sft.eval.eva_baseline import (
    BaselineError,
    EvaBaselineRunConfig,
    PromptSelection,
    PromptTuningSet,
    build_tuning_set,
    comparison_table,
    fit_size,
    hf_chat_messages,
    lora_gain,
    openai_chat_messages,
    prepare_baseline_run_dir,
    run_baseline,
    select_best_prompt,
    select_quantile_examples,
    to_few_shot_examples,
    tuning_samples,
    write_baseline_report,
    write_eval_slice_report,
    write_tuning_set,
)
from lora_sft.eval.eva_eval import EvaEvalReport, EvaEvalRunConfig, read_predictions
from lora_sft.eval.prompts import PromptTemplate, find_prompt
from lora_sft.train.config import ImagePreprocess
from lora_sft.train.eva_data import load_eva_samples
from tests.test_data_eva_sft import NOW, write_sources
from tests.test_train_eva_data import SAMPLES, repo  # noqa: F401

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
PREPROCESS = ImagePreprocess(min_pixels=4096, max_pixels=65536)
MODEL = {"model_id": "Qwen/Qwen3.5-4B", "revision": "a" * 40}
TUNING_REF = {"path": "data/manifests/eva/prompt-tuning/v1.json", "sha256": "e" * 64}
GGUF = {
    "model_file": "Qwen3.5-4B-Q4_K_M.gguf",
    "model_sha256": "3" * 64,
    "mmproj_file": "mmproj-Qwen3.5-4B-f16.gguf",
    "mmproj_sha256": "4" * 64,
    "quantization": "q4_k_m",
}

Splits = dict[str, tuple[EvaSftSample, ...]]


@pytest.fixture
def splits(repo: Path) -> Splits:  # noqa: F811
    return {
        name: load_eva_samples(repo / SAMPLES / f"{name}.jsonl", name)
        for name in ("train", "val", "eval")
    }


def make_tuning(splits: Splits, k: int = 2) -> PromptTuningSet:
    return build_tuning_set(
        splits["train"],
        splits["val"],
        splits["eval"],
        k=k,
        version="v1",
        sft_manifest="manifests/eva/sft/v1/manifest.json",
        sft_manifest_sha256="d" * 64,
        now=NOW,
    )


@pytest.fixture
def tuning(splits: Splits) -> PromptTuningSet:
    return make_tuning(splits)


@pytest.fixture
def items(splits: Splits, tuning: PromptTuningSet) -> tuple[EvaSftSample, ...]:
    return tuning_samples(tuning, splits["train"], splits["val"], splits["eval"])[0]


@pytest.fixture
def examples(splits: Splits, tuning: PromptTuningSet) -> tuple[EvaSftSample, ...]:
    return tuning_samples(tuning, splits["train"], splits["val"], splits["eval"])[1]


def config_data(
    samples: tuple[EvaSftSample, ...],
    method: str = "b1_rubric",
    prompt_id: str = "rubric-score@v2",
    **overrides: object,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": "1.0",
        "scoring": {
            "schema_version": "1.0",
            "run_id": f"{method.replace('_', '-')}-tune",
            "split": "all",
            "ids_fingerprint": eval_fingerprint(s.image_id for s in samples),
            "model": MODEL,
            "adapter": None,
            "prompt_id": prompt_id,
            "chat_template_sha256": "b" * 64,
            "preprocess": PREPROCESS.model_dump(),
            "max_new_tokens": 32,
            "device": "cuda",
        },
        "tuning_set": TUNING_REF,
        "method": method,
        "runtime": "transformers",
        "gguf": None,
    }
    data.update(overrides)
    return data


def make_config(
    samples: tuple[EvaSftSample, ...], **kw: object
) -> EvaBaselineRunConfig:
    return EvaBaselineRunConfig.model_validate(config_data(samples, **kw))


def prompt(prompt_id: str = "rubric-score@v2") -> PromptTemplate:
    return find_prompt(prompt_id, PROMPTS)


class EchoModel:
    """A fake model that answers each image with its SFT target JSON."""

    def __init__(self, samples: tuple[EvaSftSample, ...]) -> None:
        self.by_path = {s.image_path: s for s in samples}
        self.seen: list[tuple[Message, ...]] = []

    def __call__(self, messages: tuple[Message, ...]) -> str:
        self.seen.append(messages)
        return self.by_path[messages[-1].images[0].image].messages[-1].text


# --- tuning set ---


def test_tuning_set_covers_everything_but_examples(
    splits: Splits, tuning: PromptTuningSet, items: tuple[EvaSftSample, ...]
) -> None:
    every = {s.image_id for part in splits.values() for s in part}

    assert len(tuning.example_ids) == 2
    assert {s.image_id for s in items} == every - set(tuning.example_ids)
    assert tuning.item_count == len(every) - 2
    assert tuning.eval_ids_fingerprint == eval_fingerprint(
        s.image_id for s in splits["eval"]
    )
    assert {s.split for s in items} == {"train", "val", "eval"}


def test_tuning_samples_detect_changed_data(
    splits: Splits, tuning: PromptTuningSet
) -> None:
    with pytest.raises(BaselineError, match="指纹不符"):
        tuning_samples(tuning, splits["train"], splits["val"], splits["eval"][:1])
    with pytest.raises(BaselineError, match="不在 train 里"):
        tuning_samples(tuning, splits["val"], splits["val"], splits["eval"])


def test_tuning_set_rejects_overlap_and_bad_eval(splits: Splits) -> None:
    with pytest.raises(BaselineError, match="重复 image_id"):
        build_tuning_set(
            splits["train"],
            splits["train"],
            splits["eval"],
            k=1,
            version="v1",
            sft_manifest="m.json",
            sft_manifest_sha256="d" * 64,
            now=NOW,
        )
    with pytest.raises(BaselineError, match="split 必须都是 eval"):
        build_tuning_set(
            splits["train"],
            splits["val"],
            splits["val"],
            k=1,
            version="v1",
            sft_manifest="m.json",
            sft_manifest_sha256="d" * 64,
            now=NOW,
        )


def test_write_tuning_set_never_overwrites(
    tmp_path: Path, tuning: PromptTuningSet
) -> None:
    path = tmp_path / "prompt-tuning" / "v1.json"

    write_tuning_set(path, tuning)

    assert PromptTuningSet.model_validate_json(path.read_text("utf-8")) == tuning
    with pytest.raises(BaselineError, match="不覆盖"):
        write_tuning_set(path, tuning)


def test_tuning_set_example_ids_unique(tuning: PromptTuningSet) -> None:
    data = tuning.model_dump() | {"example_ids": ["photo_000001"] * 2}

    with pytest.raises(ValidationError, match="example_ids 有重复"):
        PromptTuningSet.model_validate(data)


def test_select_quantile_examples_spans_scores(splits: Splits) -> None:
    train = splits["train"]
    ordered = sorted(train, key=lambda s: (s.source_labels.aesthetic.mean, s.image_id))

    assert select_quantile_examples(train, 3) == (ordered[1], ordered[3], ordered[5])
    assert select_quantile_examples(train, len(train)) == tuple(ordered)


@pytest.mark.parametrize("k", [0, 7])
def test_select_quantile_examples_bounds(splits: Splits, k: int) -> None:
    with pytest.raises(BaselineError, match="k="):
        select_quantile_examples(splits["train"], k)


def test_select_quantile_examples_only_from_train(splits: Splits) -> None:
    with pytest.raises(BaselineError, match="只能取自 train"):
        select_quantile_examples(splits["val"], 1)


def test_few_shot_replies_are_sft_targets(splits: Splits) -> None:
    train = splits["train"]
    shots = to_few_shot_examples(train[:2])

    assert [e.reply_text for e in shots] == [s.messages[-1].text for s in train[:2]]
    assert shots[0].media_path == train[0].image_path


# --- config ---


def test_config_round_trip(items: tuple[EvaSftSample, ...]) -> None:
    config = make_config(items)

    assert config.run_id == "b1-rubric-tune" and not config.is_few_shot
    assert EvaBaselineRunConfig.model_validate_json(config.model_dump_json()) == config


def test_llama_cpp_config_with_gguf(items: tuple[EvaSftSample, ...]) -> None:
    config = make_config(items, runtime="llama_cpp", gguf=GGUF)

    assert config.gguf is not None and config.gguf.quantization == "q4_k_m"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "lora_sft"}, "不是提示词基线"),
        ({"method": "b3_retrieval"}, "B3 检索增强还没有实现"),
        ({"runtime": "llama_cpp"}, "必须且只能填写 gguf"),
        ({"gguf": GGUF}, "必须且只能填写 gguf"),
        ({"runtime": "mlx"}, "runtime"),
        ({"tuning_set": {"path": "x.json", "sha256": "x"}}, "sha256"),
    ],
)
def test_config_rules(
    items: tuple[EvaSftSample, ...], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_config(items, **overrides)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("split", "eval", "split=all"),
        (
            "adapter",
            {
                "train_run_name": "eva-qwen35-4b-lora-r16",
                "path": "outputs/x/adapter",
                "weights_sha256": "c" * 64,
            },
            "不加载 LoRA",
        ),
    ],
)
def test_config_scoring_rules(
    items: tuple[EvaSftSample, ...], key: str, value: object, message: str
) -> None:
    data = config_data(items)
    data["scoring"][key] = value

    with pytest.raises(ValidationError, match=message):
        EvaBaselineRunConfig.model_validate(data)


# --- runs ---


def test_run_scores_resumes_and_reports(
    repo: Path,  # noqa: F811
    splits: Splits,
    tuning: PromptTuningSet,
    items: tuple[EvaSftSample, ...],
) -> None:
    config = make_config(items)
    run_dir = repo / "runs" / config.run_id
    model = EchoModel(items)

    first = run_baseline(
        config, prompt(), tuning, items, (), run_dir=run_dir, complete=model
    )
    second = run_baseline(
        config, prompt(), tuning, items, (), run_dir=run_dir, complete=model
    )
    report = write_baseline_report(config, items, run_dir=run_dir, now=NOW)
    sliced = write_eval_slice_report(
        config, tuning, splits["eval"], run_dir=run_dir, now=NOW
    )

    assert (first, second) == (len(items), 0)
    assert [m.role for m in model.seen[0]] == ["system", "user"]
    assert {p.image_id for p in read_predictions(run_dir)} == {
        s.image_id for s in items
    }
    assert (report.split, report.n_items) == ("all", len(items))
    assert report.metrics["aesthetic"].srcc == pytest.approx(1.0)
    # the fixture's technical vote means are constant, so SRCC is undefined
    assert "technical" in report.metric_errors
    assert (sliced.split, sliced.n_items) == ("eval", len(splits["eval"]))
    assert sliced.run_id == config.run_id
    assert sliced.ids_fingerprint == tuning.eval_ids_fingerprint
    saved = EvaEvalReport.model_validate_json(
        (run_dir / "report-eval.json").read_text(encoding="utf-8")
    )
    assert saved == sliced


def test_eval_slice_rejects_other_eval_set(
    repo: Path,  # noqa: F811
    splits: Splits,
    tuning: PromptTuningSet,
    items: tuple[EvaSftSample, ...],
) -> None:
    with pytest.raises(BaselineError, match="eval 指纹不符"):
        write_eval_slice_report(
            make_config(items), tuning, splits["val"], run_dir=repo, now=NOW
        )


def test_run_records_parse_failures(
    repo: Path,  # noqa: F811
    tuning: PromptTuningSet,
    items: tuple[EvaSftSample, ...],
) -> None:
    config = make_config(items, method="b0_zero_shot", prompt_id="zero-shot-score@v2")
    run_dir = repo / "runs" / config.run_id
    seen: list[int] = []

    run_baseline(
        config,
        prompt("zero-shot-score@v2"),
        tuning,
        items,
        (),
        run_dir=run_dir,
        complete=lambda m: "Scores: aesthetic 3.0, technical 3.0",
        on_progress=lambda done, total, p: seen.append(done),
    )

    assert seen == list(range(1, len(items) + 1))
    assert all(p.scores is None and p.parse_error for p in read_predictions(run_dir))


def test_few_shot_run_puts_examples_first(
    repo: Path,  # noqa: F811
    tuning: PromptTuningSet,
    items: tuple[EvaSftSample, ...],
    examples: tuple[EvaSftSample, ...],
) -> None:
    config = make_config(items, method="b2_few_shot", prompt_id="few-shot-score@v2")
    model = EchoModel(items)

    run_baseline(
        config,
        prompt("few-shot-score@v2"),
        tuning,
        items,
        examples,
        run_dir=repo / "runs" / config.run_id,
        complete=model,
    )

    messages = model.seen[0]
    assert config.is_few_shot
    assert [m.role for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1].images[0].image == examples[0].image_path
    assert messages[2].text == examples[0].messages[-1].text


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("prompt", "与配置"),
        ("method", "不用于图片上的"),
        ("fingerprint", "ids_fingerprint"),
        ("example_scored", "不能出现在被评的图片里"),
        ("examples", "example_ids"),
    ],
)
def test_run_rejects_mismatches(
    repo: Path,  # noqa: F811
    tuning: PromptTuningSet,
    items: tuple[EvaSftSample, ...],
    examples: tuple[EvaSftSample, ...],
    change: str,
    message: str,
) -> None:
    samples = items
    config = make_config(samples)
    used_prompt = prompt()
    shots: tuple[EvaSftSample, ...] = ()
    if change == "prompt":
        used_prompt = prompt("zero-shot-score@v2")
    elif change == "method":
        config = make_config(samples, prompt_id="eva-dual-score@v1")
        used_prompt = prompt("eva-dual-score@v1")
    elif change == "fingerprint":
        samples = items[:1]
    elif change == "example_scored":
        samples = (*items, examples[0])
        config = make_config(samples)
    else:
        shots = examples

    with pytest.raises(BaselineError, match=message):
        run_baseline(
            config,
            used_prompt,
            tuning,
            samples,
            shots,
            run_dir=repo / "runs" / "x",
            complete=EchoModel(items),
        )


def test_resume_rejects_changed_config(
    repo: Path, items: tuple[EvaSftSample, ...]  # noqa: F811
) -> None:
    run_dir = repo / "runs" / "b1-rubric-tune"
    prepare_baseline_run_dir(run_dir, make_config(items))
    changed = make_config(items, runtime="llama_cpp", gguf=GGUF)

    with pytest.raises(BaselineError, match="不能续跑"):
        prepare_baseline_run_dir(run_dir, changed)


def test_resume_rejects_foreign_predictions(
    repo: Path, items: tuple[EvaSftSample, ...]  # noqa: F811
) -> None:
    config = make_config(items)
    run_dir = repo / "runs" / config.run_id
    prepare_baseline_run_dir(run_dir, config)
    line = {
        "run_id": "other-run",
        "image_id": items[0].image_id,
        "raw_output": "x",
        "scores": None,
        "parse_error": "bad",
        "latency_ms": 1.0,
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(line) + "\n", "utf-8")

    with pytest.raises(BaselineError, match="混有其他运行"):
        prepare_baseline_run_dir(run_dir, config)


# --- best prompt ---


def tune_report(
    config: EvaBaselineRunConfig,
    aesthetic: float,
    technical: float | None,
    rate: float = 1.0,
) -> EvaEvalReport:
    metrics = {"aesthetic": {"srcc": aesthetic, "plcc": aesthetic, "mae_1_5": 0.3}}
    errors = {}
    if technical is None:
        errors["technical"] = "有一组分数全部相同，相关系数无定义"
    else:
        metrics["technical"] = {"srcc": technical, "plcc": technical, "mae_1_5": 0.3}
    return EvaEvalReport.model_validate(
        {
            "schema_version": "1.0",
            "run_id": config.run_id,
            "split": "all",
            "ids_fingerprint": config.scoring.ids_fingerprint,
            "created_at": NOW,
            "n_items": 100,
            "n_parsed": round(100 * rate),
            "parse_success_rate": rate,
            "metrics": metrics,
            "metric_errors": errors,
            "latency_p50_ms": 500.0,
            "latency_p95_ms": 600.0,
        }
    )


def tune_runs(
    items: tuple[EvaSftSample, ...],
    scores: dict[str, tuple[float, float | None, float]],
) -> list[tuple[EvaBaselineRunConfig, EvaEvalReport]]:
    prompts = {
        "b0_zero_shot": "zero-shot-score@v2",
        "b1_rubric": "rubric-score@v2",
        "b2_few_shot": "few-shot-score@v2",
    }
    runs = []
    for method, (aesthetic, technical, rate) in scores.items():
        config = make_config(items, method=method, prompt_id=prompts[method])
        runs.append((config, tune_report(config, aesthetic, technical, rate)))
    return runs


def test_select_best_prompt_keeps_single_image_for_lora(
    items: tuple[EvaSftSample, ...],
) -> None:
    runs = tune_runs(
        items,
        {
            "b0_zero_shot": (0.40, 0.30, 1.0),
            "b1_rubric": (0.55, 0.40, 1.0),
            "b2_few_shot": (0.60, 0.45, 1.0),
        },
    )

    selection = select_best_prompt(runs, min_parse_rate=0.95, now=NOW)

    assert selection.best_prompt_id == "few-shot-score@v2"
    assert selection.lora_prompt_id == "rubric-score@v2"
    scores = {c.prompt_id: c.score for c in selection.candidates}
    assert scores["rubric-score@v2"] == pytest.approx(0.475)


def test_select_best_prompt_skips_ineligible(items: tuple[EvaSftSample, ...]) -> None:
    runs = tune_runs(
        items,
        {
            "b0_zero_shot": (0.40, 0.30, 1.0),
            "b1_rubric": (0.70, 0.60, 0.80),
            "b2_few_shot": (0.65, None, 1.0),
        },
    )

    selection = select_best_prompt(runs, min_parse_rate=0.95, now=NOW)

    assert selection.best_prompt_id == selection.lora_prompt_id == "zero-shot-score@v2"
    by_prompt = {c.prompt_id: c for c in selection.candidates}
    assert "parse_success_rate" in by_prompt["rubric-score@v2"].ineligible
    assert "technical" in by_prompt["few-shot-score@v2"].ineligible


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({"b0_zero_shot": (0.4, 0.3, 1.0), "b1_rubric": (0.3, 0.4, 1.0)}, "并列"),
        ({"b0_zero_shot": (0.4, 0.3, 0.5)}, "没有合格的提示词"),
        ({"b2_few_shot": (0.4, 0.3, 1.0)}, "没有合格的单图提示词"),
    ],
)
def test_select_best_prompt_refuses_to_guess(
    items: tuple[EvaSftSample, ...],
    scores: dict[str, tuple[float, float | None, float]],
    message: str,
) -> None:
    with pytest.raises(BaselineError, match=message):
        select_best_prompt(tune_runs(items, scores), min_parse_rate=0.95, now=NOW)


def test_select_best_prompt_requires_same_setup(
    items: tuple[EvaSftSample, ...],
) -> None:
    runs = tune_runs(
        items, {"b0_zero_shot": (0.4, 0.3, 1.0), "b1_rubric": (0.5, 0.3, 1.0)}
    )
    other = make_config(items, runtime="llama_cpp", gguf=GGUF)
    duplicate = make_config(items, method="b0_zero_shot", prompt_id="rubric-score@v2")

    with pytest.raises(BaselineError, match="没有候选"):
        select_best_prompt([], min_parse_rate=0.95, now=NOW)
    with pytest.raises(BaselineError, match="运行时不同"):
        select_best_prompt(
            [*runs, (other, tune_report(other, 0.5, 0.5))],
            min_parse_rate=0.95,
            now=NOW,
        )
    with pytest.raises(BaselineError, match="不是这次运行在调优集上的报告"):
        select_best_prompt([(runs[0][0], runs[1][1])], min_parse_rate=0.95, now=NOW)
    with pytest.raises(BaselineError, match="出现了多次"):
        select_best_prompt(
            [runs[1], (duplicate, tune_report(duplicate, 0.4, 0.3))],
            min_parse_rate=0.95,
            now=NOW,
        )


def test_selection_rejects_few_shot_lora_prompt(
    items: tuple[EvaSftSample, ...],
) -> None:
    runs = tune_runs(
        items, {"b1_rubric": (0.5, 0.4, 1.0), "b2_few_shot": (0.6, 0.5, 1.0)}
    )
    selection = select_best_prompt(runs, min_parse_rate=0.95, now=NOW)
    data = selection.model_dump() | {"lora_prompt_id": "few-shot-score@v2"}

    with pytest.raises(ValidationError, match="不能是少样本提示词"):
        PromptSelection.model_validate(data)
    with pytest.raises(ValidationError, match="不是合格的候选"):
        PromptSelection.model_validate(
            selection.model_dump() | {"best_prompt_id": "rubric-score@v9"}
        )


# --- LoRA gain ---


def eval_report(run_id: str, aesthetic: float, technical: float) -> EvaEvalReport:
    return EvaEvalReport.model_validate(
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "split": "eval",
            "ids_fingerprint": "7" * 64,
            "created_at": NOW,
            "n_items": 400,
            "n_parsed": 400,
            "parse_success_rate": 1.0,
            "metrics": {
                "aesthetic": {"srcc": aesthetic, "plcc": aesthetic, "mae_1_5": 0.3},
                "technical": {"srcc": technical, "plcc": technical, "mae_1_5": 0.3},
            },
            "metric_errors": {},
            "latency_p50_ms": 500.0,
            "latency_p95_ms": 600.0,
        }
    )


def lora_config(**overrides: object) -> EvaEvalRunConfig:
    data: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": "eva-qwen35-4b-lora-r16-eval",
        "split": "eval",
        "ids_fingerprint": "7" * 64,
        "model": MODEL,
        "adapter": {
            "train_run_name": "eva-qwen35-4b-lora-r16",
            "path": "outputs/eva-qwen35-4b-lora-r16/adapter",
            "weights_sha256": "c" * 64,
        },
        "prompt_id": "rubric-score@v2",
        "chat_template_sha256": "b" * 64,
        "preprocess": PREPROCESS.model_dump(),
        "max_new_tokens": 32,
        "device": "cuda",
    }
    data.update(overrides)
    return EvaEvalRunConfig.model_validate(data)


@pytest.fixture
def gain_inputs(items: tuple[EvaSftSample, ...]) -> dict[str, Any]:
    runs = tune_runs(
        items, {"b0_zero_shot": (0.4, 0.3, 1.0), "b1_rubric": (0.5, 0.4, 1.0)}
    )
    return {
        "selection": select_best_prompt(runs, min_parse_rate=0.95, now=NOW),
        "base_config": runs[1][0],
        "base_eval": eval_report(runs[1][0].run_id, 0.50, 0.35),
        "lora_config": lora_config(),
        "lora_report": eval_report("eva-qwen35-4b-lora-r16-eval", 0.62, 0.41),
    }


def test_lora_gain_is_lora_minus_same_prompt(gain_inputs: dict[str, Any]) -> None:
    gain = lora_gain(**gain_inputs, now=NOW)

    assert gain.prompt_id == "rubric-score@v2"
    assert gain.dimensions["aesthetic"].delta_srcc == pytest.approx(0.12)
    assert gain.dimensions["technical"].delta_plcc == pytest.approx(0.06)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("lora_config", lora_config(prompt_id="zero-shot-score@v2"), "应为选定的"),
        ("lora_config", lora_config(adapter=None), "没有加载 LoRA"),
        (
            "lora_config",
            lora_config(preprocess={"min_pixels": 1, "max_pixels": 2}),
            "像素预算不同",
        ),
        ("lora_config", lora_config(max_new_tokens=64), "max_new_tokens"),
        ("lora_report", eval_report("other-run", 0.6, 0.4), "报告与运行配置不对应"),
    ],
)
def test_lora_gain_requires_fair_comparison(
    gain_inputs: dict[str, Any], key: str, value: object, message: str
) -> None:
    gain_inputs[key] = value

    with pytest.raises(BaselineError, match=message):
        lora_gain(**gain_inputs, now=NOW)


def test_lora_gain_base_must_follow_selection(
    gain_inputs: dict[str, Any], items: tuple[EvaSftSample, ...]
) -> None:
    gain_inputs["base_config"] = make_config(
        items, method="b0_zero_shot", prompt_id="zero-shot-score@v2"
    )

    with pytest.raises(BaselineError, match="不是选定提示词"):
        lora_gain(**gain_inputs, now=NOW)


def test_lora_gain_base_must_be_transformers(
    gain_inputs: dict[str, Any], items: tuple[EvaSftSample, ...]
) -> None:
    gain_inputs["base_config"] = make_config(items, runtime="llama_cpp", gguf=GGUF)

    with pytest.raises(BaselineError, match="也必须是 transformers"):
        lora_gain(**gain_inputs, now=NOW)


def test_lora_gain_base_must_belong_to_selection(
    gain_inputs: dict[str, Any], items: tuple[EvaSftSample, ...]
) -> None:
    gain_inputs["base_config"] = make_config(
        items, tuning_set={"path": "other.json", "sha256": "f" * 64}
    )

    with pytest.raises(BaselineError, match="不属于这次提示词选择"):
        lora_gain(**gain_inputs, now=NOW)


def test_lora_gain_needs_same_eval_images(gain_inputs: dict[str, Any]) -> None:
    base = gain_inputs["base_eval"]
    gain_inputs["base_eval"] = base.model_copy(update={"ids_fingerprint": "8" * 64})

    with pytest.raises(BaselineError, match="同一份 eval 集"):
        lora_gain(**gain_inputs, now=NOW)


def test_lora_gain_needs_defined_metrics(gain_inputs: dict[str, Any]) -> None:
    base = gain_inputs["base_eval"]
    gain_inputs["base_eval"] = base.model_copy(
        update={
            "metrics": {"aesthetic": base.metrics["aesthetic"]},
            "metric_errors": {"technical": "无定义"},
        }
    )

    with pytest.raises(BaselineError, match="算不出提升"):
        lora_gain(**gain_inputs, now=NOW)


# --- SFT data uses the selected prompt ---


@pytest.mark.parametrize(
    ("path", "ok"),
    [
        ("prompts/baseline-v2/rubric-score.prompt.md", True),
        ("prompts/baseline-v2/zero-shot-score.prompt.md", True),
        ("prompts/baseline-v2/few-shot-score.prompt.md", False),
    ],
)
def test_sft_preparation_accepts_single_image_prompts(
    tmp_path: Path, path: str, ok: bool
) -> None:
    write_sources(tmp_path)

    def prepare() -> object:
        return prepare_eva_sft(
            tmp_path,
            Path(path),
            version="v2",
            seed=42,
            ratios=SplitRatios(train=0.5, val=0.25, eval=0.25),
            created_at=NOW,
            dry_run=True,
        )

    if ok:
        plan = prepare()
        assert {s.prompt_id for s in plan.samples} == {  # type: ignore[attr-defined]
            find_prompt(
                Path(path).name.removesuffix(".prompt.md") + "@v2", PROMPTS
            ).prompt_id
        }
    else:
        with pytest.raises(EvaPreparationError, match="不带少样本示例"):
            prepare()


# --- message conversion ---


def user(*parts: object) -> Message:
    return Message(role="user", content=parts)


def test_hf_chat_messages_use_placeholders() -> None:
    messages = (
        Message(role="system", content=(TextPart(type="text", text="细则"),)),
        user(
            {"type": "image", "image": "images/eva/1.jpg"},
            {"type": "text", "text": "评分"},
        ),
    )

    chat, images = hf_chat_messages(messages)

    assert chat == [
        {"role": "system", "content": [{"type": "text", "text": "细则"}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "评分"}],
        },
    ]
    assert images == ["images/eva/1.jpg"]


def test_conversions_reject_video(tmp_path: Path) -> None:
    messages = (user(VideoPart(type="video", video="videos/video_000001.mp4")),)

    with pytest.raises(BaselineError, match="不支持视频"):
        hf_chat_messages(messages)
    with pytest.raises(BaselineError, match="不支持视频"):
        openai_chat_messages(messages, tmp_path, PREPROCESS)


def test_openai_messages_resize_image(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    Image.new("RGB", (640, 480), "red").save(tmp_path / "images" / "a.png")
    messages = (
        Message(role="system", content=(TextPart(type="text", text="细则"),)),
        user(
            {"type": "image", "image": "images/a.png"},
            {"type": "text", "text": "评分"},
        ),
    )

    payload = openai_chat_messages(messages, tmp_path, PREPROCESS)

    assert payload[0] == {"role": "system", "content": "细则"}
    url = payload[1]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        assert image.size == fit_size(640, 480, PREPROCESS)
    assert payload[1]["content"][1] == {"type": "text", "text": "评分"}


@pytest.mark.parametrize(
    ("size", "expected"),
    [((640, 480), (288, 224)), ((32, 32), (64, 64)), ((128, 128), (128, 128))],
)
def test_fit_size(size: tuple[int, int], expected: tuple[int, int]) -> None:
    width, height = fit_size(*size, PREPROCESS)

    assert (width, height) == expected
    assert width % 32 == 0 and height % 32 == 0
    assert width * height <= PREPROCESS.max_pixels


# --- comparison table ---


def report(
    run_id: str, fingerprint: str = "7" * 64, **overrides: object
) -> EvaEvalReport:
    data: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "split": "all",
        "ids_fingerprint": fingerprint,
        "created_at": NOW,
        "n_items": 10,
        "n_parsed": 9,
        "parse_success_rate": 0.9,
        "metrics": {
            "aesthetic": {"srcc": 0.51, "plcc": 0.55, "mae_1_5": 0.4},
            "technical": {"srcc": 0.32, "plcc": 0.3, "mae_1_5": 0.5},
        },
        "metric_errors": {},
        "latency_p50_ms": 800.0,
        "latency_p95_ms": 900.0,
    }
    data.update(overrides)
    return EvaEvalReport.model_validate(data)


def test_comparison_table_rows() -> None:
    undefined = report(
        "b0-val",
        metrics={"aesthetic": {"srcc": 0.2, "plcc": 0.2, "mae_1_5": 0.6}},
        metric_errors={"technical": "有一组分数全部相同，相关系数无定义"},
    )

    table = comparison_table([report("b1-val"), undefined]).splitlines()

    assert table[1] == "b1-val | 0.900 | 0.510/0.550 | 0.320/0.300 | 800"
    assert "无定义（有一组分数全部相同" in table[2]


def test_comparison_table_requires_same_images() -> None:
    with pytest.raises(BaselineError, match="不是同一批图片"):
        comparison_table([report("a"), report("b", fingerprint="8" * 64)])
    with pytest.raises(BaselineError, match="没有报告"):
        comparison_table([])
