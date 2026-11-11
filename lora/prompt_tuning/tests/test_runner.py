from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from ptune.prompts import Message, PromptTemplate, TextPart, find_prompt
from ptune.runner import (
    Job,
    TuningError,
    build_report,
    comparison_table,
    hf_chat_messages,
    ids_fingerprint,
    load_job,
    parse_reply,
    prepare_run_dir,
    read_predictions,
    run_baseline,
    select_best_prompt,
    verify_images,
    write_reports,
    write_selection,
)
from ptune.schema import (
    EvaBaselineRunConfig,
    EvaEvalReport,
    PromptSelection,
    TuningItem,
)
from pydantic import ValidationError

from tests.conftest import NOW, ROWS, JobFiles, write_job

PROMPTS = {
    "b0_zero_shot": "zero-shot-score@v2",
    "b1_rubric": "rubric-score@v2",
    "b2_few_shot": "few-shot-score@v2",
}


@pytest.fixture
def job(job_files: JobFiles) -> Job:
    return load_job(job_files.root, job_files.job)


def make_config(
    job: Job,
    method: str = "b1_rubric",
    items: tuple[TuningItem, ...] | None = None,
    **overrides: object,
) -> EvaBaselineRunConfig:
    scored = job.items if items is None else items
    data: dict[str, Any] = {
        "schema_version": "1.0",
        "scoring": {
            "schema_version": "1.0",
            "run_id": f"{method.split('_')[0]}-tune",
            "split": "all",
            "ids_fingerprint": ids_fingerprint(i.image_id for i in scored),
            "model": job.config.model,
            "adapter": None,
            "prompt_id": PROMPTS.get(method, "few-shot-score@v2"),
            "chat_template_sha256": job.config.chat_template_sha256,
            "preprocess": job.config.preprocess,
            "max_new_tokens": job.config.max_new_tokens,
            "device": "cuda",
        },
        "tuning_set": {"path": "data/tuning-set.json", "sha256": "f" * 64},
        "method": method,
        "runtime": "transformers",
        "gguf": None,
    }
    data.update(overrides)
    return EvaBaselineRunConfig.model_validate(data)


def prompt(job: Job, prompt_id: str) -> PromptTemplate:
    return find_prompt(prompt_id, job.root / "prompts")


class EchoModel:
    """Answers each image with its SFT target, as a perfect model would."""

    def __init__(self, job: Job) -> None:
        self.by_path = {i.image_path: i for i in (*job.items, *job.examples)}
        self.seen: list[tuple[Message, ...]] = []

    def __call__(self, messages: tuple[Message, ...]) -> str:
        self.seen.append(messages)
        return self.by_path[messages[-1].images[0].image].reply_text


# --- job loading ---


def test_load_job_splits_items_and_examples(job: Job) -> None:
    assert [i.image_id for i in job.examples] == ["photo_000101", "photo_000102"]
    assert len(job.items) == 7 and all(i.role == "item" for i in job.items)
    assert [i.image_id for i in job.eval_items] == [
        "photo_000005",
        "photo_000006",
        "photo_000007",
    ]
    verify_images(job, [*job.items, *job.examples])


def test_load_job_detects_changed_files(job_files: JobFiles) -> None:
    items = job_files.root / "data" / "items.jsonl"
    items.write_text(items.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(TuningError, match="items.jsonl 的哈希"):
        load_job(job_files.root, job_files.job)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"photo_000101": ("train", "item", 4.5, 2.6)}, "example_ids"),
        ({"photo_000101": ("val", "example", 4.5, 2.6)}, "只能取自 train"),
        ({"photo_000008": ("eval", "item", 5.0, 2.0)}, "数量或指纹"),
    ],
)
def test_load_job_checks_tuning_set(
    tmp_path: Path,
    job_files: JobFiles,
    change: dict[str, tuple[str, str, float, float]],
    message: str,
) -> None:
    other = write_job(tmp_path / "other", ROWS | change)
    tuning = (job_files.root / "data" / "tuning-set.json").read_bytes()
    (other.root / "data" / "tuning-set.json").write_bytes(tuning)
    config = json.loads(other.job.read_text(encoding="utf-8"))
    config["tuning_set"]["sha256"] = hashlib.sha256(tuning).hexdigest()
    other.job.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TuningError, match=message):
        load_job(other.root, other.job)


def test_load_job_rejects_duplicate_rows(job_files: JobFiles) -> None:
    items = job_files.root / "data" / "items.jsonl"
    lines = items.read_text(encoding="utf-8").splitlines()
    payload = "\n".join([*lines, lines[0]]) + "\n"
    items.write_text(payload, encoding="utf-8", newline="\n")
    config = json.loads(job_files.job.read_text(encoding="utf-8"))
    config["items"]["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    job_files.job.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TuningError, match="重复 image_id"):
        load_job(job_files.root, job_files.job)


def test_verify_images_detects_missing_and_changed(job: Job) -> None:
    first = job.items[0]
    (job.root / "data" / first.image_path).write_bytes(b"changed")
    with pytest.raises(TuningError, match="哈希与记录不符"):
        verify_images(job, [first])
    (job.root / "data" / first.image_path).unlink()
    with pytest.raises(TuningError, match="图片不存在"):
        verify_images(job, [first])


@pytest.mark.parametrize(
    "reply", ['{"aesthetic": 3.33, "technical": 3.0}', "not json", '{"aesthetic": 3}']
)
def test_tuning_item_reply_must_be_one_decimal_target(reply: str) -> None:
    with pytest.raises(ValidationError, match="reply_text|一位小数"):
        TuningItem(
            image_id="photo_000001",
            split="train",
            role="item",
            image_path="images/eva/1.jpg",
            image_sha256="a" * 64,
            aesthetic_mean=5.0,
            technical_mean=2.0,
            reply_text=reply,
        )


# --- runs ---


def test_parse_reply_success_and_failure() -> None:
    ok = parse_reply("b1-tune", "photo_000001", '{"aesthetic":3.2,"technical":2.0}', 5)
    fenced = parse_reply("b1-tune", "photo_000001", '```json\n{"aesthetic":3}\n```', 5)

    assert ok.scores is not None and ok.scores.aesthetic == 3.2
    assert fenced.scores is None and fenced.parse_error


ACCEPTED_WRAPPERS = [
    '```json\n{\n  "aesthetic": 3.5,\n  "technical": 4.0\n}\n```\n',
    '```JSON\n{"aesthetic": 3.5, "technical": 4.0}\n```',
    '```\n{"aesthetic": 3.5, "technical": 4.0}\n```',
    '  \n```json\n{"aesthetic": 3.5, "technical": 4.0}```  ',
    '<think>\n\n</think>\n\n{"aesthetic": 3.5, "technical": 4.0}',
    '<think>\nlook\n</think>\n```json\n{"aesthetic": 3.5, "technical": 4.0}\n```',
    'Scores: ```json\n{"aesthetic": 3.5, "technical": 4.0}\n```',
    '```json\n{"aesthetic": 3.5, "technical": 4.0}\n```\nNice photo.',
    '```json\n{"aesthetic": 3.5, "technical": 4.0}\n',
    '```python\n{"aesthetic": 3.5, "technical": 4.0}\n```',
    '``json\n{\n  "aesthetic": 3.5,\n  "technical": 4.0\n}\n``\n',
    '~~~json\n{"aesthetic": 3.5, "technical": 4.0}\n~~~',
    '`{"aesthetic": 3.5, "technical": 4.0}`',
    '评分如下：{"aesthetic": 3.5, "technical": 4.0}。',
    '{"technical": 4.0, "aesthetic": 3.5}',
    '{"aesthetic": 3.5, "technical": 4.0} <think></think>',
    '<think>{"aesthetic": 1.0, "technical": 1.0}</think>{"aesthetic": 3.5, '
    '"technical": 4.0}',
]
REJECTED_WRAPPERS = [
    '```json\n{"aesthetic": 3.5, "technical": 4.0, "overall": 3}\n```',
    '```json\n{"aesthetic": 3.5}\n```\n```json\n{"technical": 4.0}\n```',
    '例 {"aesthetic": 2.0, "technical": 2.0}；'
    '本图 {"aesthetic": 3.5, "technical": 4.0}',
    '{"scores": {"aesthetic": 3.5, "technical": 4.0}}',
    '{"aesthetic": 3.5, "technical": ',
    "{'aesthetic': 3.5, 'technical': 4.0}",
    '{"aesthetic": 3.5, "technical": 4.0,}',
    '{"aesthetic": 7, "technical": 4.0}',
    "美学 3.5 分，技术 4.0 分",
    "<think>\nlook",
    "",
]


@pytest.mark.parametrize("reply", ACCEPTED_WRAPPERS)
def test_parse_reply_removes_format_wrappers(reply: str) -> None:
    prediction = parse_reply("b1-tune", "photo_000001", reply, 5)

    assert prediction.scores is not None
    assert (prediction.scores.aesthetic, prediction.scores.technical) == (3.5, 4.0)
    assert prediction.raw_output == reply


@pytest.mark.parametrize("reply", REJECTED_WRAPPERS)
def test_parse_reply_rejects_anything_else(reply: str) -> None:
    prediction = parse_reply("b1-tune", "photo_000001", reply, 5)

    assert prediction.scores is None and prediction.parse_error


def test_run_scores_resumes_and_writes_both_reports(job: Job) -> None:
    config = make_config(job)
    run_dir = job.root / "runs" / config.run_id
    model = EchoModel(job)
    used = prompt(job, "rubric-score@v2")

    first = run_baseline(config, used, job, job.items, run_dir=run_dir, complete=model)
    second = run_baseline(config, used, job, job.items, run_dir=run_dir, complete=model)
    report, sliced = write_reports(config, job, job.items, run_dir=run_dir, now=NOW)

    assert (first, second) == (7, 0)
    assert [m.role for m in model.seen[0]] == ["system", "user"]
    assert (report.split, report.n_items, report.parse_success_rate) == ("all", 7, 1.0)
    assert report.metrics["aesthetic"].srcc == pytest.approx(1.0)
    assert (sliced.split, sliced.n_items) == ("eval", 3)
    assert sliced.ids_fingerprint == job.tuning_set.eval_ids_fingerprint
    saved = EvaEvalReport.model_validate_json(
        (run_dir / "report-eval.json").read_text(encoding="utf-8")
    )
    assert saved == sliced


def test_smoke_run_writes_only_tuning_report(job: Job) -> None:
    items = job.items[:3]
    config = make_config(job, items=items)
    run_dir = job.root / "runs" / config.run_id
    run_baseline(
        config,
        prompt(job, "rubric-score@v2"),
        job,
        items,
        run_dir=run_dir,
        complete=EchoModel(job),
    )

    reports = write_reports(config, job, items, run_dir=run_dir, now=NOW)

    assert len(reports) == 1 and not (run_dir / "report-eval.json").exists()


def test_parse_failures_are_recorded(job: Job) -> None:
    config = make_config(job, "b0_zero_shot")
    run_dir = job.root / "runs" / config.run_id
    seen: list[int] = []

    run_baseline(
        config,
        prompt(job, "zero-shot-score@v2"),
        job,
        job.items,
        run_dir=run_dir,
        complete=lambda m: "这张照片美学 3 分",
        on_progress=lambda done, total, p: seen.append(done),
    )
    report = write_reports(config, job, job.items, run_dir=run_dir, now=NOW)[0]

    assert seen == list(range(1, 8))
    assert report.parse_success_rate == 0.0
    assert set(report.metric_errors) == {"aesthetic", "technical"}


def test_few_shot_run_labels_examples(job: Job) -> None:
    config = make_config(job, "b2_few_shot")
    model = EchoModel(job)

    run_baseline(
        config,
        prompt(job, "few-shot-score@v2"),
        job,
        job.items,
        run_dir=job.root / "runs" / config.run_id,
        complete=model,
    )

    messages = model.seen[0]
    assert [m.role for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1].images[0].image == job.examples[0].image_path
    assert messages[1].text.startswith("【示例 1/2】")
    assert messages[2].text == job.examples[0].reply_text
    assert messages[-1].text.startswith("【待评照片】以上 2 张示例到此结束")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("prompt", "与配置"),
        ("method", "不用于图片上的"),
        ("fingerprint", "ids_fingerprint"),
        ("example_scored", "不能出现在被评的图片里"),
    ],
)
def test_run_rejects_mismatches(job: Job, change: str, message: str) -> None:
    items = job.items
    config = make_config(job)
    used = prompt(job, "rubric-score@v2")
    if change == "prompt":
        used = prompt(job, "zero-shot-score@v2")
    elif change == "method":
        config = make_config(job, "b0_zero_shot")
        config = config.model_copy(
            update={
                "scoring": config.scoring.model_copy(
                    update={"prompt_id": "rubric-score@v2"}
                )
            }
        )
    elif change == "fingerprint":
        items = job.items[:1]
    else:
        items = (*job.items, job.examples[0])
        config = make_config(job, items=items)

    with pytest.raises(TuningError, match=message):
        run_baseline(
            config,
            used,
            job,
            items,
            run_dir=job.root / "runs" / "x",
            complete=EchoModel(job),
        )


def test_resume_rejects_changed_config_and_foreign_predictions(job: Job) -> None:
    config = make_config(job)
    run_dir = job.root / "runs" / config.run_id
    prepare_run_dir(run_dir, config)
    changed = make_config(job, "b0_zero_shot")
    changed = changed.model_copy(
        update={"scoring": changed.scoring.model_copy(update={"run_id": "b1-tune"})}
    )
    with pytest.raises(TuningError, match="不能续跑"):
        prepare_run_dir(run_dir, changed)

    line = {
        "run_id": "other-run",
        "image_id": "photo_000001",
        "raw_output": "x",
        "scores": None,
        "parse_error": "bad",
        "latency_ms": 1.0,
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(line) + "\n", "utf-8")
    with pytest.raises(TuningError, match="混有其他运行"):
        prepare_run_dir(run_dir, config)


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        ({"method": "lora_sft"}, "不是提示词基线"),
        ({"method": "b3_retrieval"}, "B3"),
        ({"runtime": "llama_cpp"}, "gguf"),
    ],
)
def test_baseline_config_rules(
    job: Job, config_overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_config(job, **config_overrides)


def test_report_requires_matching_predictions(job: Job) -> None:
    config = make_config(job)
    run_dir = job.root / "runs" / config.run_id
    run_baseline(
        config,
        prompt(job, "rubric-score@v2"),
        job,
        job.items,
        run_dir=run_dir,
        complete=EchoModel(job),
    )
    predictions = read_predictions(run_dir)

    with pytest.raises(TuningError, match="重复 image_id"):
        build_report(config.scoring, [*predictions, predictions[0]], job.items, now=NOW)
    with pytest.raises(TuningError, match="不一致"):
        build_report(config.scoring, predictions[1:], job.items, now=NOW)
    foreign = [p.model_copy(update={"run_id": "other-run"}) for p in predictions]
    with pytest.raises(TuningError, match="混有其他运行"):
        build_report(config.scoring, foreign, job.items, now=NOW)


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
    job: Job, scores: dict[str, tuple[float, float | None, float]]
) -> list[tuple[EvaBaselineRunConfig, EvaEvalReport]]:
    runs = []
    for method, (aesthetic, technical, rate) in scores.items():
        config = make_config(job, method)
        runs.append((config, tune_report(config, aesthetic, technical, rate)))
    return runs


def test_select_keeps_single_image_prompt_for_lora(job: Job) -> None:
    runs = tune_runs(
        job,
        {
            "b0_zero_shot": (0.40, 0.30, 1.0),
            "b1_rubric": (0.55, 0.40, 1.0),
            "b2_few_shot": (0.60, 0.45, 1.0),
        },
    )

    selection = select_best_prompt(runs, min_parse_rate=0.95, now=NOW)

    assert selection.best_prompt_id == "few-shot-score@v2"
    assert selection.lora_prompt_id == "rubric-score@v2"


def test_select_skips_ineligible(job: Job) -> None:
    runs = tune_runs(
        job,
        {
            "b0_zero_shot": (0.40, 0.30, 1.0),
            "b1_rubric": (0.70, 0.60, 0.80),
            "b2_few_shot": (0.65, None, 1.0),
        },
    )

    selection = select_best_prompt(runs, min_parse_rate=0.95, now=NOW)

    assert selection.best_prompt_id == selection.lora_prompt_id == "zero-shot-score@v2"


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({"b0_zero_shot": (0.4, 0.3, 1.0), "b1_rubric": (0.3, 0.4, 1.0)}, "并列"),
        ({"b0_zero_shot": (0.4, 0.3, 0.5)}, "没有合格的提示词"),
        ({"b2_few_shot": (0.4, 0.3, 1.0)}, "没有合格的单图提示词"),
    ],
)
def test_select_refuses_to_guess(
    job: Job, scores: dict[str, tuple[float, float | None, float]], message: str
) -> None:
    with pytest.raises(TuningError, match=message):
        select_best_prompt(tune_runs(job, scores), min_parse_rate=0.95, now=NOW)


def test_select_requires_same_setup(job: Job) -> None:
    runs = tune_runs(
        job, {"b0_zero_shot": (0.4, 0.3, 1.0), "b1_rubric": (0.5, 0.3, 1.0)}
    )
    other = make_config(
        job, "b2_few_shot", tuning_set={"path": "other.json", "sha256": "0" * 64}
    )
    duplicate = make_config(job, "b0_zero_shot")

    with pytest.raises(TuningError, match="没有候选"):
        select_best_prompt([], min_parse_rate=0.95, now=NOW)
    with pytest.raises(TuningError, match="运行时不同"):
        select_best_prompt(
            [*runs, (other, tune_report(other, 0.5, 0.5))], min_parse_rate=0.95, now=NOW
        )
    with pytest.raises(TuningError, match="不是这次运行"):
        select_best_prompt([(runs[0][0], runs[1][1])], min_parse_rate=0.95, now=NOW)
    with pytest.raises(TuningError, match="出现了多次"):
        select_best_prompt(
            [runs[0], (duplicate, tune_report(duplicate, 0.4, 0.3))],
            min_parse_rate=0.95,
            now=NOW,
        )


def test_write_selection_never_overwrites(job: Job, tmp_path: Path) -> None:
    runs = tune_runs(job, {"b1_rubric": (0.5, 0.4, 1.0)})
    selection = select_best_prompt(runs, min_parse_rate=0.95, now=NOW)
    path = tmp_path / "runs" / "selection.json"

    write_selection(path, selection)

    saved = PromptSelection.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved == selection
    with pytest.raises(TuningError, match="不覆盖"):
        write_selection(path, selection)


def test_comparison_table(job: Job) -> None:
    config = make_config(job)
    undefined = tune_report(config, 0.2, None)

    table = comparison_table([tune_report(config, 0.51, 0.32), undefined])

    assert table.splitlines()[1] == "b1-tune | 1.000 | 0.510/0.510 | 0.320/0.320 | 500"
    assert "无定义（有一组分数全部相同" in table.splitlines()[2]
    with pytest.raises(TuningError, match="不是同一批图片"):
        comparison_table(
            [undefined, undefined.model_copy(update={"ids_fingerprint": "0" * 64})]
        )
    with pytest.raises(TuningError, match="没有报告"):
        comparison_table([])


def test_hf_chat_messages_use_placeholders() -> None:
    messages = (
        Message(role="system", content=(TextPart(type="text", text="细则"),)),
        Message(
            role="user",
            content=(
                {"type": "image", "image": "images/eva/1.jpg"},
                {"type": "text", "text": "评分"},
            ),
        ),
    )

    chat, images = hf_chat_messages(messages)

    assert chat[1]["content"] == [{"type": "image"}, {"type": "text", "text": "评分"}]
    assert images == ["images/eva/1.jpg"]
