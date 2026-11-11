"""Run prompt baselines on the EVA prompt-tuning set and pick the best prompt.

Every baseline (B0, B1, B2) scores the same images: all of EVA except the
few-shot examples. Replies are parsed strictly as ``EvaDualScoreOutput``
once format wrappers (a think block, a code fence) are removed (a failure
is recorded, never repaired) and reports have the main repository's
format, so its LoRA comparison reads them directly. Correlations are against
the unrounded EVA vote means.

Each run lives in ``runs/<run_id>/``: ``config.json`` is written once and must
match on resume, ``predictions.jsonl`` gets one line per image, ``report.json``
covers the tuning set and ``report-eval.json`` only its SFT eval-split images
(the prompt without LoRA, on the images the LoRA model is evaluated on).

The model is reached through a ``complete`` callable taking our messages and
returning the reply text, so this module needs neither torch nor a GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ptune.metrics import MetricsError, percentile, plcc, srcc
from ptune.prompts import (
    FewShotExample,
    ImagePart,
    Message,
    PromptTemplate,
    TextPart,
    build_messages,
)
from ptune.schema import (
    EVA_DIMENSIONS,
    FEW_SHOT_METHODS,
    DimensionMetrics,
    EvaBaselineRunConfig,
    EvaDualScoreOutput,
    EvaEvalReport,
    EvaEvalRunConfig,
    EvaPrediction,
    JobConfig,
    PromptCandidate,
    PromptSelection,
    PromptTuningSet,
    TuningItem,
)

Complete = Callable[[tuple[Message, ...]], str]


class TuningError(ValueError):
    """Raised when a job, run or report is inconsistent with its inputs."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_fingerprint(image_ids: Iterable[str]) -> str:
    """Order-independent SHA-256 of image ids (same as the main repository)."""

    joined = "\n".join(sorted(image_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# --- job inputs ---


@dataclass(frozen=True)
class Job:
    """A verified job: config, tuning set, scored items and examples."""

    root: Path
    config: JobConfig
    tuning_set: PromptTuningSet
    items: tuple[TuningItem, ...]
    examples: tuple[TuningItem, ...]

    @property
    def eval_items(self) -> tuple[TuningItem, ...]:
        """Scored items from the SFT eval split."""

        return tuple(item for item in self.items if item.split == "eval")


def load_job(root: Path, config_path: Path) -> Job:
    """Read the job files and prove they match each other and the hashes."""

    config = JobConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    for ref in (config.tuning_set, config.items):
        if sha256_file(root / ref.path) != ref.sha256:
            raise TuningError(f"{ref.path} 的哈希与 job 配置不符")
    tuning_set = PromptTuningSet.model_validate_json(
        (root / config.tuning_set.path).read_text(encoding="utf-8")
    )
    rows = [
        TuningItem.model_validate_json(line)
        for line in (root / config.items.path).read_text(encoding="utf-8").splitlines()
    ]
    ids = [row.image_id for row in rows]
    if len(set(ids)) != len(ids):
        raise TuningError("items 里有重复 image_id")
    items = tuple(
        sorted((r for r in rows if r.role == "item"), key=lambda r: r.image_id)
    )
    by_id = {r.image_id: r for r in rows if r.role == "example"}
    if set(by_id) != set(tuning_set.example_ids):
        raise TuningError("items 里的示例与提示词调优集的 example_ids 不符")
    if any(by_id[i].split != "train" for i in tuning_set.example_ids):
        raise TuningError("少样本示例只能取自 train")
    eval_ids = [item.image_id for item in items if item.split == "eval"]
    if (
        len(items) != tuning_set.item_count
        or ids_fingerprint(i.image_id for i in items) != tuning_set.ids_fingerprint
        or ids_fingerprint(eval_ids) != tuning_set.eval_ids_fingerprint
    ):
        raise TuningError("items 与提示词调优集的数量或指纹不符")
    examples = tuple(by_id[i] for i in tuning_set.example_ids)
    return Job(root, config, tuning_set, items, examples)


def verify_images(job: Job, items: Iterable[TuningItem]) -> None:
    """Check every image exists with the recorded SHA-256."""

    for item in items:
        path = job.root / "data" / item.image_path
        if not path.is_file():
            raise TuningError(f"图片不存在: {path}")
        if sha256_file(path) != item.image_sha256:
            raise TuningError(f"图片哈希与记录不符: {path}")


# --- runs ---


# Format noise around the JSON object, removed before strict parsing
# (ADR-0013): an optional leading think block, then whatever surrounds the one
# complete top-level JSON object in the reply (code fences of any style,
# inline code, prose). Zero or several objects leave the text as it is, so it
# fails; the object itself is never edited.
_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)
_JSON_DECODER = json.JSONDecoder()


def _json_objects(text: str) -> list[str]:
    """Source spans of every complete top-level JSON object in ``text``."""

    spans: list[str] = []
    start = text.find("{")
    while start != -1:
        try:
            value, end = _JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            spans.append(text[start:end])
        start = text.find("{", end)
    return spans


def unwrap_reply(raw_output: str) -> str:
    """The single JSON object in the reply, verbatim; else the reply itself."""

    text = _THINK_BLOCK.sub("", raw_output, count=1)
    objects = _json_objects(text)
    return objects[0] if len(objects) == 1 else text


def parse_reply(
    run_id: str, image_id: str, raw_output: str, latency_ms: float
) -> EvaPrediction:
    """Parse model text strictly after ``unwrap_reply``; keep any error."""

    try:
        scores = EvaDualScoreOutput.model_validate_json(unwrap_reply(raw_output))
    except ValidationError as exc:
        return EvaPrediction(
            run_id=run_id,
            image_id=image_id,
            raw_output=raw_output,
            scores=None,
            parse_error=str(exc.errors()[0]["msg"]),
            latency_ms=latency_ms,
        )
    return EvaPrediction(
        run_id=run_id,
        image_id=image_id,
        raw_output=raw_output,
        scores=scores,
        parse_error=None,
        latency_ms=latency_ms,
    )


def read_predictions(run_dir: Path) -> list[EvaPrediction]:
    """Load every prediction written so far."""

    path = run_dir / "predictions.jsonl"
    return [
        EvaPrediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def prepare_run_dir(run_dir: Path, config: EvaBaselineRunConfig) -> set[str]:
    """Create or resume a run directory; return image ids already scored."""

    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = EvaBaselineRunConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        if existing != config:
            raise TuningError(f"{config_path} 与本次配置不同，不能续跑；请换 run_id")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path.write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    if not (run_dir / "predictions.jsonl").exists():
        return set()
    done: set[str] = set()
    for prediction in read_predictions(run_dir):
        if prediction.run_id != config.run_id:
            raise TuningError(f"{run_dir} 混有其他运行的预测")
        done.add(prediction.image_id)
    return done


def few_shot_examples(examples: Sequence[TuningItem]) -> tuple[FewShotExample, ...]:
    """Examples whose replies are the SFT targets, verbatim."""

    return tuple(
        FewShotExample(
            image_id=item.image_id,
            image_path=item.image_path,
            reply_text=item.reply_text,
        )
        for item in examples
    )


def run_baseline(
    config: EvaBaselineRunConfig,
    prompt: PromptTemplate,
    job: Job,
    items: Sequence[TuningItem],
    *,
    run_dir: Path,
    complete: Complete,
    clock: Callable[[], float] = time.perf_counter,
    on_progress: Callable[[int, int, EvaPrediction], None] | None = None,
) -> int:
    """Score every item not yet in ``predictions.jsonl``; return how many ran.

    ``items`` is the job's items, or a prefix of them for a smoke test.
    """

    if prompt.prompt_id != config.scoring.prompt_id:
        raise TuningError(
            f"提示词 {prompt.prompt_id} 与配置 {config.scoring.prompt_id} 不符"
        )
    if config.method not in prompt.methods or prompt.media != "image":
        raise TuningError(f"{prompt.prompt_id} 不用于图片上的 {config.method.value}")
    if ids_fingerprint(i.image_id for i in items) != config.scoring.ids_fingerprint:
        raise TuningError("图片与配置的 ids_fingerprint 不符")
    if {i.image_id for i in items} & set(job.tuning_set.example_ids):
        raise TuningError("少样本示例图不能出现在被评的图片里")
    shots = few_shot_examples(job.examples) if config.is_few_shot else ()
    done = prepare_run_dir(run_dir, config)
    todo = [item for item in items if item.image_id not in done]
    path = run_dir / "predictions.jsonl"
    for number, item in enumerate(todo, 1):
        messages = build_messages(prompt, item.image_id, item.image_path, shots)
        started = clock()
        text = complete(messages)
        latency_ms = (clock() - started) * 1000
        prediction = parse_reply(config.run_id, item.image_id, text, latency_ms)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(prediction.model_dump(mode="json"), ensure_ascii=False))
            f.write("\n")
        if on_progress is not None:
            on_progress(number, len(todo), prediction)
    return len(todo)


# --- reports ---


def build_report(
    scoring: EvaEvalRunConfig,
    predictions: Sequence[EvaPrediction],
    items: Sequence[TuningItem],
    *,
    now: datetime,
) -> EvaEvalReport:
    """Aggregate predictions for every item exactly as the main repository does."""

    by_id = {item.image_id: item for item in items}
    predicted = {p.image_id: p for p in predictions}
    if len(predicted) != len(predictions):
        raise TuningError("预测里有重复 image_id")
    if predicted.keys() != by_id.keys():
        missing = sorted(by_id.keys() - predicted.keys())[:3]
        extra = sorted(predicted.keys() - by_id.keys())[:3]
        raise TuningError(f"预测与图片不一致: missing={missing}, extra={extra}")
    if any(p.run_id != scoring.run_id for p in predictions):
        raise TuningError("预测里混有其他运行")
    parsed = [p for p in predictions if p.scores is not None]
    metrics: dict[str, DimensionMetrics] = {}
    metric_errors: dict[str, str] = {}
    for dim in EVA_DIMENSIONS:
        pred = [getattr(p.scores, dim) for p in parsed]
        means = [by_id[p.image_id].mean(dim) for p in parsed]
        targets = [getattr(by_id[p.image_id].target, dim) for p in parsed]
        try:
            metrics[dim] = DimensionMetrics(
                srcc=srcc(pred, means),
                plcc=plcc(pred, means),
                mae_1_5=sum(abs(a - b) for a, b in zip(pred, targets, strict=True))
                / len(parsed),
            )
        except MetricsError as exc:
            metric_errors[dim] = str(exc)
    latencies = [p.latency_ms for p in predictions]
    return EvaEvalReport(
        schema_version="1.0",
        run_id=scoring.run_id,
        split=scoring.split,
        ids_fingerprint=scoring.ids_fingerprint,
        created_at=now,
        n_items=len(predictions),
        n_parsed=len(parsed),
        parse_success_rate=len(parsed) / len(predictions),
        metrics=metrics,
        metric_errors=metric_errors,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )


def _write(path: Path, record: EvaEvalReport | PromptSelection) -> None:
    path.write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_reports(
    config: EvaBaselineRunConfig,
    job: Job,
    items: Sequence[TuningItem],
    *,
    run_dir: Path,
    now: datetime,
) -> list[EvaEvalReport]:
    """Write ``report.json``; for a full run also ``report-eval.json``."""

    predictions = read_predictions(run_dir)
    report = build_report(config.scoring, predictions, items, now=now)
    _write(run_dir / "report.json", report)
    if len(items) != len(job.items):
        return [report]
    eval_items = job.eval_items
    ids = {item.image_id for item in eval_items}
    scoring = config.scoring.model_copy(
        update={"split": "eval", "ids_fingerprint": job.tuning_set.eval_ids_fingerprint}
    )
    sliced = build_report(
        scoring, [p for p in predictions if p.image_id in ids], eval_items, now=now
    )
    _write(run_dir / "report-eval.json", sliced)
    return [report, sliced]


# --- choosing the best prompt ---


def _candidate(
    config: EvaBaselineRunConfig, report: EvaEvalReport, min_parse_rate: float
) -> PromptCandidate:
    ineligible: dict[str, str] = dict(report.metric_errors)
    if report.parse_success_rate < min_parse_rate:
        ineligible["parse_success_rate"] = (
            f"解析成功率 {report.parse_success_rate:.3f} 低于 {min_parse_rate}"
        )
    scores = {dim: metric.srcc for dim, metric in report.metrics.items()}
    return PromptCandidate(
        run_id=config.run_id,
        prompt_id=config.scoring.prompt_id,
        method=config.method,
        parse_success_rate=report.parse_success_rate,
        srcc=scores,
        ineligible=ineligible,
        score=None if ineligible else sum(scores.values()) / len(EVA_DIMENSIONS),
    )


def _best(candidates: Sequence[PromptCandidate], label: str) -> PromptCandidate:
    scored = [c for c in candidates if c.score is not None]
    if not scored:
        raise TuningError(f"没有合格的{label}")
    top = max(c.score for c in scored)  # type: ignore[type-var]
    winners = [c for c in scored if c.score == top]
    if len(winners) > 1:
        raise TuningError(f"{label}并列: {[c.prompt_id for c in winners]}，请人工决定")
    return winners[0]


def select_best_prompt(
    runs: Sequence[tuple[EvaBaselineRunConfig, EvaEvalReport]],
    *,
    min_parse_rate: float,
    now: datetime,
) -> PromptSelection:
    """Rank full tuning runs; they must differ only in prompt and method.

    ``lora_prompt_id`` is the best prompt without few-shot examples.
    """

    if not runs:
        raise TuningError("没有候选运行")
    first = runs[0][0]

    def setup(config: EvaBaselineRunConfig) -> tuple[Any, ...]:
        return (
            config.tuning_set,
            config.scoring.ids_fingerprint,
            config.scoring.model,
            config.scoring.preprocess,
            config.scoring.max_new_tokens,
            config.runtime,
            config.gguf,
        )

    for config, report in runs:
        if setup(config) != setup(first):
            raise TuningError(f"{config.run_id} 与其他运行的图片、模型或运行时不同")
        if (report.run_id, report.split, report.ids_fingerprint) != (
            config.run_id,
            "all",
            config.scoring.ids_fingerprint,
        ):
            raise TuningError(f"{config.run_id} 的报告不是这次运行在调优集上的报告")
    prompts = [config.scoring.prompt_id for config, _ in runs]
    if len(set(prompts)) != len(prompts):
        raise TuningError("同一个提示词出现了多次")
    candidates = tuple(_candidate(c, r, min_parse_rate) for c, r in runs)
    best = _best(candidates, "提示词")
    single = _best(
        [c for c in candidates if c.method not in FEW_SHOT_METHODS], "单图提示词"
    )
    return PromptSelection(
        schema_version="1.0",
        created_at=now,
        tuning_set=first.tuning_set,
        ids_fingerprint=first.scoring.ids_fingerprint,
        model=first.scoring.model,
        runtime=first.runtime,
        min_parse_rate=min_parse_rate,
        candidates=candidates,
        best_prompt_id=best.prompt_id,
        lora_prompt_id=single.prompt_id,
    )


def write_selection(path: Path, selection: PromptSelection) -> None:
    """Write the selection; an existing file is never overwritten."""

    if path.exists():
        raise TuningError(f"{path} 已存在，不覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, selection)


def comparison_table(reports: Sequence[EvaEvalReport]) -> str:
    """Plain-text table of runs; all runs must cover the same images."""

    if not reports:
        raise TuningError("没有报告")
    if len({(r.split, r.ids_fingerprint) for r in reports}) != 1:
        raise TuningError("这些运行评的不是同一批图片，不能放在一张表里比较")
    lines = ["run | 解析率 | 美学 SRCC/PLCC | 技术 SRCC/PLCC | p50 ms"]
    for report in reports:
        cells = []
        for dim in EVA_DIMENSIONS:
            metric = report.metrics.get(dim)
            cells.append(
                f"{metric.srcc:.3f}/{metric.plcc:.3f}"
                if metric is not None
                else f"无定义（{report.metric_errors[dim]}）"
            )
        lines.append(
            f"{report.run_id} | {report.parse_success_rate:.3f} | "
            f"{cells[0]} | {cells[1]} | {report.latency_p50_ms:.0f}"
        )
    return "\n".join(lines)


def hf_chat_messages(
    messages: Iterable[Message],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Hugging Face chat-template messages plus image paths in order.

    Image items carry only ``type``, exactly as in training: pixels go to the
    processor separately and the template renders the vision placeholder.
    """

    converted: list[dict[str, Any]] = []
    images: list[str] = []
    for message in messages:
        content: list[dict[str, str]] = []
        for part in message.content:
            if isinstance(part, ImagePart):
                content.append({"type": "image"})
                images.append(part.image)
            elif isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
        converted.append({"role": message.role, "content": content})
    return converted, images
