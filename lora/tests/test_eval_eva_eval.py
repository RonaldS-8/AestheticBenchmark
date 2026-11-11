from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lora_sft.data.eva_sft import EvaSftSample
from lora_sft.data.schema import eval_fingerprint
from lora_sft.eval.eva_eval import (
    INELIGIBLE_SELECTION_SCORE,
    EvaEvalError,
    EvaEvalReport,
    EvaEvalRunConfig,
    EvaPrediction,
    SelectionScore,
    append_prediction,
    build_eva_report,
    parse_eva_reply,
    prepare_eva_run_dir,
    read_predictions,
    selection_score,
)
from lora_sft.train.config import BaseModelRef, ImagePreprocess
from lora_sft.train.eva_data import load_eva_samples
from tests.test_data_eva_sft import NOW
from tests.test_train_eva_data import SAMPLES, repo  # noqa: F401

RUN = "eva-b0-val"


@pytest.fixture
def samples(repo: Path) -> tuple[EvaSftSample, ...]:  # noqa: F811
    return load_eva_samples(repo / SAMPLES / "train.jsonl", "train")


def run_config(
    samples: tuple[EvaSftSample, ...], **overrides: object
) -> EvaEvalRunConfig:
    data = {
        "schema_version": "1.0",
        "run_id": RUN,
        "split": "val",
        "ids_fingerprint": eval_fingerprint(s.image_id for s in samples),
        "model": BaseModelRef(model_id="Qwen/Qwen3.5-4B", revision="a" * 40),
        "adapter": None,
        "prompt_id": "eva-dual-score@v1",
        "chat_template_sha256": "b" * 64,
        "preprocess": ImagePreprocess(min_pixels=65536, max_pixels=1048576),
        "max_new_tokens": 32,
        "device": "cuda",
        **overrides,
    }
    return EvaEvalRunConfig.model_validate(data)


def test_parse_reply_success_and_failure() -> None:
    ok = parse_eva_reply(RUN, "photo_000001", '{"aesthetic":3.2,"technical":2.0}', 5)
    bad = parse_eva_reply(RUN, "photo_000001", '```json\n{"aesthetic":3}\n```', 5)
    out_of_range = parse_eva_reply(
        RUN, "photo_000001", '{"aesthetic":7,"technical":2}', 5
    )

    assert ok.scores is not None and ok.scores.aesthetic == 3.2
    assert bad.scores is None and bad.parse_error
    assert out_of_range.scores is None


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
    prediction = parse_eva_reply(RUN, "photo_000001", reply, 5)

    assert prediction.scores is not None
    assert (prediction.scores.aesthetic, prediction.scores.technical) == (3.5, 4.0)
    assert prediction.raw_output == reply


@pytest.mark.parametrize("reply", REJECTED_WRAPPERS)
def test_parse_reply_rejects_anything_else(reply: str) -> None:
    prediction = parse_eva_reply(RUN, "photo_000001", reply, 5)

    assert prediction.scores is None and prediction.parse_error


def test_prediction_requires_exactly_one_outcome() -> None:
    with pytest.raises(ValidationError, match="恰好填一个"):
        EvaPrediction(
            run_id=RUN,
            image_id="photo_000001",
            raw_output="x",
            scores=None,
            parse_error=None,
            latency_ms=1,
        )


def test_run_dir_create_resume_and_reject(
    tmp_path: Path, samples: tuple[EvaSftSample, ...]
) -> None:
    config = run_config(samples)
    run_dir = tmp_path / RUN

    assert prepare_eva_run_dir(run_dir, config) == set()
    append_prediction(run_dir, parse_eva_reply(RUN, samples[0].image_id, "{}", 3))
    assert prepare_eva_run_dir(run_dir, config) == {samples[0].image_id}

    with pytest.raises(EvaEvalError, match="不能续跑"):
        prepare_eva_run_dir(run_dir, run_config(samples, max_new_tokens=64))

    other = parse_eva_reply("other-run", samples[1].image_id, "{}", 3)
    append_prediction(run_dir, other)
    with pytest.raises(EvaEvalError, match="混有其他运行"):
        prepare_eva_run_dir(run_dir, config)


def predictions_for(
    samples: tuple[EvaSftSample, ...], replies: list[str]
) -> list[EvaPrediction]:
    return [
        parse_eva_reply(RUN, sample.image_id, reply, 10.0 + i)
        for i, (sample, reply) in enumerate(zip(samples, replies, strict=True))
    ]


def test_report_correlates_with_vote_means(
    samples: tuple[EvaSftSample, ...],
) -> None:
    replies = [s.messages[-1].text for s in samples[:-1]] + ["not json"]

    report = build_eva_report(
        run_config(samples),
        predictions_for(samples, replies),
        samples,
        now=NOW,
    )

    assert (report.n_items, report.n_parsed) == (len(samples), len(samples) - 1)
    assert report.metrics["aesthetic"].srcc > 0.9
    assert report.metrics["aesthetic"].mae_1_5 == 0
    assert set(report.metric_errors) == {"technical"}
    assert report.latency_p95_ms >= report.latency_p50_ms


def test_report_rejects_missing_duplicate_and_foreign_predictions(
    samples: tuple[EvaSftSample, ...],
) -> None:
    config = run_config(samples)
    replies = [s.messages[-1].text for s in samples]
    predictions = predictions_for(samples, replies)

    with pytest.raises(EvaEvalError, match="不一致"):
        build_eva_report(config, predictions[1:], samples, now=NOW)
    with pytest.raises(EvaEvalError, match="重复"):
        build_eva_report(config, [*predictions, predictions[0]], samples, now=NOW)
    foreign = predictions[0].model_copy(update={"run_id": "other-run"})
    with pytest.raises(EvaEvalError, match="混有其他运行"):
        build_eva_report(config, [foreign, *predictions[1:]], samples, now=NOW)


def test_report_requires_every_dimension_accounted_for(
    samples: tuple[EvaSftSample, ...],
) -> None:
    replies = [s.messages[-1].text for s in samples]
    report = build_eva_report(
        run_config(samples),
        predictions_for(samples, replies),
        samples,
        now=NOW,
    )
    data = json.loads(report.model_dump_json())
    data["metric_errors"] = {}
    data["metrics"].pop("aesthetic")

    with pytest.raises(ValidationError, match="每个维度"):
        EvaEvalReport.model_validate(data)


def test_read_predictions_round_trip(
    tmp_path: Path, samples: tuple[EvaSftSample, ...]
) -> None:
    run_dir = tmp_path / RUN
    prepare_eva_run_dir(run_dir, run_config(samples))
    prediction = parse_eva_reply(
        RUN, samples[0].image_id, '{"aesthetic":2.0,"technical":2.0}', 4
    )
    append_prediction(run_dir, prediction)

    assert read_predictions(run_dir) == [prediction]


# --- checkpoint selection ---


def with_varied_technical(
    samples: tuple[EvaSftSample, ...],
) -> tuple[EvaSftSample, ...]:
    """The fixture's technical means are all equal; spread them out."""

    varied = []
    for i, sample in enumerate(samples):
        technical = sample.source_labels.technical.model_copy(
            update={"mean": 1.5 + 0.3 * i}
        )
        labels = sample.source_labels.model_copy(update={"technical": technical})
        varied.append(sample.model_copy(update={"source_labels": labels}))
    return tuple(varied)


def tracking_replies(samples: tuple[EvaSftSample, ...]) -> dict[str, str]:
    """Aesthetic copies the target; technical rises with the technical mean."""

    replies = {}
    for i, sample in enumerate(samples):
        aesthetic = json.loads(sample.messages[-1].text)["aesthetic"]
        replies[sample.image_id] = json.dumps(
            {"aesthetic": aesthetic, "technical": round(1.0 + 0.5 * i, 1)}
        )
    return replies


def test_selection_score_averages_dimension_srcc(
    samples: tuple[EvaSftSample, ...],
) -> None:
    varied = with_varied_technical(samples)

    result = selection_score(tracking_replies(varied), varied, min_parse_rate=0.95)

    assert result.ineligible == {}
    assert result.parse_success_rate == 1.0
    assert result.srcc["technical"] == pytest.approx(1.0)
    assert result.srcc["aesthetic"] > 0.9
    assert result.score == pytest.approx(
        (result.srcc["aesthetic"] + result.srcc["technical"]) / 2
    )
    fenced = {
        image_id: f"```json\n{text}\n```"
        for image_id, text in tracking_replies(varied).items()
    }
    assert selection_score(fenced, varied, min_parse_rate=0.95) == result


def test_selection_score_ineligible_below_parse_rate(
    samples: tuple[EvaSftSample, ...],
) -> None:
    varied = with_varied_technical(samples)
    replies = tracking_replies(varied)
    replies[varied[0].image_id] = "```json\nnot json\n```"

    result = selection_score(replies, varied, min_parse_rate=0.95)

    assert result.parse_success_rate == pytest.approx(5 / 6)
    assert set(result.ineligible) == {"parse_success_rate"}
    assert result.score == INELIGIBLE_SELECTION_SCORE
    # the boundary is inclusive: exactly the minimum still qualifies
    exact = selection_score(replies, varied, min_parse_rate=5 / 6)
    assert exact.ineligible == {}


def test_selection_score_ineligible_when_srcc_undefined(
    samples: tuple[EvaSftSample, ...],
) -> None:
    # technical vote means are constant in the fixture
    replies = {s.image_id: s.messages[-1].text for s in samples}

    result = selection_score(replies, samples, min_parse_rate=0.95)

    assert set(result.ineligible) == {"technical"}
    assert "aesthetic" in result.srcc
    assert result.score == INELIGIBLE_SELECTION_SCORE


def test_selection_score_rejects_mismatched_replies(
    samples: tuple[EvaSftSample, ...],
) -> None:
    replies = {s.image_id: s.messages[-1].text for s in samples[1:]}

    with pytest.raises(EvaEvalError, match="不一致"):
        selection_score(replies, samples, min_parse_rate=0.95)


@pytest.mark.parametrize(
    ("ineligible", "score"),
    [({}, INELIGIBLE_SELECTION_SCORE), ({"technical": "x"}, 0.5)],
)
def test_selection_score_record_rejects_inconsistent_score(
    ineligible: dict[str, str], score: float
) -> None:
    with pytest.raises(ValidationError, match="不合格"):
        SelectionScore(
            n_items=2,
            parse_success_rate=1.0,
            srcc={},
            ineligible=ineligible,
            score=score,
        )
