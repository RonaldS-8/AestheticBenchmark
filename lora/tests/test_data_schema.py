from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from lora_sft.data.schema import (
    Annotation,
    AnnotationRecord,
    Category,
    SplitError,
    SplitManifest,
    check_split,
    ensure_not_eval,
    eval_fingerprint,
    mean_scores,
)

SHA_EVAL = "a" * 64
SHA_TRAIN = "b" * 64
SHA_VAL = "c" * 64


def make_record(
    base: dict[str, Any], image_id: str, sha: str, n_annotators: int = 3
) -> AnnotationRecord:
    data = copy.deepcopy(base)
    data.update(
        image_id=image_id,
        image_path=f"images/{image_id}.jpg",
        image_sha256=sha,
        annotations=data["annotations"][:n_annotators],
    )
    annotations = [Annotation.model_validate(a) for a in data["annotations"]]
    data["mean"] = mean_scores(annotations).model_dump()
    return AnnotationRecord.model_validate(data)


def make_manifest(
    train: tuple[str, ...] = ("photo_000002",),
    val: tuple[str, ...] = ("photo_000003",),
    eval_ids: tuple[str, ...] = ("photo_000001",),
) -> SplitManifest:
    return SplitManifest(
        schema_version="1.0",
        split_version="split-v1",
        created_at="2026-11-15T10:00:00+08:00",
        seed=7,
        train=train,
        val=val,
        eval=eval_ids,
        eval_fingerprint=eval_fingerprint(eval_ids),
    )


def test_fixture_is_valid(annotation_data: dict[str, Any]) -> None:
    record = AnnotationRecord.model_validate(annotation_data)

    assert record.category is Category.PORTRAIT
    assert (record.mean.aesthetic, record.mean.technical, record.mean.overall) == (
        4.33,
        2.67,
        3.67,
    )


def test_json_round_trip(annotation_data: dict[str, Any]) -> None:
    record = AnnotationRecord.model_validate(annotation_data)

    assert AnnotationRecord.model_validate_json(record.model_dump_json()) == record


def test_mean_scores_requires_annotations() -> None:
    with pytest.raises(ValueError, match="没有标注"):
        mean_scores([])


def test_stored_mean_must_match(annotation_data: dict[str, Any]) -> None:
    annotation_data["mean"]["overall"] = 3.7

    with pytest.raises(ValidationError, match="mean 与标注不一致"):
        AnnotationRecord.model_validate(annotation_data)


def test_duplicate_annotator_rejected(annotation_data: dict[str, Any]) -> None:
    annotation_data["annotations"][1]["annotator"] = "A01"

    with pytest.raises(ValidationError, match="重复标注"):
        AnnotationRecord.model_validate(annotation_data)


def test_empty_annotations_rejected(annotation_data: dict[str, Any]) -> None:
    annotation_data["annotations"] = []

    with pytest.raises(ValidationError, match="annotations"):
        AnnotationRecord.model_validate(annotation_data)


def test_image_path_must_match_id(annotation_data: dict[str, Any]) -> None:
    annotation_data["image_path"] = "images/photo_000999.jpg"

    with pytest.raises(ValidationError, match="必须与 image_id"):
        AnnotationRecord.model_validate(annotation_data)


@pytest.mark.parametrize("field", ["category", "guideline_version"])
def test_bad_enum_like_fields_rejected(
    annotation_data: dict[str, Any], field: str
) -> None:
    annotation_data[field] = "nonsense"

    with pytest.raises(ValidationError, match=field):
        AnnotationRecord.model_validate(annotation_data)


@pytest.mark.parametrize(
    ("score", "valid"), [(1, True), (5, True), (0, False), (6, False)]
)
def test_annotation_score_boundary(
    annotation_data: dict[str, Any], score: int, valid: bool
) -> None:
    data = annotation_data["annotations"][0] | {"overall": score}

    if valid:
        assert Annotation.model_validate(data).overall == score
    else:
        with pytest.raises(ValidationError, match="overall"):
            Annotation.model_validate(data)


@pytest.mark.parametrize(
    ("length", "valid"), [(1, True), (100, True), (0, False), (101, False)]
)
def test_reason_length_boundary(
    annotation_data: dict[str, Any], length: int, valid: bool
) -> None:
    data = annotation_data["annotations"][0] | {"reason": "好" * length}

    if valid:
        assert len(Annotation.model_validate(data).reason) == length
    else:
        with pytest.raises(ValidationError, match="reason"):
            Annotation.model_validate(data)


def test_annotator_id_format(annotation_data: dict[str, Any]) -> None:
    data = annotation_data["annotations"][0] | {"annotator": "A1"}

    with pytest.raises(ValidationError, match="annotator"):
        Annotation.model_validate(data)


def test_annotation_needs_timezone(annotation_data: dict[str, Any]) -> None:
    data = annotation_data["annotations"][0] | {"annotated_at": "2026-11-10T09:00:00"}

    with pytest.raises(ValidationError, match="annotated_at"):
        Annotation.model_validate(data)


@pytest.mark.parametrize("field", ["highlights", "issues"])
def test_duplicate_dimension_rejected(
    annotation_data: dict[str, Any], field: str
) -> None:
    data = annotation_data["annotations"][0] | {"highlights": [], "issues": []}
    data[field] = ["color", "color"]

    with pytest.raises(ValidationError, match=f"{field} 中有重复维度"):
        Annotation.model_validate(data)


def test_dimension_cannot_be_highlight_and_issue(
    annotation_data: dict[str, Any],
) -> None:
    data = annotation_data["annotations"][0] | {
        "highlights": ["focus"],
        "issues": ["focus"],
    }

    with pytest.raises(ValidationError, match="同时是亮点和问题"):
        Annotation.model_validate(data)


def test_empty_dimension_lists_allowed(annotation_data: dict[str, Any]) -> None:
    data = annotation_data["annotations"][0] | {"highlights": [], "issues": []}

    assert Annotation.model_validate(data).issues == ()


def test_eval_fingerprint_is_order_independent() -> None:
    assert eval_fingerprint(["photo_000002", "photo_000001"]) == eval_fingerprint(
        ["photo_000001", "photo_000002"]
    )


def test_split_manifest_valid() -> None:
    manifest = make_manifest()

    assert manifest.eval == ("photo_000001",)


@pytest.mark.parametrize(
    ("train", "val"),
    [
        (("photo_000001",), ("photo_000003",)),
        (("photo_000002",), ("photo_000002",)),
        (("photo_000002", "photo_000002"), ("photo_000003",)),
    ],
)
def test_split_overlap_rejected(train: tuple[str, ...], val: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="同时出现在"):
        make_manifest(train=train, val=val)


def test_split_fingerprint_must_match() -> None:
    data = make_manifest().model_dump() | {"eval_fingerprint": "0" * 64}

    with pytest.raises(ValidationError, match="评测集被改动"):
        SplitManifest.model_validate(data)


@pytest.mark.parametrize("split", ["train", "val", "eval"])
def test_split_parts_cannot_be_empty(split: str) -> None:
    data = make_manifest().model_dump() | {split: ()}

    with pytest.raises(ValidationError, match=split):
        SplitManifest.model_validate(data)


def records_for(
    annotation_data: dict[str, Any],
    eval_annotators: int = 3,
    train_sha: str = SHA_TRAIN,
) -> dict[str, AnnotationRecord]:
    return {
        "photo_000001": make_record(
            annotation_data, "photo_000001", SHA_EVAL, eval_annotators
        ),
        "photo_000002": make_record(annotation_data, "photo_000002", train_sha, 1),
        "photo_000003": make_record(annotation_data, "photo_000003", SHA_VAL, 1),
    }


def test_check_split_passes(annotation_data: dict[str, Any]) -> None:
    check_split(make_manifest(), records_for(annotation_data))


def test_check_split_missing_record(annotation_data: dict[str, Any]) -> None:
    records = records_for(annotation_data)
    del records["photo_000003"]

    with pytest.raises(SplitError, match="photo_000003 没有标注记录"):
        check_split(make_manifest(), records)


def test_check_split_eval_needs_three_annotators(
    annotation_data: dict[str, Any],
) -> None:
    with pytest.raises(SplitError, match="只有 2 人标注"):
        check_split(make_manifest(), records_for(annotation_data, eval_annotators=2))


def test_check_split_detects_same_file_in_train_and_eval(
    annotation_data: dict[str, Any],
) -> None:
    with pytest.raises(SplitError, match="是同一文件"):
        check_split(make_manifest(), records_for(annotation_data, train_sha=SHA_EVAL))


def test_ensure_not_eval_accepts_training_ids() -> None:
    ensure_not_eval(["photo_000002", "photo_009999"], make_manifest())


def test_ensure_not_eval_rejects_eval_ids() -> None:
    with pytest.raises(SplitError, match="photo_000001"):
        ensure_not_eval(["photo_000002", "photo_000001"], make_manifest())
