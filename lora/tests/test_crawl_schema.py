from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from lora_sft.crawl.schema import (
    CapturedResponse,
    ComplianceError,
    LicenseType,
    PhotoCandidate,
    PhotoRecord,
    TrainingPurpose,
    ensure_trainable,
)


def test_fixture_is_valid_record(record_data: dict[str, Any]) -> None:
    record = PhotoRecord.model_validate(record_data)

    assert record.image_id == "photo_000123"
    assert record.source.license.license_type is LicenseType.CC0
    assert record.signals.comments is None
    assert record.site_tags == ("portrait", "backlight")


def test_json_round_trip(record_data: dict[str, Any]) -> None:
    record = PhotoRecord.model_validate(record_data)

    assert PhotoRecord.model_validate_json(record.model_dump_json()) == record


def test_capture_fixture_is_valid(capture_data: dict[str, Any]) -> None:
    capture = CapturedResponse.model_validate(capture_data)

    assert capture.body_path == "raw/example_site/example_site-20261102-000001.json"


def test_record_is_frozen(record_data: dict[str, Any]) -> None:
    record = PhotoRecord.model_validate(record_data)

    with pytest.raises(ValidationError):
        record.image_id = "photo_000124"


def test_unknown_key_rejected(record_data: dict[str, Any]) -> None:
    record_data["source"]["extra_field"] = "x"

    with pytest.raises(ValidationError, match="extra_field"):
        PhotoRecord.model_validate(record_data)


def test_wrong_schema_version_rejected(record_data: dict[str, Any]) -> None:
    record_data["schema_version"] = "0.9"

    with pytest.raises(ValidationError, match="schema_version"):
        PhotoRecord.model_validate(record_data)


def test_omitted_signal_rejected(record_data: dict[str, Any]) -> None:
    del record_data["signals"]["views"]

    with pytest.raises(ValidationError, match="views"):
        PhotoRecord.model_validate(record_data)


def test_negative_signal_rejected(record_data: dict[str, Any]) -> None:
    record_data["signals"]["likes"] = -1

    with pytest.raises(ValidationError, match="likes"):
        PhotoRecord.model_validate(record_data)


@pytest.mark.parametrize(
    ("image_id", "valid"),
    [
        ("photo_000001", True),
        ("photo_1000000", True),
        ("photo_12345", False),
        ("img_000001", False),
    ],
)
def test_image_id_format(
    record_data: dict[str, Any], image_id: str, valid: bool
) -> None:
    record_data["image_id"] = image_id
    record_data["image"]["path"] = f"images/{image_id}.jpg"

    if valid:
        assert PhotoRecord.model_validate(record_data).image_id == image_id
    else:
        with pytest.raises(ValidationError, match="image_id"):
            PhotoRecord.model_validate(record_data)


def test_image_name_must_match_id(record_data: dict[str, Any]) -> None:
    record_data["image"]["path"] = "images/photo_000999.jpg"

    with pytest.raises(ValidationError, match="必须与 image_id"):
        PhotoRecord.model_validate(record_data)


@pytest.mark.parametrize(("width", "valid"), [(1, True), (0, False), (-5, False)])
def test_image_width_boundary(
    record_data: dict[str, Any], width: int, valid: bool
) -> None:
    record_data["image"]["width"] = width

    if valid:
        assert PhotoRecord.model_validate(record_data).image.width == width
    else:
        with pytest.raises(ValidationError, match="width"):
            PhotoRecord.model_validate(record_data)


@pytest.mark.parametrize(
    ("fmt", "path", "valid"),
    [
        ("jpeg", "images/photo_000123.JPEG", True),
        ("png", "images/photo_000123.png", True),
        ("webp", "images/photo_000123.webp", True),
        ("png", "images/photo_000123.jpg", False),
        ("jpeg", "images/photo_000123", False),
    ],
)
def test_image_suffix_matches_format(
    record_data: dict[str, Any], fmt: str, path: str, valid: bool
) -> None:
    record_data["image"].update(format=fmt, path=path)

    if valid:
        assert PhotoRecord.model_validate(record_data).image.format == fmt
    else:
        with pytest.raises(ValidationError, match="不一致"):
            PhotoRecord.model_validate(record_data)


def test_bad_sha256_rejected(record_data: dict[str, Any]) -> None:
    record_data["image"]["sha256"] = "ABC"

    with pytest.raises(ValidationError, match="sha256"):
        PhotoRecord.model_validate(record_data)


def test_naive_datetime_rejected(record_data: dict[str, Any]) -> None:
    record_data["downloaded_at"] = "2026-11-02T10:15:00"

    with pytest.raises(ValidationError, match="downloaded_at"):
        PhotoRecord.model_validate(record_data)


@pytest.mark.parametrize(
    "body_path",
    [
        "/raw/a.json",
        "C:/raw/a.json",
        "raw\\a.json",
        "../raw/a.json",
        "raw/../../a.json",
        "./raw/a.json",
        "raw//a.json",
        "raw/a.json/",
        "",
        ".",
    ],
)
def test_unsafe_relative_path_rejected(
    capture_data: dict[str, Any], body_path: str
) -> None:
    capture_data["body_path"] = body_path

    with pytest.raises(ValidationError, match="相对 POSIX 路径"):
        CapturedResponse.model_validate(capture_data)


@pytest.mark.parametrize(("status", "valid"), [(100, True), (599, True), (600, False)])
def test_status_code_boundary(
    capture_data: dict[str, Any], status: int, valid: bool
) -> None:
    capture_data["status_code"] = status

    if valid:
        assert CapturedResponse.model_validate(capture_data).status_code == status
    else:
        with pytest.raises(ValidationError, match="status_code"):
            CapturedResponse.model_validate(capture_data)


@pytest.mark.parametrize(
    ("mean", "valid"), [(1.0, True), (10.0, True), (0.99, False), (10.01, False)]
)
def test_rating_mean_within_scale(
    record_data: dict[str, Any], mean: float, valid: bool
) -> None:
    record_data["signals"]["rating"]["mean"] = mean

    if valid:
        rating = PhotoRecord.model_validate(record_data).signals.rating
        assert rating is not None and rating.mean == mean
    else:
        with pytest.raises(ValidationError, match="超出评分区间"):
            PhotoRecord.model_validate(record_data)


def test_rating_scale_must_be_ordered(record_data: dict[str, Any]) -> None:
    record_data["signals"]["rating"].update(scale_min=5, scale_max=5, mean=5)

    with pytest.raises(ValidationError, match="scale_min"):
        PhotoRecord.model_validate(record_data)


@pytest.mark.parametrize(
    ("training", "commercial"), [(True, False), (False, True), (True, True)]
)
def test_unknown_license_cannot_be_marked_usable(
    record_data: dict[str, Any], training: bool, commercial: bool
) -> None:
    record_data["source"]["license"].update(
        license_type="unknown",
        training_allowed=training,
        commercial_allowed=commercial,
    )

    with pytest.raises(ValidationError, match="授权类型未知"):
        PhotoRecord.model_validate(record_data)


def test_unknown_license_allowed_when_marked_unusable(
    record_data: dict[str, Any],
) -> None:
    record_data["source"]["license"].update(
        license_type="unknown", training_allowed=False, commercial_allowed=False
    )

    record = PhotoRecord.model_validate(record_data)

    assert record.source.license.license_type is LicenseType.UNKNOWN


@pytest.mark.parametrize("license_type", ["cc_by_nc", "cc_by_nc_sa", "cc_by_nc_nd"])
def test_non_commercial_license_cannot_allow_commercial(
    record_data: dict[str, Any], license_type: str
) -> None:
    record_data["source"]["license"]["license_type"] = license_type

    with pytest.raises(ValidationError, match="不允许商用"):
        PhotoRecord.model_validate(record_data)


def test_ensure_trainable_accepts_licensed_record(
    record_data: dict[str, Any],
) -> None:
    ensure_trainable(
        PhotoRecord.model_validate(record_data), TrainingPurpose.COMMERCIAL
    )


def test_ensure_trainable_accepts_candidate(candidate_data: dict[str, Any]) -> None:
    ensure_trainable(
        PhotoCandidate.model_validate(candidate_data), TrainingPurpose.COMMERCIAL
    )


def test_ensure_trainable_rejects_training_forbidden(
    candidate_data: dict[str, Any],
) -> None:
    candidate_data["source"]["license"].update(
        license_type="proprietary", training_allowed=False
    )

    with pytest.raises(ComplianceError, match="example_site/889123: 授权不允许用于"):
        ensure_trainable(
            PhotoCandidate.model_validate(candidate_data), TrainingPurpose.RESEARCH
        )


def test_ensure_trainable_rejects_non_commercial(
    candidate_data: dict[str, Any],
) -> None:
    candidate_data["source"]["license"].update(
        license_type="cc_by_nc", commercial_allowed=False
    )

    with pytest.raises(ComplianceError, match="授权不允许商用"):
        ensure_trainable(
            PhotoCandidate.model_validate(candidate_data), TrainingPurpose.COMMERCIAL
        )


def test_ensure_trainable_accepts_research_only(
    candidate_data: dict[str, Any],
) -> None:
    candidate_data["source"]["license"].update(
        license_type="cc_by_nc", commercial_allowed=False
    )

    ensure_trainable(
        PhotoCandidate.model_validate(candidate_data), TrainingPurpose.RESEARCH
    )
