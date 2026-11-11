from __future__ import annotations

import pytest
from qv_head.features import extract_features
from qv_head.probe import ProbeError
from qv_head.probe_run import REPORT_FILE, layer_columns, run_probe
from qv_head.records import FeatureSet
from qv_head.schema import ProbeReport
from tests.helpers import NOW

from tests.conftest import TinyJob


def _features(job: TinyJob) -> FeatureSet:
    return extract_features(
        job.config,
        data_root=job.data_root,
        models_dir=job.models_dir,
        runs_dir=job.runs_dir,
        run_id="tiny-features",
        limit=None,
        now=NOW,
        log=lambda _m: None,
    )


def test_layer_columns_pick_the_right_slices(tiny_job: TinyJob) -> None:
    record = _features(tiny_job)

    assert layer_columns(record, "all") == slice(0, 112)
    assert layer_columns(record, "last") == slice(32, 64)
    assert layer_columns(record, "merger") == slice(64, 112)


def test_probe_reports_every_layer_set_and_score(tiny_job: TinyJob) -> None:
    _features(tiny_job)
    messages: list[str] = []

    def probe(run_id: str) -> ProbeReport:
        return run_probe(
            data_root=tiny_job.data_root,
            runs_dir=tiny_job.runs_dir,
            features_run_id="tiny-features",
            run_id=run_id,
            folds=2,
            bootstrap=100,
            seed=0,
            now=NOW,
            log=messages.append,
        )

    report = probe("tiny-probe")

    assert [(r.layers, r.dimension) for r in report.results] == [
        (layers, dim)
        for layers in ("all", "last", "merger")
        for dim in ("aesthetic", "technical")
    ]
    assert [r.feature_dim for r in report.results[::2]] == [112, 32, 48]
    saved = (tiny_job.runs_dir / "tiny-probe" / REPORT_FILE).read_text()
    assert ProbeReport.model_validate_json(saved) == report
    assert len(messages) == 6 and messages[0].startswith("[探针] all")
    with pytest.raises(ProbeError, match="已存在"):
        probe("tiny-probe")
