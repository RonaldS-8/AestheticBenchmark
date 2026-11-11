from __future__ import annotations

import json
from pathlib import Path

import pytest
import run_probe
import torch
from aes_head import backbones
from aes_head.backbones import BackboneSpec
from aes_head.data import sha256_file
from aes_head.run import RunError, format_report
from aes_head.run import run_probe as run
from aes_head.schema import WeightsFile, WeightsIndex
from tests.helpers import NOW, make_rows, tiny_cnn, write_dataset

TINY = BackboneSpec(
    name="tiny_cnn",
    build=tiny_cnn,
    weights_url="https://example.test/tiny_cnn-00000000.pth",
    taps=("1", "3"),
    dims=(8, 16),
)


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setitem(backbones.SPECS, TINY.name, TINY)
    monkeypatch.setattr(run_probe, "BACKBONES", (TINY.name,))
    data, models, runs = tmp_path / "data", tmp_path / "models", tmp_path / "runs"
    data.mkdir()
    models.mkdir()
    write_dataset(data, make_rows(24, 10, 3))
    path = models / "tiny.pth"
    torch.manual_seed(0)
    torch.save(tiny_cnn().state_dict(), path)
    index = WeightsIndex(
        schema_version="1.0",
        files=(
            WeightsFile(
                backbone=TINY.name,
                url=TINY.weights_url,
                path=path.name,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            ),
        ),
    )
    (models / "WEIGHTS.json").write_text(index.model_dump_json(), encoding="utf-8")
    return {"data_root": data, "models_dir": models, "runs_dir": runs}


def _run(layout: dict[str, Path], run_id: str, **overrides: object) -> object:
    options: dict[str, object] = {
        "run_id": run_id,
        "backbones": [TINY.name],
        "device": "cpu",
        "limit": None,
        "folds": 3,
        "alphas": (1.0, 100.0),
        "bootstrap": 100,
        "seed": 0,
        "now": NOW,
        "log": lambda _message: None,
    }
    return run(**layout, **{**options, **overrides})  # type: ignore[arg-type]


def test_run_probe_writes_report_and_cache(layout: dict[str, Path]) -> None:
    report = _run(layout, "probe-test")

    run_dir = layout["runs_dir"] / "probe-test"
    saved = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["n_train"] == 24 and saved["n_val"] == 10
    assert (run_dir / "features-tiny_cnn.pt").is_file()
    assert [(r.layers, r.dimension) for r in report.results] == [  # type: ignore[attr-defined]
        ("all", "aesthetic"),
        ("all", "technical"),
        ("last", "aesthetic"),
        ("last", "technical"),
    ]
    assert "tiny_cnn" in format_report(report)  # type: ignore[arg-type]


def test_run_probe_never_overwrites_and_reuses_cache(layout: dict[str, Path]) -> None:
    first = _run(layout, "probe-test")
    with pytest.raises(RunError, match="已存在"):
        _run(layout, "probe-test")
    (layout["runs_dir"] / "probe-test" / "report.json").unlink()
    messages: list[str] = []

    second = _run(layout, "probe-test", log=messages.append)

    assert any("复用" in m for m in messages)
    assert second.results == first.results  # type: ignore[attr-defined]


def test_run_probe_limit_is_marked(layout: dict[str, Path]) -> None:
    report = _run(layout, "probe-smoke", limit=9)

    assert report.limited and report.n_train == 9  # type: ignore[attr-defined]
    assert "冒烟" in format_report(report)  # type: ignore[arg-type]


def test_run_probe_rejects_foreign_weights(layout: dict[str, Path]) -> None:
    path = layout["models_dir"] / "WEIGHTS.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["files"][0]["url"] = "https://example.test/other-11ad3fa6.pth"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RunError, match="权重来源"):
        _run(layout, "probe-test")


def test_cli_prints_table(
    layout: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--run-id", "probe-cli", "--backbones", TINY.name]
    argv += ["--folds", "3", "--bootstrap", "100", "--limit", "12"]
    argv += [f"--{k.replace('_', '-')}={v}" for k, v in layout.items()]

    assert run_probe.main(argv) == 0
    assert "tiny_cnn" in capsys.readouterr().out
