from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_sft.train.config import LoraSftConfig, load_train_config
from lora_sft.train.resume import (
    BUILD_INFO,
    CHECKPOINT_FILES,
    TRAINER_STATE,
    ResumeError,
    check_resume_environment,
    checkpoint_problem,
    find_resume_point,
    source_version,
)
from scripts.train_lora_sft import check_output_dir, record_resume

ROOT = Path(__file__).resolve().parents[1]
ENV = {"torch": "2.8.0", "transformers": "5.20.2", "peft": "0.21.2"}
# An extracted upload archive has BUILD_INFO.json instead of .git.
needs_git = pytest.mark.skipif(
    not (ROOT / ".git").exists(), reason="仓库根没有 .git（例如解压的上传包）"
)


def make_checkpoint(
    output_dir: Path, step: int, *, skip: str | None = None, state_step: int = -1
) -> Path:
    path = output_dir / f"checkpoint-{step}"
    path.mkdir(parents=True)
    for name in CHECKPOINT_FILES:
        if name == skip:
            continue
        if name == TRAINER_STATE:
            recorded = step if state_step < 0 else state_step
            (path / name).write_text(json.dumps({"global_step": recorded}))
        else:
            (path / name).write_bytes(b"x")
    return path


def test_newest_complete_checkpoint_wins(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 102)
    make_checkpoint(tmp_path, 204)
    (tmp_path / "checkpoint-notes").mkdir()
    (tmp_path / "metrics.jsonl").write_text("")

    point = find_resume_point(tmp_path)

    assert (point.step, point.path.name, point.skipped) == (204, "checkpoint-204", ())


def test_half_written_checkpoint_is_passed_over(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 102)
    make_checkpoint(tmp_path, 204, skip=TRAINER_STATE)

    point = find_resume_point(tmp_path)

    assert point.step == 102
    assert point.skipped == (("checkpoint-204", f"缺少 {TRAINER_STATE}"),)


def test_numeric_order_not_text_order(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 99)
    make_checkpoint(tmp_path, 102)

    assert find_resume_point(tmp_path).step == 102


def test_truncated_or_mismatched_state_is_incomplete(tmp_path: Path) -> None:
    truncated = make_checkpoint(tmp_path, 306)
    (truncated / TRAINER_STATE).write_text('{"global_step": 3')
    mismatched = make_checkpoint(tmp_path, 408, state_step=102)

    assert "无法解析" in str(checkpoint_problem(truncated, 306))
    assert "与目录名不符" in str(checkpoint_problem(mismatched, 408))


@pytest.mark.parametrize("missing", ["optimizer.pt", "adapter_model.safetensors"])
def test_missing_files_are_reported(tmp_path: Path, missing: str) -> None:
    path = make_checkpoint(tmp_path, 102, skip=missing)

    assert checkpoint_problem(path, 102) == f"缺少 {missing}"


def test_no_complete_checkpoint_fails_with_instructions(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 102, skip="scheduler.pt")

    with pytest.raises(ResumeError, match="checkpoint-102: 缺少 scheduler.pt"):
        find_resume_point(tmp_path)
    with pytest.raises(ResumeError, match="没有检查点目录"):
        find_resume_point(tmp_path / "never-started")


def test_environment_must_match_on_pinned_packages() -> None:
    check_resume_environment({**ENV, "devices": ["A10"]}, {**ENV, "devices": ["A100"]})

    with pytest.raises(ResumeError, match="transformers 5.20.2 → 5.22.1"):
        check_resume_environment(ENV, {**ENV, "transformers": "5.22.1"})


@needs_git
def test_source_version_from_git_checkout() -> None:
    version = source_version(ROOT)

    assert version["source"] == "git"
    assert len(version["commit"]) == 40
    assert isinstance(version["uncommitted"], list)


def test_source_version_from_archive_and_missing(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match=BUILD_INFO):
        source_version(tmp_path)

    info = {"commit": "a" * 40, "uncommitted": ["README.md"], "packed_at": "x"}
    (tmp_path / BUILD_INFO).write_text(json.dumps(info), encoding="utf-8")

    assert source_version(tmp_path) == {
        "source": "archive",
        "commit": "a" * 40,
        "uncommitted": ["README.md"],
    }


def stored_run(output_dir: Path, config: LoraSftConfig) -> None:
    output_dir.mkdir(parents=True)
    info = {"config": config.model_dump(mode="json"), "environment": ENV}
    (output_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")


def eva_config() -> LoraSftConfig:
    config = load_train_config(ROOT / "configs" / "eva-qwen35-4b-lora-r16.toml")
    assert isinstance(config, LoraSftConfig)
    return config


def test_output_dir_new_run_and_refusals(tmp_path: Path) -> None:
    config = eva_config()
    run = tmp_path / "run"

    assert check_output_dir(run, config, resume=False) is None
    run.mkdir()
    with pytest.raises(FileExistsError, match="RESUME=1"):
        check_output_dir(run, config, resume=False)
    with pytest.raises(FileNotFoundError, match="run_info.json"):
        check_output_dir(run, config, resume=True)


def test_output_dir_resume_needs_same_config_and_checkpoint(tmp_path: Path) -> None:
    config = eva_config()
    run = tmp_path / "run"
    stored_run(run, config)

    with pytest.raises(ResumeError, match="没有检查点目录"):
        check_output_dir(run, config, resume=True)
    make_checkpoint(run, 102)
    point = check_output_dir(run, config, resume=True)
    assert point is not None and point.step == 102

    other = config.model_copy(update={"seed": 7})
    with pytest.raises(ValueError, match="配置与本次不同"):
        check_output_dir(run, other, resume=True)


@needs_git
def test_record_resume_appends_history_and_marker(tmp_path: Path) -> None:
    run = tmp_path / "run"
    stored_run(run, eva_config())
    make_checkpoint(run, 102)
    make_checkpoint(run, 204, skip="optimizer.pt")
    (run / "metrics.jsonl").write_text('{"step": 110, "loss": 1.0}\n')
    point = find_resume_point(run)

    record_resume(run, point, ENV)
    record_resume(run, point, ENV)

    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    assert [r["from_step"] for r in info["resumes"]] == [102, 102]
    assert info["resumes"][0]["skipped_incomplete"] == [
        ["checkpoint-204", "缺少 optimizer.pt"]
    ]
    assert info["resumes"][0]["source"]["source"] == "git"
    lines = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"] == "resume"
    assert len(lines) == 3
