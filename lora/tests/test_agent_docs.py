from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_LINE = 80


def test_claude_md_matches_agents_md() -> None:
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert claude == agents, "CLAUDE.md 与 AGENTS.md 必须保持一致"


@pytest.mark.parametrize(
    "name",
    [
        "CLAUDE.md",
        "AGENTS.md",
        *(p.relative_to(ROOT).as_posix() for p in sorted(ROOT.glob("docs/**/*.md"))),
        *(p.relative_to(ROOT).as_posix() for p in sorted(ROOT.glob("prompts/**/*.md"))),
    ],
)
def test_markdown_lines_within_limit(name: str) -> None:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()

    too_long = [i for i, line in enumerate(lines, 1) if len(line) > MAX_LINE]

    assert not too_long, f"{name} 第 {too_long} 行超过 {MAX_LINE} 字符"
