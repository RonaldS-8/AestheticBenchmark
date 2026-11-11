"""Progress output that shows up at once, even when piped to ``tee`` or a file."""

from __future__ import annotations


def echo(message: str) -> None:
    """Print and flush, so ``tail -F`` on a log sees each line immediately."""

    print(message, flush=True)
