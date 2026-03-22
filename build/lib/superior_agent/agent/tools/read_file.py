"""read_file — reads the contents of a file relative to workdir."""

from __future__ import annotations

import os
from pathlib import Path


def read_file(path: str, workdir: str) -> str:
    """Description: Reads the contents of a file at the given path.
    Args: path: File path relative to workdir
    Returns: File contents as a string, or an error message
    When to use: When the user asks to see, examine, or inspect a file's contents"""
    resolved = _resolve(path, workdir)
    if not resolved.is_file():
        return f"Error: '{path}' is not a file or does not exist."
    try:
        return resolved.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"Error reading '{path}': {exc}"


def _resolve(path: str, workdir: str) -> Path:
    """Resolve *path* relative to *workdir*, unless it is absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(workdir) / p
