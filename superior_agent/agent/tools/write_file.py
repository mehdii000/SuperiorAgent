"""write_file — writes content to a file relative to workdir."""

from __future__ import annotations

from pathlib import Path


def write_file(path: str, content: str, workdir: str) -> str:
    """Description: Writes content to a file at the given path, creating directories as needed.
    Args: path: File path relative to workdir  content: Text content to write to the file
    Returns: Confirmation message with the absolute path written
    When to use: When the user asks to create, write, or modify a file"""
    resolved = _resolve(path, workdir)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {resolved}"
    except Exception as exc:  # noqa: BLE001
        return f"Error writing '{path}': {exc}"


def _resolve(path: str, workdir: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(workdir) / p
