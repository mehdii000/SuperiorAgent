"""list_directory — lists files and directories at a given path."""

from __future__ import annotations

import os
from pathlib import Path


def list_directory(path: str, workdir: str) -> str:
    """Description: Lists files and directories at the given path.
    Args: path: Directory path relative to workdir
    Returns: Formatted directory listing with file types and sizes
    When to use: When the user asks to see what files exist in a directory"""
    resolved = _resolve(path, workdir)
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory or does not exist."
    try:
        entries: list[str] = []
        for item in sorted(resolved.iterdir()):
            if item.is_dir():
                entries.append(f"  📁 {item.name}/")
            else:
                size = item.stat().st_size
                entries.append(f"  📄 {item.name}  ({_human_size(size)})")
        if not entries:
            return f"Directory '{path}' is empty."
        header = f"Contents of {resolved}:\n"
        return header + "\n".join(entries)
    except Exception as exc:  # noqa: BLE001
        return f"Error listing '{path}': {exc}"


def _resolve(path: str, workdir: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(workdir) / p


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"
