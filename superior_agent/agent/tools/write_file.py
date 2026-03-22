"""write_file — writes content to a file relative to workdir."""

from __future__ import annotations

from pathlib import Path


import tempfile
import os

def write_file(path: str, content: str, workdir: str, overwrite: bool = True) -> str:
    """Description: Writes content to a file. Optimized to skip writing if content is identical.
    Args: path: File path relative to workdir  content: Text content to write  workdir: The working directory  overwrite: Whether to overwrite if the file exists (default True)
    Returns: Confirmation message
    When to use: Use when creating or updating a file. Use this for whole-file writes."""
    resolved = _resolve(path, workdir)
    try:
        if resolved.exists():
            if not overwrite:
                return f"Error: File '{path}' already exists and overwrite is set to False."
            
            # Identity check to skip redundant I/O
            if resolved.is_file():
                existing_content = resolved.read_text(encoding="utf-8")
                if existing_content == content:
                    return f"Skipped: Content for '{path}' is identical to existing file."

        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write: write to temp file first then rename
        with tempfile.NamedTemporaryFile("w", dir=resolved.parent, encoding="utf-8", delete=False) as tf:
            tf.write(content)
            temp_name = tf.name
        
        try:
            os.replace(temp_name, str(resolved))
        except Exception:
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise

        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as exc:  # noqa: BLE001
        return f"Error writing '{path}': {exc}"


def _resolve(path: str, workdir: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(workdir) / p
