"""edit_file — replaces a block of text in a file relative to workdir."""

from __future__ import annotations

from pathlib import Path


def edit_file(path: str, old_text: str, new_text: str, workdir: str) -> str:
    """Description: Replaces a specific block of text (old_text) with new_text in a file.
    Args: path: File path relative to workdir  old_text: The exact text block to replace  new_text: The text to replace it with
    Returns: Confirmation message or an error message
    When to use: When you want to modify a specific part of a file without overwriting the entire file. Use read_file first to get the exact text to replace."""
    resolved = _resolve(path, workdir)
    
    if not resolved.is_file():
        return f"Error: '{path}' is not a file or does not exist."
    
    try:
        content = resolved.read_text(encoding="utf-8")
        
        if old_text not in content:
            return f"Error: Could not find the exact text block in '{path}'. Make sure old_text matches exactly, including whitespace."
        
        # Check for uniqueness to avoid accidental multiple replacements if not intended
        # But usually, it's fine to replace all occurrences if they match exactly.
        # We'll stick to a simple replace for now.
        count = content.count(old_text)
        new_content = content.replace(old_text, new_text)
        
        resolved.write_text(new_content, encoding="utf-8")
        return f"Successfully edited '{path}'. Replaced {count} occurrence(s)."
        
    except Exception as exc:  # noqa: BLE001
        return f"Error editing '{path}': {exc}"


def _resolve(path: str, workdir: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(workdir) / p
