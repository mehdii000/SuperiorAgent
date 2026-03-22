"""edit_file — replaces a block of text in a file relative to workdir."""

from __future__ import annotations

from pathlib import Path


def edit_file(
    path: str,
    old_text: str = "",
    new_text: str = "",
    workdir: str = "",
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Description: Replaces a specific block of text or a range of lines with new_text.
    Args: path: File path relative to workdir  old_text: The exact text block to replace (ignored if line numbers are used)  new_text: The text to replace it with  workdir: The working directory  start_line: 1-indexed start line (inclusive)  end_line: 1-indexed end line (inclusive)
    Returns: Confirmation message or an error message
    When to use: Use to modify a specific part of a file. Use line numbers for precision if known, otherwise use old_text for exact matching."""
    resolved = _resolve(path, workdir)
    
    if not resolved.is_file():
        return f"Error: '{path}' is not a file or does not exist."
    
    try:
        content_lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)
        
        if start_line is not None and end_line is not None:
            # Line-based replacement
            if start_line < 1 or end_line < start_line or end_line > len(content_lines):
                return f"Error: Invalid line range {start_line}-{end_line}. File has {len(content_lines)} lines."
            
            # Verify old_text if provided (as a safety check)
            if old_text:
                actual_block = "".join(content_lines[start_line - 1 : end_line])
                if old_text.strip() != actual_block.strip():
                    return f"Error: The provided old_text does not match the content at lines {start_line}-{end_line}."

            # Replace the range
            # Ensure new_text ends with a newline if it's replacing whole lines
            if not new_text.endswith("\n") and end_line < len(content_lines):
                 new_text += "\n"
                 
            content_lines[start_line - 1 : end_line] = [new_text]
            new_content = "".join(content_lines)
            resolved.write_text(new_content, encoding="utf-8")
            return f"Successfully edited '{path}' at lines {start_line}-{end_line}."

        else:
            # Fallback to text-based replacement
            if not old_text:
                return "Error: Either old_text or both start_line and end_line must be provided."
            
            content = "".join(content_lines)
            if old_text not in content:
                return f"Error: Could not find the exact text block in '{path}'. Make sure old_text matches exactly."
            
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
