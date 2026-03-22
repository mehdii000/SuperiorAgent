"""update_artifact — updates a virtual artifact (tasks, implementation_plan) in the session store."""

from __future__ import annotations

from typing import Any


def update_artifact(name: str, content: str, artifacts: Any) -> str:
    """Description: Updates the content of a virtual artifact (e.g., 'tasks' or 'implementation_plan').
    Args: name: Artifact name ('tasks', 'implementation_plan', etc.)  content: New markdown content for the artifact
    Returns: Confirmation message
    When to use: When you want to update your task list or implementation plan so they are visible in the UI sidebar."""
    try:
        artifacts.upsert(name, content)
        return f"Successfully updated artifact '{name}'."
    except Exception as exc:  # noqa: BLE001
        return f"Error updating artifact '{name}': {exc}"
