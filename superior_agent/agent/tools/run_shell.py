"""run_shell — executes a shell command in the working directory."""

from __future__ import annotations

import subprocess
from typing import Any


def run_shell(command: str, workdir: str, platform_profile: dict[str, Any]) -> str:
    """Description: Executes a shell command in the working directory and returns the output.
    Args: command: The shell command to execute  workdir: Working directory for the command  platform_profile: OS profile dictionary with shell info
    Returns: Combined stdout and stderr output from the command
    When to use: When the user needs to run a system command, install packages, or execute scripts"""
    shell_cmd = _build_shell_cmd(command, platform_profile)
    try:
        result = subprocess.run(
            shell_cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds."
    except Exception as exc:  # noqa: BLE001
        return f"Error running command: {exc}"


def _build_shell_cmd(command: str, profile: dict[str, Any]) -> str:
    """Wrap the user's command for the detected shell."""
    # On Windows with powershell, wrap differently than bash
    shell = profile.get("shell", "bash")
    if shell == "powershell":
        return f'powershell -NoProfile -Command "{command}"'
    elif shell == "cmd":
        return f'cmd /c "{command}"'
    return command
