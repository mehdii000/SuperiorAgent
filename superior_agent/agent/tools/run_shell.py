"""run_shell — executes a shell command in the working directory."""

from __future__ import annotations

import subprocess
from typing import Any


def run_shell(
    command: str,
    workdir: str,
    platform_profile: dict[str, Any],
    timeout: int = 60,
    max_output_length: int = 10000,
) -> str:
    """Description: Executes a shell command and returns the output (truncated if too long).
    Args: command: The shell command to execute  workdir: Working directory  platform_profile: OS profile  timeout: Timeout in seconds (default 60)  max_output_length: Max characters to return (default 10000)
    Returns: stdout and stderr output
    When to use: Use for system commands, builds, or scripts. Avoid interactive commands."""
    timeout = int(timeout)
    max_output_length = int(max_output_length)
    shell_cmd = _build_shell_cmd(command, platform_profile)
    try:
        result = subprocess.run(
            shell_cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        
        # Check for truncation
        if len(output) > max_output_length:
            header = f"--- Output Truncated ({len(output)} chars total, showing first {max_output_length}) ---\n"
            output = header + output[:max_output_length] + "\n... (truncated)"

        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds."
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
