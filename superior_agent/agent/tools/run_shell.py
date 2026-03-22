"""run_shell — executes a shell command in the working directory."""

from __future__ import annotations

import subprocess
from typing import Any


import asyncio
import asyncio.subprocess

async def run_shell(
    command: str,
    workdir: str,
    platform_profile: dict[str, Any],
    timeout: int = 60,
    max_output_length: int = 10000,
    wait_seconds: int | None = None,
    brain: Any = None,
) -> str:
    """Description: Executes a shell command asynchronously (truncated if too long).
    Args: command: The shell command to execute  workdir: Working directory  platform_profile: OS profile  timeout: Timeout in seconds (default 60)  max_output_length: Max characters to return (default 10000)  wait_seconds: If set, waits this long then backgrounds if still running.
    Returns: stdout and stderr output
    When to use: Use for system commands, builds, or scripts. Avoid interactive commands.
    Tags: shell, command, execute, run, system, process, powershell, bash, cmd, background
    """
    timeout = int(timeout)
    max_output_length = int(max_output_length)
    shell_cmd = _build_shell_cmd(command, platform_profile)
    
    try:
        process = await asyncio.create_subprocess_shell(
            shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL, # Isolate from user terminal
            cwd=workdir,
        )
        
        stdout_chunks = []
        stderr_chunks = []
        
        async def _read_stream(stream, chunks):
            try:
                while True:
                    line = await stream.read(4096)
                    if not line:
                        break
                    chunks.append(line.decode(errors="replace"))
            except Exception:
                 pass

        # Start reading tasks
        stdout_task = asyncio.create_task(_read_stream(process.stdout, stdout_chunks))
        stderr_task = asyncio.create_task(_read_stream(process.stderr, stderr_chunks))

        try:
            if wait_seconds:
                # Wait for completion OR for wait_seconds to expire
                try:
                    await asyncio.wait_for(process.wait(), timeout=int(wait_seconds))
                    status = f"Finished (exit code: {process.returncode})"
                except asyncio.TimeoutError:
                    status = f"Still running in background after {wait_seconds}s (PID: {process.pid})"
                    if brain and hasattr(brain, "processes"):
                        brain.processes[process.pid] = {"command": command, "process": process}
            else:
                # Regular wait with full timeout
                await asyncio.wait_for(process.wait(), timeout=timeout)
                status = f"Finished (exit code: {process.returncode})"
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            status = f"Timed out after {timeout}s (killed)"

        # Stop reading tasks (they'll finish when streams close)
        # If still running, we just take what we have so far
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        
        output = stdout
        if stderr:
             output += ("\n--- stderr ---\n" + stderr) if output else stderr
             
        # Check for truncation
        if len(output) > max_output_length:
            header = f"--- Output Truncated ({len(output)} chars total, showing first {max_output_length}) ---\n"
            output = header + output[:max_output_length] + "\n... (truncated)"

        return f"[{status}]\n{output.strip() or '(no output)'}"
        
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
