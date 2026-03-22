import os
import signal

def stop_process(pid: int, brain) -> str:
    """Description: Stops a background process started by the agent using its PID.
    Args: pid: The Process ID to stop
    Returns: Confirmation message
    When to use: When you want to stop a dev server or any long-running command.
    Tags: process, background, stop, kill, terminate"""
    try:
        p_id = int(pid)
        if p_id in brain.processes:
            # Try to terminate gracefully, then kill
            try:
                if os.name == 'nt':
                    # Windows
                    os.system(f"taskkill /F /T /PID {p_id}")
                else:
                    os.kill(p_id, signal.SIGTERM)
                
                info = brain.processes.pop(p_id)
                cmd = info.get("command", "unknown")
                return f"Successfully stopped process {p_id} ({cmd})."
            except Exception as e:
                return f"Error stopping process {p_id}: {e}"
        else:
            return f"Process {p_id} is not in the agent's tracked list."
    except ValueError:
        return f"Invalid PID: {pid}"
