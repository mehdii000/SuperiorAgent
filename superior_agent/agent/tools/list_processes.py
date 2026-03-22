def list_processes(brain) -> str:
    """Description: Lists all background processes currently tracked by the agent.
    Args: None
    Returns: A formatted list of PIDs and their commands.
    When to use: When you want to see what is running in the background (like dev servers).
    Tags: process, background, list, running, ps"""
    if not brain or not hasattr(brain, "processes") or not brain.processes:
        return "No background processes are currently being tracked."
    
    lines = ["Currently tracked background processes:"]
    for pid, info in brain.processes.items():
        cmd = info.get("command", "unknown")
        lines.append(f"  - PID {pid}: {cmd}")
    return "\n".join(lines)
