def get_session_info(brain, artifacts) -> str:
    """Description: Get a summary of the current session state, including background processes and stored artifacts.
    Args: None
    Returns: Markdown summary of the session.
    When to use: Use this to check what the user currently sees in their sidebar or to see which background processes are tracked.
    Tags: session, state, context, artifacts, processes, info"""
    lines = ["## Session Information\n"]
    
    # 1. Background Processes
    lines.append("### Background Processes")
    if not brain or not hasattr(brain, "processes") or not brain.processes:
        lines.append("_No active background processes._")
    else:
        for pid, info in brain.processes.items():
            cmd = info.get("command", "unknown")
            lines.append(f"- **PID {pid}**: `{cmd}`")
    
    # 2. Artifacts
    lines.append("\n### Stored Artifacts")
    names = artifacts.list_all()
    if not names:
        lines.append("_No artifacts stored._")
    else:
        for name in names:
            content = artifacts.get(name) or ""
            # Show a small preview or just the headers
            preview = content.split("\n", 1)[0]
            lines.append(f"- **{name}**: {preview}")
            
    return "\n".join(lines)
