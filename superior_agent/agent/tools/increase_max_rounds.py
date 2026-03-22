def increase_max_rounds(brain, increment: int = 10) -> str:
    """Description: Increases the maximum number of tool execution rounds the agent can perform.
    Args: increment: Number of rounds to add (default 10)
    Returns: Confirmation message with the new limit
    When to use: When you are hitting the round limit (e.g., Round 18/20) but need more steps to finish a complex task."""
    try:
        inc = int(increment)
        brain.max_tool_rounds += inc
        return f"Successfully increased max tool rounds by {inc}. New limit: {brain.max_tool_rounds}"
    except Exception as exc:
        return f"Error increasing rounds: {exc}"
