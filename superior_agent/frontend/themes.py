"""Visual theme definitions for the CLI frontend.

Uses Rich styles and a curated colour palette.
"""

from __future__ import annotations

from rich.style import Style
from rich.theme import Theme

# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------

THINKING_COLOR = "#8b8b8b"       # muted grey for thinking blocks
RESPONSE_COLOR = "#e0e0e0"       # near-white for response text
TOOL_NAME_COLOR = "#61afef"      # soft blue for tool names
TOOL_RESULT_COLOR = "#98c379"    # soft green for tool output
STATE_COLOR = "#c678dd"          # purple for state changes
ERROR_COLOR = "#e06c75"          # soft red for errors
ACCENT_COLOR = "#61afef"         # blue accent
BANNER_COLOR = "#56b6c2"         # teal for the startup banner
PROMPT_COLOR = "#e5c07b"         # warm gold for the prompt arrow
DIM_COLOR = "#5c6370"            # dim grey for secondary info

# ------------------------------------------------------------------
# Rich Theme
# ------------------------------------------------------------------

AGENT_THEME = Theme(
    {
        "thinking": Style(color=THINKING_COLOR, italic=True, dim=True),
        "response": Style(color=RESPONSE_COLOR),
        "tool.name": Style(color=TOOL_NAME_COLOR, bold=True),
        "tool.result": Style(color=TOOL_RESULT_COLOR),
        "state": Style(color=STATE_COLOR, bold=True),
        "error": Style(color=ERROR_COLOR, bold=True),
        "accent": Style(color=ACCENT_COLOR),
        "banner": Style(color=BANNER_COLOR, bold=True),
        "prompt": Style(color=PROMPT_COLOR, bold=True),
        "dim": Style(color=DIM_COLOR),
    }
)
