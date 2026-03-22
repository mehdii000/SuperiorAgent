"""Shared data models for the core LLM bridge layer.

All inter-layer communication uses these types — no module imports
across layer boundaries.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# LLM Event Types — emitted by the streaming bridge
# ---------------------------------------------------------------------------

class EventType(enum.Enum):
    """Discriminator for the LLMEvent union."""
    THINKING_CHUNK = "thinking_chunk"
    RESPONSE_CHUNK = "response_chunk"
    TOOL_CALL = "tool_call"
    STATE_CHANGE = "state_change"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LLMEvent:
    """A single event emitted by the LLM bridge during streaming."""
    type: EventType
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""
    new_state: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Message — the wire format for LLM conversation turns
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Message:
    """A single message in the LLM conversation."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


# ---------------------------------------------------------------------------
# Context statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ContextStats:
    """Snapshot of context-window utilisation."""
    used_tokens: int
    max_tokens: int

    @property
    def usage_pct(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return (self.used_tokens / self.max_tokens) * 100.0

    @property
    def level(self) -> str:
        """Human label for the current usage band."""
        pct = self.usage_pct
        if pct < 60:
            return "normal"
        if pct < 80:
            return "moderate"
        if pct < 90:
            return "high"
        return "critical"


# ---------------------------------------------------------------------------
# Tier classification (returned by the complexity-decision call)
# ---------------------------------------------------------------------------

class Tier(enum.Enum):
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass(frozen=True, slots=True)
class TierDecision:
    """Result of the agent's complexity classification."""
    tier: Tier
    rationale: str
    requires_tool: bool
