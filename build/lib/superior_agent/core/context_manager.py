"""Context Manager — monitors and compresses the LLM context window.

Implements the tiered compression strategy from §3.3:
  < 60 %  → no action
  60–80 % → summarise oldest non-pinned turns
  80–90 % → aggressive summarisation; warn agent
  > 90 %  → hard stop — force summarisation before next LLM call
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .models import ContextStats, Message

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Returned by `check_and_compress` to describe what happened."""
    messages: list[Message]
    was_compressed: bool = False
    warning: str = ""
    stats: ContextStats | None = None


class ContextManager:
    """Monitors token usage and compresses history when thresholds are hit."""

    def __init__(
        self,
        max_tokens: int = 32_768,
        *,
        low_threshold: float = 0.60,
        mid_threshold: float = 0.80,
        high_threshold: float = 0.90,
    ) -> None:
        self.max_tokens = max_tokens
        self.low_threshold = low_threshold
        self.mid_threshold = mid_threshold
        self.high_threshold = high_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_compress(
        self,
        messages: list[Message],
        pinned_ids: set[int],
        token_counter: Any,       # callable(list[Message]) → int
    ) -> CompressionResult:
        """Evaluate context utilisation and compress if necessary.

        Parameters
        ----------
        messages:
            Full message history (most-recent last).
        pinned_ids:
            Set of *message indices* that must never be compressed.
            Index 0 is normally the system prompt.
        token_counter:
            A callable that accepts ``list[Message]`` and returns the
            total token count.

        Returns
        -------
        CompressionResult
            Possibly shortened message list, compression flag, and any
            warning string.
        """
        used = token_counter(messages)
        pct = used / self.max_tokens if self.max_tokens else 0

        stats = ContextStats(used_tokens=used, max_tokens=self.max_tokens)
        result = CompressionResult(messages=list(messages), stats=stats)

        if pct < self.low_threshold:
            return result

        if pct >= self.high_threshold:
            # Critical — force aggressive compression
            logger.warning(
                "Context at %.0f%% — forcing aggressive compression", pct * 100
            )
            result.messages = self._aggressive_compress(messages, pinned_ids)
            result.was_compressed = True
            result.warning = f"Context critically full ({pct*100:.0f}%). Aggressively compressed."
        elif pct >= self.mid_threshold:
            # High — aggressive summarisation + warn
            logger.info("Context at %.0f%% — aggressive summarisation", pct * 100)
            result.messages = self._aggressive_compress(messages, pinned_ids)
            result.was_compressed = True
            result.warning = f"Context usage high ({pct*100:.0f}%). Summarised old turns."
        else:
            # Moderate — summarise oldest non-pinned
            logger.info("Context at %.0f%% — light summarisation", pct * 100)
            result.messages = self._light_compress(messages, pinned_ids)
            result.was_compressed = True

        # Recount after compression
        new_used = token_counter(result.messages)
        result.stats = ContextStats(used_tokens=new_used, max_tokens=self.max_tokens)
        return result

    # ------------------------------------------------------------------
    # Compression strategies
    # ------------------------------------------------------------------

    def _light_compress(
        self,
        messages: list[Message],
        pinned_ids: set[int],
    ) -> list[Message]:
        """Summarise the oldest *half* of non-pinned, non-system messages."""
        return self._compress(messages, pinned_ids, fraction=0.5)

    def _aggressive_compress(
        self,
        messages: list[Message],
        pinned_ids: set[int],
    ) -> list[Message]:
        """Summarise the oldest ¾ of non-pinned, non-system messages."""
        return self._compress(messages, pinned_ids, fraction=0.75)

    @staticmethod
    def _compress(
        messages: list[Message],
        pinned_ids: set[int],
        fraction: float,
    ) -> list[Message]:
        """Replace the oldest *fraction* of compressible messages with a
        single summary message.  Pinned messages and the system prompt
        (index 0) are always preserved.
        """
        if len(messages) <= 2:
            return list(messages)

        # Identify compressible (non-pinned, non-system) messages
        compressible: list[int] = []
        for i, msg in enumerate(messages):
            if i in pinned_ids:
                continue
            if msg.role == "system":
                continue
            compressible.append(i)

        if not compressible:
            return list(messages)

        cut = max(1, int(len(compressible) * fraction))
        to_remove = set(compressible[:cut])

        # Build a summary of the removed messages
        summaries: list[str] = []
        for idx in sorted(to_remove):
            m = messages[idx]
            # Truncate very long messages in the summary
            body = m.content[:300] + ("…" if len(m.content) > 300 else "")
            summaries.append(f"[{m.role}] {body}")

        summary_block = (
            "[Compressed conversation history]\n" + "\n".join(summaries)
        )

        # Rebuild: system prompt(s) first, then summary, then surviving messages
        result: list[Message] = []
        summary_inserted = False
        for i, msg in enumerate(messages):
            if i in to_remove:
                if not summary_inserted:
                    result.append(Message(role="system", content=summary_block))
                    summary_inserted = True
                continue
            result.append(msg)
        return result
