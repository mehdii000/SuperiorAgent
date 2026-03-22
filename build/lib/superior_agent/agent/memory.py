"""Session Memory — structured, searchable conversation memory.

Each turn is stored as a `MemoryEntry`.  Pinned entries represent
Critical Knowledge and are never compressed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single turn in session memory."""
    turn_id: int
    role: str          # "user" | "agent"
    summary: str       # compressed representation for search
    full_content: str  # preserved for retrieval
    pinned: bool = False


class SessionMemory:
    """In-memory session history with search, pinning, and compression."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, role: str, content: str, *, pinned: bool = False) -> MemoryEntry:
        """Append a new memory entry.  *summary* defaults to the first 200 chars."""
        entry = MemoryEntry(
            turn_id=self._next_id,
            role=role,
            summary=content[:200].replace("\n", " "),
            full_content=content,
            pinned=pinned,
        )
        self._entries.append(entry)
        self._next_id += 1
        return entry

    # ------------------------------------------------------------------
    # Read / Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, threshold: float = 0.5) -> list[MemoryEntry]:
        """Simple keyword search over summaries and full content.

        Returns entries sorted by relevance (number of query-word hits).
        The *threshold* parameter is reserved for future semantic search.
        """
        words = query.lower().split()
        if not words:
            return []

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            text = (entry.summary + " " + entry.full_content).lower()
            hits = sum(1 for w in words if w in text)
            score = hits / len(words)
            if score >= threshold:
                scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in scored]

    def get_pinned(self) -> list[MemoryEntry]:
        """Return all pinned (Critical Knowledge) entries."""
        return [e for e in self._entries if e.pinned]

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the *n* most recent entries."""
        return self._entries[-n:]

    def all_entries(self) -> list[MemoryEntry]:
        """Return the full session history."""
        return list(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, *, keep_pinned: bool = True) -> int:
        """Replace the oldest non-pinned entries with their summaries.

        Returns the number of entries compressed.
        """
        count = 0
        cutoff = max(0, len(self._entries) - 4)  # keep the 4 most recent
        for i in range(cutoff):
            entry = self._entries[i]
            if keep_pinned and entry.pinned:
                continue
            if entry.full_content != entry.summary:
                entry.full_content = entry.summary
                count += 1
        return count

    # ------------------------------------------------------------------
    # Conversion to LLM messages
    # ------------------------------------------------------------------

    def to_messages(self) -> list[dict[str, str]]:
        """Convert memory to a list of ``{role, content}`` dicts
        suitable for the LLM bridge's ``Message`` format."""
        msgs: list[dict[str, str]] = []
        for entry in self._entries:
            role = "user" if entry.role == "user" else "assistant"
            msgs.append({"role": role, "content": entry.full_content})
        return msgs

    def clear(self) -> None:
        """Reset session memory."""
        self._entries.clear()
        self._next_id = 0
