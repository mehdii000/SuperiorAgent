"""Tests for agent.memory — add, search, compress, pinned."""

from __future__ import annotations

import pytest

from superior_agent.agent.memory import SessionMemory


@pytest.fixture
def mem() -> SessionMemory:
    return SessionMemory()


# ------------------------------------------------------------------
# Basic operations
# ------------------------------------------------------------------

class TestAdd:
    def test_add_increments_size(self, mem: SessionMemory):
        mem.add("user", "Hello")
        assert mem.size == 1
        mem.add("agent", "Hi there")
        assert mem.size == 2

    def test_add_assigns_turn_id(self, mem: SessionMemory):
        e1 = mem.add("user", "First")
        e2 = mem.add("user", "Second")
        assert e1.turn_id == 0
        assert e2.turn_id == 1

    def test_add_pinned(self, mem: SessionMemory):
        e = mem.add("user", "Critical info", pinned=True)
        assert e.pinned is True

    def test_summary_is_truncated(self, mem: SessionMemory):
        long_text = "A" * 500
        e = mem.add("user", long_text)
        assert len(e.summary) <= 200


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

class TestSearch:
    def test_search_finds_match(self, mem: SessionMemory):
        mem.add("user", "How do I install Python?")
        mem.add("agent", "Use the official installer.")
        results = mem.search("install Python")
        assert len(results) >= 1

    def test_search_no_match(self, mem: SessionMemory):
        mem.add("user", "Hello world")
        results = mem.search("quantum physics", threshold=0.8)
        assert results == []

    def test_search_ranks_by_relevance(self, mem: SessionMemory):
        mem.add("user", "Python is great")
        mem.add("user", "Install Python on Windows with pip")
        results = mem.search("install python pip")
        assert len(results) >= 1
        # The second entry should rank higher (more keyword hits)
        assert "install" in results[0].summary.lower() or "install" in results[0].full_content.lower()


# ------------------------------------------------------------------
# Pinned entries
# ------------------------------------------------------------------

class TestPinned:
    def test_get_pinned(self, mem: SessionMemory):
        mem.add("user", "Regular stuff")
        mem.add("user", "IMPORTANT: The API key is XYZ", pinned=True)
        mem.add("user", "Another message")
        pinned = mem.get_pinned()
        assert len(pinned) == 1
        assert "API key" in pinned[0].full_content


# ------------------------------------------------------------------
# Compression
# ------------------------------------------------------------------

class TestCompress:
    def test_compress_reduces_content(self, mem: SessionMemory):
        for i in range(10):
            mem.add("user", f"Message number {i} " * 50)
        count = mem.compress()
        assert count > 0
        # Compressed entries should have summary == full_content
        for entry in mem.all_entries()[:6]:
            if not entry.pinned:
                assert entry.full_content == entry.summary

    def test_compress_preserves_pinned(self, mem: SessionMemory):
        mem.add("user", "Regular " * 50)
        mem.add("user", "Critical data", pinned=True)
        for i in range(8):
            mem.add("user", f"Filler {i} " * 50)
        mem.compress(keep_pinned=True)
        pinned = mem.get_pinned()
        assert len(pinned) == 1
        assert pinned[0].full_content == "Critical data"


# ------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------

class TestConversion:
    def test_to_messages(self, mem: SessionMemory):
        mem.add("user", "Hello")
        mem.add("agent", "Hi")
        msgs = mem.to_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_clear(self, mem: SessionMemory):
        mem.add("user", "Hello")
        mem.clear()
        assert mem.size == 0
