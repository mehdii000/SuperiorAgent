"""Tests for core.llm_bridge — mock ollama client responses."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from superior_agent.core.llm_bridge import LLMBridge, ChatResponse, ToolCallResult
from superior_agent.core.models import EventType, Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


async def _fake_stream(*chunks):
    for chunk in chunks:
        yield chunk


# ------------------------------------------------------------------
# Token counting
# ------------------------------------------------------------------

class TestTokenCounting:
    def test_count_tokens_nonempty(self):
        assert LLMBridge.count_tokens("Hello, world!") > 0

    def test_count_tokens_empty(self):
        assert LLMBridge.count_tokens("") == 0

    def test_count_messages_tokens(self):
        bridge = LLMBridge(model="test")
        msgs = [_msg("user", "Hello"), _msg("assistant", "Hi there")]
        assert bridge.count_messages_tokens(msgs) > 0

    def test_context_stats(self):
        bridge = LLMBridge(model="test", max_tokens=1000)
        stats = bridge.get_context_stats([_msg("user", "Hello")])
        assert stats.used_tokens > 0
        assert stats.max_tokens == 1000
        assert 0 < stats.usage_pct < 100


# ------------------------------------------------------------------
# Streaming (text only, no tools)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_response_simple():
    bridge = LLMBridge(model="test")
    chunks = [
        {"message": {"content": "Hello "}, "done": False},
        {"message": {"content": "world!"}, "done": True},
    ]
    bridge._client = MagicMock()
    bridge._client.chat = AsyncMock(return_value=_fake_stream(*chunks))

    events = []
    async for ev in bridge.stream_response([_msg("user", "Hi")]):
        events.append(ev)

    responses = [e for e in events if e.type == EventType.RESPONSE_CHUNK]
    assert len(responses) == 2
    assert responses[0].content == "Hello "
    assert responses[1].content == "world!"
    assert any(e.type == EventType.DONE for e in events)


@pytest.mark.asyncio
async def test_stream_response_error():
    bridge = LLMBridge(model="test")
    bridge._client = MagicMock()
    bridge._client.chat = AsyncMock(side_effect=ConnectionError("refused"))

    events = []
    async for ev in bridge.stream_response([_msg("user", "hi")]):
        events.append(ev)

    assert any(e.type == EventType.ERROR for e in events)


# ------------------------------------------------------------------
# stream_chat_with_tools
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_chat_with_tools_text_response():
    bridge = LLMBridge(model="test")
    bridge._client = MagicMock()
    bridge._client.chat = AsyncMock(return_value=_fake_stream(
        {"message": {"content": "I created the file."}, "done": True}
    ))

    events = []
    async for ev in bridge.stream_chat_with_tools([_msg("user", "create file")], []):
        events.append(ev)

    responses = [e for e in events if e.type == EventType.RESPONSE_CHUNK]
    assert len(responses) == 1
    assert responses[0].content == "I created the file."
    assert not any(e.type == EventType.TOOL_CALL for e in events)


@pytest.mark.asyncio
async def test_stream_chat_with_tools_tool_call():
    bridge = LLMBridge(model="test")
    bridge._client = MagicMock()
    bridge._client.chat = AsyncMock(return_value=_fake_stream(
        {"message": {"tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "test.txt", "content": "hello"}}}]}, "done": True}
    ))

    events = []
    async for ev in bridge.stream_chat_with_tools([_msg("user", "create file")], [{"type": "function"}]):
        events.append(ev)

    tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
    assert len(tool_events) == 1
    assert tool_events[0].tool_name == "write_file"
    assert tool_events[0].tool_args["path"] == "test.txt"


# ------------------------------------------------------------------
# one_shot (structured JSON)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_shot():
    bridge = LLMBridge(model="test")
    result_json = json.dumps({"tier": "trivial", "rationale": "simple", "requires_tool": False})
    bridge._client = MagicMock()
    bridge._client.chat = AsyncMock(return_value={
        "message": {"content": result_json}, "done": True,
    })

    data = await bridge.one_shot([_msg("user", "classify")])
    assert data["tier"] == "trivial"
    assert data["requires_tool"] is False
