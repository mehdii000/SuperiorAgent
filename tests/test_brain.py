"""Tests for agent.brain — state machine, routing, tool execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from superior_agent.core.models import EventType, LLMEvent, Message
from superior_agent.core.llm_bridge import ChatResponse, ToolCallResult
from superior_agent.agent.brain import Brain, PlatformProfile, detect_platform, AgentState
from superior_agent.agent.registry import Registry
from superior_agent.agent.artifact_controller import ArtifactController


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def pprofile() -> PlatformProfile:
    return PlatformProfile(
        os_name="windows", path_sep="\\", shell="powershell",
        list_cmd="dir", clear_cmd="cls",
        home_dir="C:\\Users\\Test", work_dir="C:\\Dev\\TestProject",
    )


@pytest.fixture
def artifact_ctrl(tmp_path: Path) -> ArtifactController:
    ctrl = ArtifactController("test-brain", root=tmp_path)
    yield ctrl
    ctrl.close()


class MockLLM:
    """Fake LLM bridge."""
    def __init__(self, tier_result=None, chat_result=None, stream_events=None):
        self.model = "test-model"
        self._tier = tier_result or {"tier": "trivial", "rationale": "simple", "requires_tool": False}
        self._chat = chat_result or ChatResponse(content="Done.")
        self._stream = stream_events or [
            LLMEvent(type=EventType.RESPONSE_CHUNK, content="Hello!"),
            LLMEvent(type=EventType.DONE),
        ]

    async def one_shot(self, messages, **kw) -> dict:
        return self._tier

    async def chat_with_tools(self, messages, tools, **kw) -> ChatResponse:
        return self._chat

    async def stream_response(self, messages, **kw) -> AsyncIterator[LLMEvent]:
        for ev in self._stream:
            yield ev


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestPlatform:
    def test_detect(self):
        p = detect_platform("/tmp/test")
        assert p.work_dir == "/tmp/test"
        assert p.os_name in ("windows", "linux")

    def test_to_dict(self, pprofile):
        d = pprofile.to_dict()
        assert d["os"] == "windows"


class TestBrainTrivial:
    @pytest.mark.asyncio
    async def test_trivial_emits_response(self, pprofile, artifact_ctrl):
        brain = Brain(MockLLM(), Registry(), artifact_ctrl, pprofile)
        events = [ev async for ev in brain.decide("What is 2+2?")]

        responses = [e for e in events if e.type == EventType.RESPONSE_CHUNK]
        assert any("Hello" in r.content for r in responses)

    @pytest.mark.asyncio
    async def test_trivial_records_memory(self, pprofile, artifact_ctrl):
        brain = Brain(MockLLM(), Registry(), artifact_ctrl, pprofile)
        async for _ in brain.decide("Hi"):
            pass
        assert brain.memory.size >= 2


class TestBrainAgentic:
    @pytest.mark.asyncio
    async def test_agentic_path_with_tool(self, pprofile, artifact_ctrl):
        """When LLM returns tool calls, should enter agentic path."""
        # First chat returns a tool call, second returns text
        call_count = 0

        class MockLLMWithTool:
            model = "test"
            async def one_shot(self, messages, **kw):
                return {"tier": "moderate", "rationale": "needs tool", "requires_tool": True}

            async def chat_with_tools(self, messages, tools, **kw):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return ChatResponse(
                        content="",
                        tool_calls=[ToolCallResult(name="write_file", arguments={"path": "a.txt", "content": "hi"})]
                    )
                return ChatResponse(content="Created a.txt.")

        brain = Brain(MockLLMWithTool(), Registry(), artifact_ctrl, pprofile)
        events = [ev async for ev in brain.decide("Create a file")]

        # Should have tool call events
        tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
        assert len(tool_events) >= 1

        # Should have a response
        responses = [e for e in events if e.type == EventType.RESPONSE_CHUNK]
        assert any("Created" in r.content for r in responses)


class TestBrainReset:
    def test_reset(self, pprofile, artifact_ctrl):
        brain = Brain(MockLLM(), Registry(), artifact_ctrl, pprofile)
        brain.memory.add("user", "test")
        brain.reset()
        assert brain.memory.size == 0
        assert brain.state == AgentState.IDLE
