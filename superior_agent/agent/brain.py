"""Brain — the stateful reasoning engine.

Agent loop architecture:
 1. Classify task complexity (one_shot structured JSON call)
 2. If trivial → stream text directly (no tools)
 3. If needs tools → non-streaming tool loop:
      a) Call LLM with tools (non-streaming, reliable parsing)
      b) If tool_calls in response → execute them, append results, loop
      c) If no tool_calls → done, emit the final text
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
import logging
import os
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from superior_agent.core.models import (
    EventType,
    LLMEvent,
    Message,
    Tier,
    TierDecision,
)

from .memory import MemoryEntry, SessionMemory

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Platform profile
# ------------------------------------------------------------------

@dataclass
class PlatformProfile:
    os_name: str
    path_sep: str
    shell: str
    list_cmd: str
    clear_cmd: str
    home_dir: str
    work_dir: str

    def to_dict(self) -> dict[str, str]:
        return {
            "os": self.os_name, "path_sep": self.path_sep,
            "shell": self.shell, "list_cmd": self.list_cmd,
            "clear_cmd": self.clear_cmd, "home_dir": self.home_dir,
            "work_dir": self.work_dir,
        }


def detect_platform(workdir: str | None = None) -> PlatformProfile:
    is_win = platform.system().lower() == "windows"
    return PlatformProfile(
        os_name="windows" if is_win else "linux",
        path_sep="\\" if is_win else "/",
        shell="powershell" if is_win else "bash",
        list_cmd="dir" if is_win else "ls",
        clear_cmd="cls" if is_win else "clear",
        home_dir=str(Path.home()),
        work_dir=workdir or os.getcwd(),
    )


# ------------------------------------------------------------------
# Agent state machine
# ------------------------------------------------------------------

class AgentState(enum.Enum):
    IDLE = "idle"
    CLASSIFYING = "classifying"
    CALLING_LLM = "calling_llm"
    TOOL_CALL = "tool_call"
    TOOL_EXEC = "tool_exec"
    STREAMING = "streaming"


# ------------------------------------------------------------------
# Tier configs
# ------------------------------------------------------------------

_TIER_CONFIGS: dict[Tier, dict[str, Any]] = {
    Tier.TRIVIAL: {"options": {"temperature": 0.7, "top_k": 20}, "enable_thinking": False},
    Tier.MODERATE: {"options": {"temperature": 0.6, "top_p": 0.95}, "enable_thinking": False},
    Tier.COMPLEX: {"options": {"temperature": 0.6, "top_p": 0.95}, "enable_thinking": False},
}

_TIER_SCHEMA = {
    "type": "object",
    "properties": {
        "tier": {"type": "string", "enum": ["trivial", "moderate", "complex"]},
        "rationale": {"type": "string"},
        "requires_tool": {"type": "boolean"},
    },
    "required": ["tier", "rationale", "requires_tool"],
}

_MAX_TOOL_ROUNDS = 5


# ------------------------------------------------------------------
# Brain
# ------------------------------------------------------------------

class Brain:

    def __init__(self, llm_bridge: Any, registry: Any, artifact_ctrl: Any, platform_profile: PlatformProfile) -> None:
        self.llm = llm_bridge
        self.registry = registry
        self.artifacts = artifact_ctrl
        self.platform = platform_profile
        self.memory = SessionMemory()
        self.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decide(self, user_input: str) -> AsyncIterator[LLMEvent]:
        self.memory.add("user", user_input)

        # Step 1: Classify
        self._set_state(AgentState.CLASSIFYING)
        yield LLMEvent(type=EventType.STATE_CHANGE, new_state="classifying",
                       content="Analysing task complexity…")

        tier = await self._classify_tier(user_input)

        yield LLMEvent(type=EventType.STATE_CHANGE, new_state="classified",
                       content=f"Tier={tier.tier.value}  requires_tool={tier.requires_tool}  ({tier.rationale})")

        # Step 2: Route
        if tier.tier == Tier.TRIVIAL and not tier.requires_tool:
            async for ev in self._handle_trivial(user_input):
                yield ev
        else:
            async for ev in self._handle_agentic(user_input, tier):
                yield ev

        self._set_state(AgentState.IDLE)
        yield LLMEvent(type=EventType.STATE_CHANGE, new_state="idle", content="Done")

    def reset(self) -> None:
        self.memory.clear()
        self.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Trivial handler — stream text, no tools
    # ------------------------------------------------------------------

    async def _handle_trivial(self, user_input: str) -> AsyncIterator[LLMEvent]:
        self._set_state(AgentState.STREAMING)
        yield LLMEvent(type=EventType.STATE_CHANGE, new_state="streaming",
                       content="Streaming response (no tools)…")

        messages = self._build_messages(user_input)
        full = ""
        async for ev in self.llm.stream_response(messages, options=_TIER_CONFIGS[Tier.TRIVIAL]["options"]):
            if ev.type == EventType.RESPONSE_CHUNK:
                full += ev.content
            yield ev

        self.memory.add("agent", full)

    # ------------------------------------------------------------------
    # Agentic handler — non-streaming tool loop + final streamed response
    # ------------------------------------------------------------------

    async def _handle_agentic(self, user_input: str, decision: TierDecision) -> AsyncIterator[LLMEvent]:
        cfg = _TIER_CONFIGS[decision.tier]
        messages = self._build_messages(user_input)
        tools_schemas = self._get_tool_schemas()

        for round_num in range(_MAX_TOOL_ROUNDS):
            # --- Call LLM (non-streaming, with tools) ---
            self._set_state(AgentState.CALLING_LLM)
            yield LLMEvent(
                type=EventType.STATE_CHANGE,
                new_state="calling_llm",
                content=f"Round {round_num+1}: calling LLM with {len(tools_schemas)} tools…",
            )

            chat_resp = await self.llm.chat_with_tools(
                messages, tools_schemas,
                enable_thinking=cfg["enable_thinking"],
                options=cfg["options"],
            )

            # Emit thinking if present
            if chat_resp.thinking:
                yield LLMEvent(type=EventType.THINKING_CHUNK, content=chat_resp.thinking)

            # --- No tool calls → we have the final response ---
            if not chat_resp.has_tool_calls:
                if chat_resp.content:
                    yield LLMEvent(type=EventType.RESPONSE_CHUNK, content=chat_resp.content)
                self.memory.add("agent", chat_resp.content)
                yield LLMEvent(type=EventType.DONE)
                return

            # --- Tool calls detected → execute them ---
            # Show any text the LLM produced alongside the tool calls
            if chat_resp.content:
                yield LLMEvent(type=EventType.RESPONSE_CHUNK, content=chat_resp.content)

            # Build assistant message WITH tool_calls for conversation history
            tool_calls_for_msg = []
            for tc in chat_resp.tool_calls:
                tool_calls_for_msg.append({
                    "function": {"name": tc.name, "arguments": tc.arguments}
                })
            messages.append(Message(
                role="assistant",
                content=chat_resp.content,
                tool_calls=tool_calls_for_msg,
            ))

            # Execute each tool
            for tc in chat_resp.tool_calls:
                # Announce the call
                self._set_state(AgentState.TOOL_CALL)
                yield LLMEvent(
                    type=EventType.TOOL_CALL,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                    content="",  # empty = call announcement
                )

                # Execute
                self._set_state(AgentState.TOOL_EXEC)
                yield LLMEvent(
                    type=EventType.STATE_CHANGE,
                    new_state="tool_exec",
                    content=f"Executing {tc.name}({', '.join(f'{k}={v!r}' for k,v in tc.arguments.items())})…",
                )

                t0 = time.time()
                result = await self._execute_tool_call(tc.name, tc.arguments)
                elapsed = time.time() - t0

                # Emit the result
                yield LLMEvent(
                    type=EventType.TOOL_CALL,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                    content=result,  # non-empty = result
                    tool_call_id=f"{elapsed:.2f}s",  # reuse field for timing
                )

                # Add tool result to messages
                messages.append(Message(role="tool", content=result, name=tc.name))

            # Loop → next round (LLM sees tool results)

        # Max rounds exhausted
        yield LLMEvent(
            type=EventType.RESPONSE_CHUNK,
            content="\n[Agent reached max tool rounds]",
        )
        yield LLMEvent(type=EventType.DONE)

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        return [meta.to_openai_schema() for meta in self.registry.list_all()]

    async def _execute_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            func, _ = self.registry.load(name)
        except KeyError:
            return f"Error: unknown tool '{name}'"

        kwargs = dict(arguments)
        sig = inspect.signature(func)
        if "workdir" in sig.parameters:
            kwargs["workdir"] = self.platform.work_dir
        if "platform_profile" in sig.parameters:
            kwargs["platform_profile"] = self.platform.to_dict()

        try:
            result = func(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"Tool error: {exc}"

    # ------------------------------------------------------------------
    # Complexity classification
    # ------------------------------------------------------------------

    async def _classify_tier(self, user_input: str) -> TierDecision:
        tool_names = ", ".join(t.name for t in self.registry.list_all()) or "none"
        messages = [
            Message(role="system", content=(
                "/no_think\n"
                "You are a task complexity classifier. Respond with JSON only.\n"
                "Decide the tier and whether tools are required.\n\n"
                "Rules:\n"
                "- 'trivial': pure knowledge/conversational (no action needed)\n"
                "- 'moderate': 1-3 steps, likely needs tools\n"
                "- 'complex': multi-step, definitely needs tools and planning\n"
                "- ANY request involving files, directories, or system commands → requires_tool=true\n"
                "- Only pure questions with no action → requires_tool=false\n\n"
                f"Available tools: {tool_names}"
            )),
            Message(role="user", content=user_input),
        ]
        try:
            data = await self.llm.one_shot(messages, format=_TIER_SCHEMA, options={"temperature": 0.3})
            return TierDecision(
                tier=Tier(data.get("tier", "moderate")),
                rationale=data.get("rationale", ""),
                requires_tool=data.get("requires_tool", True),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Classification failed: %s", exc)
            return TierDecision(tier=Tier.MODERATE, rationale=f"Failed: {exc}", requires_tool=True)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(self, user_input: str) -> list[Message]:
        msgs = [Message(role="system", content=self._system_prompt())]
        for entry in self.memory.get_recent(20):
            role = "user" if entry.role == "user" else "assistant"
            msgs.append(Message(role=role, content=entry.full_content))
        return msgs

    def _system_prompt(self) -> str:
        tool_list = self.registry.list_all()
        tool_desc = "\n".join(f"  - {t.name}: {t.description}" for t in tool_list) or "  (none)"

        return (
            "You are Superior Agent, an autonomous AI assistant.\n\n"
            "## CRITICAL RULES\n"
            "1. You are an AGENT. When asked to do something, DO IT with tools. "
            "Never ask for confirmation.\n"
            "2. After completing tool calls, write a brief summary of what you did.\n"
            "3. Do NOT call the same tool twice with identical arguments.\n\n"
            f"## Environment\n"
            f"- OS: {self.platform.os_name} | Shell: {self.platform.shell}\n"
            f"- Working directory: {self.platform.work_dir}\n\n"
            f"## Tools\n{tool_desc}\n\n"
            "All paths are relative to working directory unless absolute.\n"
        )

    def _set_state(self, state: AgentState) -> None:
        self.state = state
