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
from superior_agent.core.templates import TEMPLATES

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
    Tier.MODERATE: {"options": {"temperature": 0.5, "top_p": 0.95, "num_ctx": 16384}, "enable_thinking": True},
    Tier.COMPLEX: {"options": {"temperature": 0.5, "top_p": 0.95, "num_ctx": 16384}, "enable_thinking": True},
}

_MAX_TOOL_ROUNDS = 30


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
        self.max_tool_rounds = _MAX_TOOL_ROUNDS
        self.processes: dict[int, dict[str, Any]] = {} # pid -> {command, process}
        self.current_template = "General"
        self.active_tools: set[str] = set(TEMPLATES[self.current_template].initial_tools)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decide(self, user_input: str) -> AsyncIterator[LLMEvent]:
        self.memory.add("user", user_input)

        # Skip classification and always go agentic (Always-Agentic Approach)
        async for ev in self._handle_agentic(user_input):
            yield ev

        self._set_state(AgentState.IDLE)
        yield LLMEvent(type=EventType.STATE_CHANGE, new_state="idle", content="Done")

    def reset(self) -> None:
        self.memory.clear()
        self.state = AgentState.IDLE
        self.active_tools = set(TEMPLATES[self.current_template].initial_tools)

    def get_context_stats(self) -> dict[str, int]:
        """Return token usage for the current conversation state."""
        messages = self._build_messages("")
        stats = self.llm.get_context_stats(messages)
        return {"used": stats.used_tokens, "max": stats.max_tokens}

    def switch_template(self, name: str) -> bool:
        """Switch to a different agent template."""
        if name not in TEMPLATES:
            return False
        self.current_template = name
        # Reset tools to template defaults
        self.active_tools = set(TEMPLATES[name].initial_tools)
        return True

    async def cleanup(self) -> None:
        """Terminate all background processes."""
        if not self.processes:
            return
        
        logger.info("Cleaning up %d background process(es)...", len(self.processes))
        for pid, info in list(self.processes.items()):
            proc = info.get("process")
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    # Wait briefly for termination
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                except Exception as e:
                    logger.warning("Error terminating process %d: %s", pid, e)
        self.processes.clear()

    # ------------------------------------------------------------------
    # Agentic handler — non-streaming tool loop + final streamed response
    # ------------------------------------------------------------------

    async def _handle_agentic(self, user_input: str, decision: TierDecision | None = None) -> AsyncIterator[LLMEvent]:
        # Default to MODERATE if not specified
        tier = decision.tier if decision else Tier.MODERATE
        cfg = _TIER_CONFIGS[tier]
        messages = self._build_messages(user_input)
        
        called_tools: set[str] = set() # Track unique (name, args_json) per turn
        round_num = 0
        while round_num < self.max_tool_rounds:
            tools_schemas = self._get_tool_schemas()
            # --- Call LLM (streaming, with tools) ---
            self._set_state(AgentState.CALLING_LLM)
            yield LLMEvent(
                type=EventType.STATE_CHANGE,    
                new_state="calling_llm",
                content=f"Round {round_num+1}/{self.max_tool_rounds}: calling LLM with {len(tools_schemas)} tools…",
            )

            response_text = ""
            tool_calls = []

            async for ev in self.llm.stream_chat_with_tools(
                messages, tools_schemas,
                enable_thinking=cfg["enable_thinking"],
                options=cfg["options"],
            ):
                if ev.type == EventType.RESPONSE_CHUNK:
                    response_text += ev.content
                    yield ev
                elif ev.type == EventType.THINKING_CHUNK:
                    yield ev
                elif ev.type == EventType.TOOL_CALL:
                    tool_calls.append(ev)
                elif ev.type == EventType.ERROR:
                    yield ev
                    return # TERMINATE the loop on LLM errors

            # --- No tool calls → we have the final response ---
            if not tool_calls:
                self.memory.add("agent", response_text)
                yield LLMEvent(type=EventType.DONE)
                return

            # --- Tool calls detected → execute them ---
            # Build assistant message WITH tool_calls for conversation history
            tool_calls_for_msg = []
            for tc in tool_calls:
                tool_calls_for_msg.append({
                    "function": {"name": tc.tool_name, "arguments": tc.tool_args}
                })
            messages.append(Message(
                role="assistant",
                content=response_text,
                tool_calls=tool_calls_for_msg,
            ))

            # Execute each tool
            executed_any = False
            for tc in tool_calls:
                # Deduplication check
                tc_key = f"{tc.tool_name}:{json.dumps(tc.tool_args, sort_keys=True)}"
                if tc_key in called_tools:
                    logger.warning("Agent is repeating tool call: %s. Skipping to prevent loop.", tc_key)
                    continue
                called_tools.add(tc_key)
                executed_any = True

                # Announce the call
                self._set_state(AgentState.TOOL_CALL)
                yield LLMEvent(
                    type=EventType.TOOL_CALL,
                    tool_name=tc.tool_name,
                    tool_args=tc.tool_args,
                    content="",  # empty = call announcement
                )

                # Execute
                self._set_state(AgentState.TOOL_EXEC)
                yield LLMEvent(
                    type=EventType.STATE_CHANGE,
                    new_state="tool_exec",
                    content=f"Executing {tc.tool_name}({', '.join(f'{k}={v!r}' for k,v in tc.tool_args.items())})…",
                )

                t0 = time.time()
                result = await self._execute_tool_call(tc.tool_name, tc.tool_args)
                elapsed = time.time() - t0

                # Emit the result
                yield LLMEvent(
                    type=EventType.TOOL_CALL,
                    tool_name=tc.tool_name,
                    tool_args=tc.tool_args,
                    content=result,  # non-empty = result
                    tool_call_id=f"{elapsed:.2f}s",  # reuse field for timing
                )

                # Add tool result to messages
                messages.append(Message(role="tool", content=result, name=tc.tool_name))

            # If the model ONLY emitted redundant tools, stop the loop
            if not executed_any:
                logger.warning("No new tool calls executed this round. Stopping turn.")
                break

            round_num += 1
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

    def _get_active_tool_metadata(self):
        return [t for t in self.registry.list_all() if t.name in self.active_tools]

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        return [meta.to_openai_schema() for meta in self._get_active_tool_metadata()]

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
        if "registry" in sig.parameters:
            kwargs["registry"] = self.registry
        if "artifacts" in sig.parameters:
            kwargs["artifacts"] = self.artifacts
        if "brain" in sig.parameters:
            kwargs["brain"] = self

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = await asyncio.to_thread(func, **kwargs)
                
            # Intercept search_tools to auto-activate the discovered tools
            if name == "search_tools":
                found = self.registry.search(arguments.get("query", ""))
                for t in found:
                    self.active_tools.add(t.name)
                    
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"Tool error: {exc}"

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
        tpl = TEMPLATES[self.current_template]
        active_list = self._get_active_tool_metadata()
        active_desc = "\n".join(f"  - {t.name}: {t.description}" for t in active_list) or "  (none)"

        # Full catalog (lightweight)
        all_tools = self.registry.list_all()
        catalog_desc = "\n".join(f"  - {t.name}: {t.description}" for t in all_tools) or "  (none)"

        return (
            f"{tpl.system_prompt_prefix}\n\n"
            "## CRITICAL RULES\n"
            "1. You are an AGENT. When asked to do something, DO IT with tools. "
            "Never ask for confirmation.\n"
            "2. **HYBRID TOOL LOADING**: You see a list of ALL available tools in the 'Complete Tool Catalog' below. "
            "However, only those in the 'Currently ACTIVE Tool Schemas' are ready for direct calling. "
            "If you need a tool from the Catalog that is NOT in the Active list, you MUST call `search_tools(query='tool_name')` first to load its full parameters into your next turn.\n"
            "3. After completing tool calls, write a brief summary of what you did.\n"
            "4. Do NOT call the same tool twice with identical arguments.\n\n"
            f"## Environment\n"
            f"- OS: {self.platform.os_name} | Shell: {self.platform.shell}\n"
            f"- Working directory: {self.platform.work_dir}\n\n"
            f"## Currently ACTIVE Tool Schemas (Ready to call)\n{active_desc}\n\n"
            f"## Complete Tool Catalog (Awareness only)\n{catalog_desc}\n\n"
            "All paths are relative to working directory unless absolute.\n"
        )

    def _set_state(self, state: AgentState) -> None:
        self.state = state
