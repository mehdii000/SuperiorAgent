"""LLM Bridge — async client for Ollama's Qwen3 models.

Uses the official ``ollama`` Python library.

Two modes:
  • ``stream_response()`` — streaming for user-facing text (no tools)
  • ``chat_with_tools()`` — non-streaming for reliable tool-call detection
  • ``one_shot()`` — non-streaming for structured JSON (classifications etc.)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import ollama
import tiktoken

from .models import (
    ContextStats,
    EventType,
    LLMEvent,
    Message,
)

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MAX_TOKENS = 32_768
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class ToolCallResult:
    """Parsed tool call from a non-streaming ollama response."""
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Parsed non-streaming response — may contain text, tool calls, or both."""
    content: str = ""
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    thinking: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMBridge:
    """Async interface to Ollama via the official Python client."""

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        model: str = "qwen3:latest",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        headers = {"x-api-key": api_key} if api_key else None
        self._client = ollama.AsyncClient(host=host, headers=headers)

    # ------------------------------------------------------------------
    # Streaming (for user-facing text — NO tools)
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        messages: list[Message],
        *,
        enable_thinking: bool = False,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream a chat completion WITHOUT tools. Yields LLMEvent objects."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._msg_to_dict(m) for m in messages],
            "stream": True,
            "think": enable_thinking,
        }
        if options:
            kwargs["options"] = options

        try:
            response = await self._client.chat(**kwargs)
            async for part in response:
                message = part.get("message", {})

                thinking = message.get("thinking")
                if thinking:
                    yield LLMEvent(type=EventType.THINKING_CHUNK, content=thinking)

                content = message.get("content")
                if content:
                    yield LLMEvent(type=EventType.RESPONSE_CHUNK, content=content)

                if part.get("done", False):
                    yield LLMEvent(type=EventType.DONE)

        except Exception as exc:  # noqa: BLE001
            yield LLMEvent(type=EventType.ERROR, error=str(exc))

    # ------------------------------------------------------------------
    # Streaming with tools (prevents UI freezing during long thoughts)
    # ------------------------------------------------------------------

    async def stream_chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        enable_thinking: bool = False,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._msg_to_dict(m) for m in messages],
            "stream": True,
            "think": enable_thinking,
        }
        if tools:
            kwargs["tools"] = tools
        if options:
            kwargs["options"] = options

        try:
            response = await self._client.chat(**kwargs)
            async for part in response:
                message = part.get("message", {})

                thinking = message.get("thinking")
                if thinking:
                    yield LLMEvent(type=EventType.THINKING_CHUNK, content=thinking)

                content = message.get("content")
                if content:
                    yield LLMEvent(type=EventType.RESPONSE_CHUNK, content=content)

                # Tool calls might arrive in any chunk (often the last)
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        yield LLMEvent(
                            type=EventType.TOOL_CALL,
                            tool_name=fn.get("name", ""),
                            tool_args=fn.get("arguments", {}),
                        )

                if part.get("done", False):
                    yield LLMEvent(type=EventType.DONE)

        except Exception as exc:  # noqa: BLE001
            logger.error("stream_chat_with_tools failed: %s", exc)
            yield LLMEvent(type=EventType.ERROR, error=str(exc))

    # ------------------------------------------------------------------
    # One-shot structured JSON (classifications, decisions)
    # ------------------------------------------------------------------

    async def one_shot(
        self,
        messages: list[Message],
        *,
        format: dict[str, Any] | str | None = None,
        enable_thinking: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming call — returns the full response as parsed JSON."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._msg_to_dict(m) for m in messages],
            "stream": False,
            "think": enable_thinking,
        }
        if format is not None:
            kwargs["format"] = format
        if options:
            kwargs["options"] = options

        try:
            response = await self._client.chat(**kwargs)
            content = response.get("message", {}).get("content", "")
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse one-shot JSON: %s", exc)
            return {"error": str(exc), "raw": content if "content" in dir() else ""}
        except Exception as exc:  # noqa: BLE001
            logger.warning("One-shot call failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

    def count_messages_tokens(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            total += 4
            total += self.count_tokens(msg.content)
            if msg.tool_calls:
                total += self.count_tokens(json.dumps(msg.tool_calls))
        return total

    def get_context_stats(self, messages: list[Message]) -> ContextStats:
        used = self.count_messages_tokens(messages)
        return ContextStats(used_tokens=used, max_tokens=self.max_tokens)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_to_dict(msg: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name:
            d["name"] = msg.name
        return d

    async def close(self) -> None:
        pass
