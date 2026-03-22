"""CLI Frontend — developer-friendly terminal UI.

Shows every step the agent takes:
  • Classification result and rationale
  • Each LLM call with round number
  • Tool calls with full arguments
  • Tool results with execution time
  • Final response text
  • Turn summary with timing and tool list
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.syntax import Syntax
from rich import box

from .themes import AGENT_THEME
from superior_agent.core.models import EventType, LLMEvent

logger = logging.getLogger(__name__)

_COMMANDS = {
    "/reset":     "Reset session memory and start fresh",
    "/tools":     "List all discovered tools with schemas",
    "/memory":    "Show session memory entries",
    "/artifacts": "Display current artifacts",
    "/exit":      "Exit the agent",
    "/help":      "Show this help message",
}

# Compact state labels for the log
_STATE_ICONS = {
    "classifying": ("🔍", "CLASSIFY"),
    "classified":  ("📊", "RESULT  "),
    "streaming":   ("📡", "STREAM  "),
    "calling_llm": ("🤖", "LLM CALL"),
    "tool_call":   ("🔧", "TOOL    "),
    "tool_exec":   ("⚡", "EXEC    "),
    "idle":        ("✅", "IDLE    "),
}


def _ts() -> str:
    """Compact timestamp for log lines."""
    return time.strftime("%H:%M:%S")


class CLI:

    def __init__(self, brain: Any, artifact_ctrl: Any) -> None:
        self.brain = brain
        self.artifacts = artifact_ctrl
        self.console = Console(theme=AGENT_THEME)
        self._cancelled = False
        self._tool_history: list[dict[str, Any]] = []
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._render_banner()
        self._setup_signal_handlers()

        while True:
            try:
                user_input = await self._get_input()
            except (EOFError, KeyboardInterrupt):
                self._goodbye()
                break

            if not user_input.strip():
                continue

            if user_input.strip().startswith("/"):
                if await self._handle_command(user_input.strip()):
                    break
                continue

            self._cancelled = False
            self._tool_history = []
            self._turn_count += 1
            await self._process(user_input)

    async def _get_input(self) -> str:
        self.console.print()
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.console.input("[prompt]❯ [/prompt]")
        )

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def _process(self, user_input: str) -> None:
        self.console.print()
        self.console.print(Rule(f"[dim]Turn {self._turn_count}[/dim]", style="dim"))

        start = time.time()
        response_text = ""
        response_started = False

        try:
            async for ev in self.brain.decide(user_input):
                if self._cancelled:
                    self.console.print("\n[dim]⚠ Cancelled.[/dim]")
                    break

                if ev.type == EventType.STATE_CHANGE:
                    self._log_state(ev)

                elif ev.type == EventType.THINKING_CHUNK:
                    self.console.print(
                        Text(ev.content, style="thinking"), end=""
                    )

                elif ev.type == EventType.RESPONSE_CHUNK:
                    if not response_started:
                        self.console.print()
                        response_started = True
                    response_text += ev.content
                    self.console.print(
                        Text(ev.content, style="response"), end=""
                    )

                elif ev.type == EventType.TOOL_CALL:
                    self._render_tool(ev)

                elif ev.type == EventType.ERROR:
                    self.console.print(f"\n[error]✗ {ev.error}[/error]")

                elif ev.type == EventType.DONE:
                    pass

        except Exception as exc:
            self.console.print(f"\n[error]✗ Exception: {exc}[/error]")

        elapsed = time.time() - start
        self.console.print()
        self._render_summary(elapsed)

    # ------------------------------------------------------------------
    # State log line
    # ------------------------------------------------------------------

    def _log_state(self, ev: LLMEvent) -> None:
        state = ev.new_state or ""
        icon, label = _STATE_ICONS.get(state, ("🔄", state.upper().ljust(8)))
        detail = ev.content or ""

        if state == "idle":
            return  # skip idle — the summary covers it

        self.console.print(
            f"  [dim]{_ts()}[/dim]  {icon} [accent]{label}[/accent]  [dim]{detail}[/dim]"
        )

    # ------------------------------------------------------------------
    # Tool call / result rendering
    # ------------------------------------------------------------------

    def _render_tool(self, ev: LLMEvent) -> None:
        has_result = bool(ev.content)

        if not has_result:
            # Tool call announcement (before execution)
            args_str = "  ".join(f"[dim]{k}=[/dim]{v!r}" for k, v in ev.tool_args.items())
            self.console.print(
                f"  [dim]{_ts()}[/dim]  🔧 [tool.name]{ev.tool_name}[/tool.name]  {args_str}"
            )
        else:
            # Tool result (after execution)
            elapsed_str = ev.tool_call_id or ""  # we store timing here
            result = ev.content
            truncated = len(result) > 600
            display_result = result[:600] + ("…" if truncated else "")

            self._tool_history.append({
                "name": ev.tool_name,
                "args": ev.tool_args,
                "result_preview": result[:200],
                "elapsed": elapsed_str,
            })

            # Build the panel
            lines: list[str] = []
            for k, v in ev.tool_args.items():
                lines.append(f"  [dim]{k}:[/dim] {v}")
            lines.append("")
            lines.append(f"  [tool.result]→ Result ({elapsed_str}):[/tool.result]")
            for line in display_result.split("\n"):
                lines.append(f"    {line}")

            self.console.print(Panel(
                "\n".join(lines),
                title=f"[tool.name]⚡ {ev.tool_name}[/tool.name]",
                border_style="accent",
                padding=(0, 1),
            ))

    # ------------------------------------------------------------------
    # Turn summary
    # ------------------------------------------------------------------

    def _render_summary(self, elapsed: float) -> None:
        parts = [f"[dim]⏱ {elapsed:.1f}s[/dim]"]

        if self._tool_history:
            names = [t["name"] for t in self._tool_history]
            parts.append(f"[accent]🔧 {len(names)} tool(s):[/accent] [dim]{', '.join(names)}[/dim]")
        else:
            parts.append("[dim]no tools used[/dim]")

        self.console.print(Rule(style="dim"))
        self.console.print("  " + "  │  ".join(parts))

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _render_banner(self) -> None:
        pp = self.brain.platform
        self.console.print()
        self.console.print(Panel(
            f"[banner]S U P E R I O R   A G E N T[/banner]",
            border_style="accent",
            padding=(0, 4),
        ))

        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column("K", style="dim", width=12)
        t.add_column("V", style="accent")
        t.add_row("  OS", pp.os_name)
        t.add_row("  Shell", pp.shell)
        t.add_row("  Model", self.brain.llm.model)
        t.add_row("  Workdir", pp.work_dir)

        tools = self.brain.registry.list_all()
        t.add_row("  Tools", ", ".join(tool.name for tool in tools) if tools else "(none)")
        self.console.print(t)
        self.console.print()
        self.console.print("[dim]Type /help for commands. Agent acts autonomously — no confirmations.[/dim]")
        self.console.print(Rule(style="dim"))

    def _goodbye(self) -> None:
        self.console.print("\n[banner]👋 Goodbye![/banner]")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def _handle_command(self, cmd: str) -> bool:
        c = cmd.lower().split()[0]

        if c == "/exit":
            self._goodbye()
            return True

        elif c == "/help":
            self.console.print()
            for name, desc in _COMMANDS.items():
                self.console.print(f"  [accent]{name:<14}[/accent] [dim]{desc}[/dim]")

        elif c == "/reset":
            self.brain.reset()
            self.console.print("[accent]✓ Session reset.[/accent]")

        elif c == "/tools":
            tools = self.brain.registry.list_all()
            if not tools:
                self.console.print("[dim]No tools discovered.[/dim]")
            else:
                t = Table(title="Registered Tools", border_style="accent", box=box.SIMPLE)
                t.add_column("Name", style="tool.name")
                t.add_column("Description")
                t.add_column("Parameters", style="dim")
                for tool in tools:
                    params = ", ".join(p["name"] for p in tool.parameters) if tool.parameters else "-"
                    t.add_row(tool.name, tool.description, params)
                self.console.print(t)

        elif c == "/memory":
            entries = self.brain.memory.get_recent(15)
            if not entries:
                self.console.print("[dim]Empty memory.[/dim]")
            else:
                t = Table(title="Session Memory", border_style="accent", box=box.SIMPLE)
                t.add_column("#", style="dim", width=4)
                t.add_column("Role", width=6)
                t.add_column("📌", width=2)
                t.add_column("Summary", max_width=80)
                for e in entries:
                    pin = "📌" if e.pinned else ""
                    t.add_row(str(e.turn_id), e.role, pin, e.summary[:120])
                self.console.print(t)

        elif c == "/artifacts":
            names = self.artifacts.list_all()
            if not names:
                self.console.print("[dim]No artifacts.[/dim]")
            else:
                for name in names:
                    content = self.artifacts.get(name) or ""
                    self.console.print(Panel(
                        Markdown(content), title=f"📄 {name}",
                        border_style="accent", padding=(0, 1),
                    ))
        else:
            self.console.print(f"[dim]Unknown: {cmd}. Type /help.[/dim]")

        return False

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT, self._cancel)

    def _cancel(self) -> None:
        self._cancelled = True
