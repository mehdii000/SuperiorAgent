"""CLI Frontend — developer-friendly terminal UI using Textual.

Provides a full-screen layout with:
  • Chat history (scrollable)
  • Bottom command input
  • Right sidebar showing the plan/tasks
  • Real-time streaming of thinking, response, and tools
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Input, Static, Markdown as TMarkdown
from textual.suggester import SuggestFromList
from textual import work

from superior_agent.core.models import EventType, LLMEvent
from superior_agent.core.templates import TEMPLATES

logger = logging.getLogger(__name__)

# Compact state labels for the log
_STATE_ICONS = {
    "classifying": ("🔍", "CLASSIFYING"),
    "classified":  ("📊", "RESULT     "),
    "streaming":   ("📡", "STREAMING  "),
    "calling_llm": ("🤖", "LLM CALL   "),
    "tool_call":   ("🔧", "TOOL CALL  "),
    "tool_exec":   ("⚡", "EXECUTING  "),
    "idle":        ("✅", "IDLE       "),
}

_TIPS = [
    "The agent can search for more tools using [bold cyan]search_tools[/bold cyan].",
    "Switch templates with [bold green]/template <name>[/bold green] (Coding, Research, etc).",
    "Session memory is preserved across turns for context.",
    "High-complexity tasks use tiered reasoning automatically.",
    "Check [bold yellow]README.md[/bold yellow] for tool development guides.",
    "Background processes are tracked in the sidebar.",
    "All tools are lazy-loaded only when actually needed.",
]

def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes and control characters."""
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', text)
    # Also strip some common control characters that might break Textual
    return "".join(ch for ch in text if ch == '\n' or (ord(ch) >= 32 and ord(ch) != 127))


class ChatMessage(Static):
    """A single message block in the chat history."""
    
    def __init__(self, renderable: RenderableType, classes: str = "") -> None:
        super().__init__(renderable, classes=classes)
        self._content = renderable

    def append_text(self, new_text: str|RenderableType) -> None:
        """Allow appending text if the renderable is a string or Text."""
        if isinstance(self._content, str):
            self._content += str(new_text)
        elif isinstance(self._content, Text):
            self._content.append(new_text)
        self.update(self._content)


class AgentApp(App):
    """The main Textual application for Superior Agent."""
    
    TITLE = "Superior Agent"
    CSS = """
    #main_container {
        height: 100%;
        width: 100%;
    }
    
    #chat_area {
        width: 2fr;
        height: 100%;
        border-right: solid $primary-background-lighten-2;
    }
    
    #chat_history {
        height: 1fr;
        padding: 1;
    }
    
    #input_area {
        height: 3;
        dock: bottom;
    }
    
    #status_bar {
        height: 1;
        dock: bottom;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
    }
    
    #sidebar {
        width: 0.6fr;
        height: 100%;
        padding: 1;
        background: $panel;
    }
    
    .msg_user {
        background: $primary-background-lighten-1;
        color: $text;
        padding: 1 2;
        margin: 1 0;
        border-left: thick $accent;
    }
    
    .msg_agent_thinking {
        color: $text-muted;
        text-style: italic;
        padding: 0 2;
        margin: 0;
    }
    
    .msg_agent_response {
        color: $text;
        padding: 1 2;
        margin: 0;
    }
    
    .msg_tool {
        margin: 1 2;
    }
    
    .msg_state {
        color: $text-muted;
        text-style: italic;
        padding: 0 2;
        margin: 0;
    }
    
    .msg_error {
        color: $error;
        padding: 1 2;
        margin: 1 0;
    }
    
    .sidebar_header {
        color: $accent;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }
    
    .sidebar_sep {
        color: $primary-background-lighten-2;
        margin: 1 0;
    }

    #process_list {
        color: $text-muted;
        padding-left: 1;
        margin-top: 0;
    }

    .info_row {
        padding-left: 1;
        height: 1;
    }

    .info_label {
        color: dimgrey;
        width: 10;
    }

    .info_value {
        color: cyan;
        text-style: bold;
    }

    #tips_section {
        background: $boost;
        color: $text-muted;
        padding: 1;
        margin-top: 1;
        border: dashed $primary-background-lighten-2;
    }
    
    .tip_header {
        color: $accent;
        text-style: bold italic;
        margin-bottom: 1;
    }

    .tip_text {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, brain: Any, artifact_ctrl: Any) -> None:
        super().__init__()
        self.brain = brain
        self.artifacts = artifact_ctrl
        self._current_thinking: ChatMessage | None = None
        self._current_response: ChatMessage | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal(id="main_container"):
            # Main Chat Area
            with Vertical(id="chat_area"):
                yield VerticalScroll(id="chat_history")
                yield Static("✅ IDLE", id="status_bar")
                commands = ["/exit", "/help", "/reset", "/tools", "/memory", "/artifacts", "/template", "/debug"]
                yield Input(
                    placeholder="Ask Superior Agent... (type /help for commands)", 
                    id="chat_input",
                    suggester=SuggestFromList(commands, case_sensitive=False)
                )
                
            # Sidebar for Technical Information
            with Vertical(id="sidebar"):
                with VerticalScroll(id="sidebar_scroll"):
                    yield Static("AGENT INFO", classes="sidebar_header")
                    with Vertical(classes="info_section"):
                        with Horizontal(classes="info_row"):
                            yield Static("Model:", classes="info_label")
                            yield Static("-", id="info_model", classes="info_value")
                        with Horizontal(classes="info_row"):
                            yield Static("Mode:", classes="info_label")
                            yield Static("-", id="info_mode", classes="info_value")
                        with Horizontal(classes="info_row"):
                            yield Static("Session:", classes="info_label")
                            yield Static("-", id="info_session", classes="info_value")
                        with Horizontal(classes="info_row"):
                            yield Static("Context:", classes="info_label")
                            yield Static("-", id="info_context", classes="info_value")
                    
                    yield Static("\nRESOURCES", classes="sidebar_header")
                    with Vertical(classes="info_section"):
                        with Horizontal(classes="info_row"):
                            yield Static("Memory:", classes="info_label")
                            yield Static("-", id="info_memory", classes="info_value")
                        with Horizontal(classes="info_row"):
                            yield Static("Tools:", classes="info_label")
                            yield Static("-", id="info_tools", classes="info_value")
                        with Horizontal(classes="info_row"):
                            yield Static("Artifacts:", classes="info_label")
                            yield Static("-", id="info_artifacts", classes="info_value")

                    yield Static("\nENVIRONMENT", classes="sidebar_header")
                    with Vertical(classes="info_section"):
                        with Horizontal(classes="info_row"):
                            yield Static("OS:", classes="info_label")
                            yield Static("-", id="info_os", classes="info_value")
                        with Horizontal(classes="info_row"):
                            yield Static("Shell:", classes="info_label")
                            yield Static("-", id="info_shell", classes="info_value")
                        with Horizontal(classes="info_row"):
                            yield Static("Workdir:", classes="info_label")
                            yield Static("-", id="info_workdir", classes="info_value")

                    yield Static("\nBACKGROUND PROCESSES", classes="sidebar_header")
                    yield Static("_None_", id="process_list")

                    yield Vertical(
                        Static("💡 PRO TIP", classes="tip_header"),
                        Static("Use /tools to see what I can do!", id="tip_text", classes="tip_text"),
                        id="tips_section"
                    )
                
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat_input").focus()
        self._append_message("banner", self._get_banner_panel())
        self._update_sidebar(new_tip=True)

    async def on_unmount(self) -> None:
        """Cleanup before exiting."""
        await self.brain.cleanup()

    def _get_banner_panel(self) -> Panel:
        pp = self.brain.platform
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column("K", style="dim", width=12)
        t.add_column("V", style="bold cyan")
        t.add_row("  OS", pp.os_name)
        t.add_row("  Shell", pp.shell)
        t.add_row("  Mode", self.brain.current_template, style="bold yellow")
        t.add_row("  Model", self.brain.llm.model)
        t.add_row("  Workdir", pp.work_dir)

        tools = self.brain.registry.list_all()
        t.add_row("  Tools", ", ".join(tool.name for tool in tools) if tools else "(none)")
        
        return Panel(
            t, 
            title="S U P E R I O R   A G E N T", 
            border_style="cyan",
            padding=(1, 2)
        )

    def _update_sidebar(self, new_tip: bool = False) -> None:
        # 1. Agent Info
        self.query_one("#info_model", Static).update(self.brain.llm.model)
        self.query_one("#info_mode", Static).update(self.brain.current_template)
        self.query_one("#info_session", Static).update(self.artifacts.session_id)
        
        try:
            stats = self.brain.get_context_stats()
            self.query_one("#info_context", Static).update(f"{stats['used']} / {stats['max']}")
        except Exception:
            self.query_one("#info_context", Static).update("Unknown")

        # 2. Resources
        self.query_one("#info_memory", Static).update(f"{self.brain.memory.size} turns")
        
        active_cnt = len(self.brain.active_tools)
        total_cnt = len(self.brain.registry.list_all())
        self.query_one("#info_tools", Static).update(f"{active_cnt} active / {total_cnt} total")
        
        art_cnt = len(self.artifacts.list_all())
        self.query_one("#info_artifacts", Static).update(f"{art_cnt} found")

        # 3. Environment
        pp = self.brain.platform
        self.query_one("#info_os", Static).update(pp.os_name)
        self.query_one("#info_shell", Static).update(pp.shell)
        
        # Truncate workdir if too long
        wd = pp.work_dir
        if len(wd) > 25:
            wd = "..." + wd[-22:]
        self.query_one("#info_workdir", Static).update(wd)

        # 4. Background Processes
        proc_list = self.query_one("#process_list", Static)
        if not self.brain.processes:
            proc_list.update("_None_")
        else:
            lines = []
            for pid, info in self.brain.processes.items():
                cmd = info.get("command", "unknown")
                # Shorten command if too long
                display_cmd = cmd[:20] + "..." if len(cmd) > 20 else cmd
                lines.append(f"• [bold cyan]{pid}[/bold cyan]: {display_cmd}")
            proc_list.update("\n".join(lines))

        # 5. Tips
        if new_tip:
            import random
            tip = random.choice(_TIPS)
            self.query_one("#tip_text", Static).update(tip)

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        text = message.value.strip()
        if not text:
            return
            
        inp = self.query_one("#chat_input", Input)
        inp.value = ""
        
        # Handle slash commands locally
        if text.startswith("/"):
            await self._handle_command(text)
            return

        # Show user message
        if text.startswith("/"):
            parts = text.split()
            cmd_text = Text()
            cmd_text.append(parts[0], style="bold green")
            if len(parts) > 1:
                cmd_text.append(" " + " ".join(parts[1:]), style="cyan")
            self._append_message("user", cmd_text)
        else:
            self._append_message("user", Text(text))
        
        # Lock input and process
        inp.disabled = True
        self._current_thinking = None
        self._current_response = None
        
        self.run_agent_loop(text)

    @work
    async def run_agent_loop(self, user_input: str) -> None:
        try:
            async for ev in self.brain.decide(user_input):
                self._dispatch_event(ev)
        except Exception as exc:
            self._append_message("error", Text(f"Exception: {exc}"))
        finally:
            self._finish_loop()

    def _finish_loop(self) -> None:
        self.query_one("#chat_input").disabled = False
        self.query_one("#chat_input").focus()
        self._update_sidebar(new_tip=True)

    def _dispatch_event(self, ev: LLMEvent) -> None:
        # We are on the async main UI thread so we can just update directly
        self._handle_event_ui(ev)

    def _handle_event_ui(self, ev: LLMEvent) -> None:
        if ev.type == EventType.STATE_CHANGE:
            state_str = ev.new_state or "idle"
            icon, label = _STATE_ICONS.get(state_str, ("🔄", state_str.upper()))
            detail = f" - {ev.content}" if ev.content else ""
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(f"{icon} [bold]{label}[/bold]{detail}")
            
            # Reset chunk buffers if state changed heavily
            if state_str in ("tool_call", "calling_llm", "classifying"):
                self._current_thinking = None
                self._current_response = None
                
            # Append fluid step logs to chat like Claude Code
            if state_str in ("classifying", "classified", "calling_llm"):
                self._append_message("state", Text(f"{icon} {ev.content}", style="dim italic"))

        elif ev.type == EventType.THINKING_CHUNK:
            if not self._current_thinking:
                self._append_message("state", Text("🤔 Thinking...", style="dim italic"))
                self._current_thinking = self._append_message("agent_thinking", Text(style="dim italic"))
            self._current_thinking.append_text(ev.content)

        elif ev.type == EventType.RESPONSE_CHUNK:
            if not self._current_response:
                self._current_response = self._append_message("agent_response", Text())
            self._current_response.append_text(ev.content)

        elif ev.type == EventType.TOOL_CALL:
            has_result = bool(ev.content)
            
            if not has_result:
                # the call
                args_str = ", ".join(f"{k}={v!r}" for k, v in ev.tool_args.items())
                text = f"🔧 Calling **{ev.tool_name}**({args_str})"
                self._append_message("tool", Markdown(text))
            else:
                # the result
                elapsed_str = ev.tool_call_id or ""
                result = strip_ansi(ev.content)
                if len(result) > 1000:
                    result = result[:1000] + "\n...(truncated)"
                    
                panel = Panel(
                    result,
                    title=f"⚡ {ev.tool_name} Result ({elapsed_str})",
                    border_style="yellow",
                    padding=(0, 1),
                    expand=False
                )
                self._append_message("tool", panel)
                
                # Update sidebar immediately in case a tool updated artifacts
                self._update_sidebar()

        elif ev.type == EventType.ERROR:
            self._append_message("error", Text(f"✗ {ev.error}"))

        elif ev.type == EventType.DONE:
            pass

    def _append_message(self, msg_type: str, content: RenderableType) -> ChatMessage:
        chat = self.query_one("#chat_history")
        msg = ChatMessage(content, classes=f"msg_{msg_type}")
        chat.mount(msg)
        chat.scroll_end(animate=False)
        return msg

    async def _handle_command(self, cmd: str) -> None:
        c = cmd.lower().split()[0]
        
        if c == "/exit":
            self.exit()
            
        elif c == "/help":
            help_md = """### Commands
- `/exit`: Exit the agent
- `/reset`: Reset memory
- `/tools`: List tools
- `/memory`: Show recent memory
- `/artifacts`: List artifacts
- `/template <name>`: Switch agent template (General, Coding, Research)
- `/debug`: Show last LLM error and raw response
"""
            self._append_message("agent_response", Markdown(help_md))

        elif c == "/debug":
            llm = self.brain.llm
            debug_md = f"### LLM Debug info\n\n"
            if llm.last_raw_error:
                debug_md += f"**Last Error:** `{llm.last_raw_error}`\n\n"
            else:
                debug_md += "**Last Error:** None\n\n"
            
            if llm.last_raw_response:
                try:
                    # Robustly handle ollama Response objects (which are often Pydantic models)
                    data = llm.last_raw_response
                    if hasattr(data, 'model_dump'):
                        data = data.model_dump()
                    elif hasattr(data, 'dict'):
                        data = data.dict()
                    elif not isinstance(data, dict):
                        # Fallback for other objects
                        data = str(data)
                        
                    if isinstance(data, str):
                        resp_str = data
                    else:
                        resp_str = json.dumps(data, indent=2)
                except Exception as e:
                    resp_str = f"<Serialization Error: {e}>\n{llm.last_raw_response}"
                debug_md += f"**Last Raw Part:**\n```json\n{resp_str}\n```\n"
            else:
                debug_md += "**Last Raw Part:** None\n"
            
            self._append_message("agent_response", Markdown(debug_md))
            
        elif c == "/reset":
            self.brain.reset()
            self._append_message("agent_response", Text("✓ Session reset.", style="bold green"))
            
        elif c == "/tools":
            tools = self.brain.registry.list_all()
            if not tools:
                self._append_message("agent_response", Text("No tools discovered."))
            else:
                t = Table(title="Registered Tools", border_style="cyan")
                t.add_column("Name")
                t.add_column("Description")
                for tool in tools:
                    t.add_row(tool.name, tool.description)
                self._append_message("agent_response", t)
                
        elif c == "/memory":
            entries = self.brain.memory.get_recent(10)
            if not entries:
                self._append_message("agent_response", Text("Empty memory."))
            else:
                t = Table(title="Recent Memory", border_style="cyan")
                t.add_column("#")
                t.add_column("Role")
                t.add_column("Summary")
                for e in entries:
                    t.add_row(str(e.turn_id), e.role, e.summary[:80])
                self._append_message("agent_response", t)
                
        elif c == "/artifacts":
            names = self.artifacts.list_all()
            if not names:
                self._append_message("agent_response", Text("No artifacts."))
            else:
                self._append_message("agent_response", Text(f"Artifacts: {', '.join(names)}"))
                
        elif c == "/template":
            parts = cmd.split()
            if len(parts) < 2:
                available = ", ".join(TEMPLATES.keys())
                self._append_message("agent_response", Text(f"Current template: {self.brain.current_template}\nAvailable templates: {available}", style="cyan"))
                return
            
            new_tpl = parts[1]
            # Try to match case-insensitively
            matched = next((k for k in TEMPLATES if k.lower() == new_tpl.lower()), None)
            if matched:
                self.brain.switch_template(matched)
                self._append_message("agent_response", Text(f"✓ Switched to {matched} template.", style="bold green"))
                self._append_message("state", Text(f"System Prompt and Tools updated for {matched} mode.", style="dim italic"))
                # Redraw banner to show new mode
                banner_msg = self._append_message("banner", self._get_banner_panel())
            else:
                self._append_message("error", Text(f"Unknown template: {new_tpl}"))
        else:
            self._append_message("error", Text(f"Unknown command: {cmd}"))


class CLI:
    """Wrapper to maintain the previous CLI API compatibility."""
    def __init__(self, brain: Any, artifact_ctrl: Any) -> None:
        self.brain = brain
        self.artifacts = artifact_ctrl

    async def run(self) -> None:
        app = AgentApp(self.brain, self.artifacts)
        await app.run_async()
