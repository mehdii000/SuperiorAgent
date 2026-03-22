"""CLI Frontend — developer-friendly terminal UI using Textual.

Provides a full-screen layout with:
  • Chat history (scrollable)
  • Bottom command input
  • Right sidebar showing the plan/tasks
  • Real-time streaming of thinking, response, and tools
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Input, Static, Markdown as TMarkdown
from textual import work

from superior_agent.core.models import EventType, LLMEvent

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
        width: 1fr;
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
                yield Input(placeholder="Ask Superior Agent... (type /help for commands)", id="chat_input")
                
            # Sidebar for Artifacts/Plan
            with Vertical(id="sidebar"):
                yield TMarkdown("# Tasks / Plan\n_No tasks or plans yet._", id="sidebar_content")
                
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat_input").focus()
        self._append_message("banner", self._get_banner_panel())
        self._update_sidebar()

    def _get_banner_panel(self) -> Panel:
        pp = self.brain.platform
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column("K", style="dim", width=12)
        t.add_column("V", style="bold cyan")
        t.add_row("  OS", pp.os_name)
        t.add_row("  Shell", pp.shell)
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

    def _update_sidebar(self) -> None:
        # Load 'tasks' or 'implementation_plan' from artifacts
        tasks_content = self.artifacts.get("tasks")
        plan_content = self.artifacts.get("implementation_plan")
        
        md_text = ""
        if tasks_content:
            md_text += tasks_content + "\n\n---\n\n"
        if plan_content:
            md_text += plan_content
            
        # Also check for 'implementation' if 'implementation_plan' is missing (backward compatibility during transition)
        if not plan_content:
            legacy_plan = self.artifacts.get("implementation")
            if legacy_plan:
                md_text += legacy_plan
            
        if not md_text:
            md_text = "*No tasks or plans available.*"
            
        sidebar = self.query_one("#sidebar_content", TMarkdown)
        sidebar.update(md_text)

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
        self._update_sidebar()

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
                result = ev.content
                if len(result) > 500:
                    result = result[:500] + "\n...(truncated)"
                    
                panel = Panel(
                    result,
                    title=f"⚡ {ev.tool_name} Result ({elapsed_str})",
                    border_style="yellow",
                    padding=(0, 1)
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
"""
            self._append_message("agent_response", Markdown(help_md))
            
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
