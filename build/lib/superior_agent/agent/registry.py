"""Tool Registry — auto-discovers, searches, and lazy-loads agent tools.

Tools live in `agent/tools/` as individual `.py` files.  Each must expose
exactly one public function with a docstring formatted per §3.2.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Lightweight descriptor parsed from a tool module's docstring."""
    name: str
    module_path: str          # dotted import path
    description: str = ""
    args: dict[str, str] = field(default_factory=dict)
    returns: str = ""
    when_to_use: str = ""

    def matches(self, query: str) -> bool:
        """Case-insensitive keyword match against name, description, when_to_use."""
        q = query.lower()
        return (
            q in self.name.lower()
            or q in self.description.lower()
            or q in self.when_to_use.lower()
            or any(q in v.lower() for v in self.args.values())
        )

    def to_openai_schema(self, exclude_internal: set[str] | None = None) -> dict[str, Any]:
        """Generate an OpenAI-compatible function tool definition.

        Parameters marked in *exclude_internal* (e.g. ``{"workdir", "platform_profile"}``)
        are omitted from the schema — these are injected by the agent, not by the LLM.
        """
        exclude = exclude_internal or {"workdir", "platform_profile", "registry"}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, pdesc in self.args.items():
            if pname in exclude:
                continue
            properties[pname] = {"type": "string", "description": pdesc}
            required.append(pname)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class Registry:
    """Auto-discovers tool modules and provides search + lazy-loading."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}

    # ------------------------------------------------------------------
    # Discovery (AST-based, no imports)
    # ------------------------------------------------------------------

    def discover(self, tools_dir: str | Path) -> int:
        """Walk *tools_dir* for ``.py`` tool files, parse metadata.

        Returns the number of tools discovered.
        """
        tools_dir = Path(tools_dir)
        if not tools_dir.is_dir():
            logger.warning("Tools directory does not exist: %s", tools_dir)
            return 0

        count = 0
        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            meta = self._parse_module(py_file)
            if meta:
                self._tools[meta.name] = meta
                count += 1
                logger.debug("Discovered tool: %s", meta.name)

        logger.info("Discovered %d tool(s) in %s", count, tools_dir)
        return count

    # ------------------------------------------------------------------
    # Search (metadata only — no import)
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[ToolMetadata]:
        """Return tool metadata whose name/description/trigger matches *query*."""
        return [t for t in self._tools.values() if t.matches(query)]

    def list_all(self) -> list[ToolMetadata]:
        """Return metadata for every discovered tool."""
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Load (performs the actual import)
    # ------------------------------------------------------------------

    def load(self, tool_name: str) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Import the tool module, return ``(callable, openai_schema)``.

        Raises ``KeyError`` if the tool was not discovered.
        """
        meta = self._tools.get(tool_name)
        if meta is None:
            raise KeyError(f"Unknown tool: {tool_name!r}")

        mod = importlib.import_module(meta.module_path)
        func = getattr(mod, meta.name)
        return func, meta.to_openai_schema()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_module(py_file: Path) -> ToolMetadata | None:
        """Use the AST to extract the public function and its docstring."""
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError) as exc:
            logger.warning("Cannot parse %s: %s", py_file, exc)
            return None

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue

            doc = ast.get_docstring(node) or ""
            description = _extract_field(doc, "Description")
            args_raw = _extract_field(doc, "Args")
            returns = _extract_field(doc, "Returns")
            when = _extract_field(doc, "When to use")

            args = _parse_args_block(args_raw)

            # Build dotted module path from file path
            # e.g.  .../superior_agent/agent/tools/read_file.py
            #     → superior_agent.agent.tools.read_file
            parts = py_file.resolve().parts
            try:
                sa_idx = parts.index("superior_agent")
            except ValueError:
                module_path = py_file.stem
            else:
                module_path = ".".join(parts[sa_idx:]).removesuffix(".py")

            return ToolMetadata(
                name=node.name,
                module_path=module_path,
                description=description,
                args=args,
                returns=returns,
                when_to_use=when,
            )
        return None


# ------------------------------------------------------------------
# Docstring parsing helpers
# ------------------------------------------------------------------

def _extract_field(docstring: str, field: str) -> str:
    """Extract a ``Field: value`` line from a tool docstring."""
    pattern = rf"(?:^|\n)\s*{re.escape(field)}\s*:\s*(.+?)(?:\n\s*\w+\s*:|$)"
    m = re.search(pattern, docstring, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_args_block(raw: str) -> dict[str, str]:
    """Parse ``param: desc  param2: desc2`` into a dict."""
    args: dict[str, str] = {}
    if not raw:
        return args
    # Split on two-or-more spaces or newlines to separate param entries
    entries = re.split(r"\s{2,}|\n", raw)
    for entry in entries:
        entry = entry.strip()
        if ":" in entry:
            k, v = entry.split(":", 1)
            args[k.strip()] = v.strip()
    return args
