"""Tests for agent.registry — tool discovery, search, and loading."""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from superior_agent.agent.registry import Registry, ToolMetadata


# ------------------------------------------------------------------
# Fixture: a temp directory with sample tool files
# ------------------------------------------------------------------

@pytest.fixture
def tools_dir(tmp_path: Path) -> Path:
    """Create a temp tools directory with two sample tools."""
    (tmp_path / "__init__.py").write_text("")

    (tmp_path / "greet.py").write_text(
        textwrap.dedent('''\
            """Sample greet tool."""

            def greet(name: str) -> str:
                """Description: Greets a person by name.
                Args: name: The name of the person to greet
                Returns: A greeting string
                When to use: When the user asks to say hello to someone"""
                return f"Hello, {name}!"
        ''')
    )

    (tmp_path / "add_numbers.py").write_text(
        textwrap.dedent('''\
            """Sample math tool."""

            def add_numbers(a: str, b: str) -> str:
                """Description: Adds two numbers together.
                Args: a: First number  b: Second number
                Returns: The sum as a string
                When to use: When the user needs to add two numbers"""
                return str(int(a) + int(b))
        ''')
    )

    # Private/invalid file — should be skipped
    (tmp_path / "_internal.py").write_text("def _helper(): pass")

    return tmp_path


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestDiscovery:
    def test_discover_finds_tools(self, tools_dir: Path):
        reg = Registry()
        count = reg.discover(tools_dir)
        assert count == 2

    def test_discover_nonexistent_dir(self):
        reg = Registry()
        count = reg.discover("/nonexistent/path")
        assert count == 0

    def test_discover_skips_private(self, tools_dir: Path):
        reg = Registry()
        reg.discover(tools_dir)
        names = [t.name for t in reg.list_all()]
        assert "_internal" not in names
        assert "_helper" not in names


class TestSearch:
    def test_search_by_keyword(self, tools_dir: Path):
        reg = Registry()
        reg.discover(tools_dir)
        results = reg.search("greet")
        assert len(results) == 1
        assert results[0].name == "greet"

    def test_search_by_description(self, tools_dir: Path):
        reg = Registry()
        reg.discover(tools_dir)
        results = reg.search("adds")
        assert len(results) == 1
        assert results[0].name == "add_numbers"

    def test_search_no_match(self, tools_dir: Path):
        reg = Registry()
        reg.discover(tools_dir)
        results = reg.search("zzzznonexistent")
        assert results == []


class TestMetadata:
    def test_metadata_fields(self, tools_dir: Path):
        reg = Registry()
        reg.discover(tools_dir)
        results = reg.search("greet")
        meta = results[0]
        assert meta.description == "Greets a person by name."
        assert "name" in meta.args
        assert meta.when_to_use != ""

    def test_to_openai_schema(self, tools_dir: Path):
        reg = Registry()
        reg.discover(tools_dir)
        meta = reg.search("greet")[0]
        schema = meta.to_openai_schema(exclude_internal=set())
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert "name" in schema["function"]["parameters"]["properties"]


class TestLoading:
    def test_load_existing(self, tools_dir: Path):
        """Loading a discovered tool should return a callable."""
        reg = Registry()
        reg.discover(tools_dir)

        # Patch the module path so importlib can find it
        for meta in reg.list_all():
            # Since these are temp files, we need to make them importable
            import sys
            sys.path.insert(0, str(tools_dir.parent))
            meta.module_path = tools_dir.name + "." + meta.name

        func, schema = reg.load("greet")
        assert callable(func)
        assert func("World") == "Hello, World!"

    def test_load_unknown(self, tools_dir: Path):
        reg = Registry()
        reg.discover(tools_dir)
        with pytest.raises(KeyError, match="Unknown tool"):
            reg.load("nonexistent_tool")
