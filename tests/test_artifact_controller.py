"""Tests for agent.artifact_controller — CRUD, history, defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from superior_agent.agent.artifact_controller import ArtifactController


@pytest.fixture
def ctrl(tmp_path: Path) -> ArtifactController:
    """Create an ArtifactController using a temp directory for the DB."""
    c = ArtifactController("test-session", root=tmp_path)
    yield c
    c.close()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestDefaults:
    def test_default_artifacts_exist(self, ctrl: ArtifactController):
        names = ctrl.list_all()
        assert "tasks" in names
        assert "implementation" in names

    def test_default_tasks_content(self, ctrl: ArtifactController):
        content = ctrl.get("tasks")
        assert content is not None
        assert "Tasks" in content

    def test_default_implementation_content(self, ctrl: ArtifactController):
        content = ctrl.get("implementation")
        assert content is not None
        assert "Implementation" in content


class TestCRUD:
    def test_upsert_create(self, ctrl: ArtifactController):
        ctrl.upsert("notes", "My notes")
        assert ctrl.get("notes") == "My notes"

    def test_upsert_update(self, ctrl: ArtifactController):
        ctrl.upsert("notes", "Version 1")
        ctrl.upsert("notes", "Version 2")
        assert ctrl.get("notes") == "Version 2"

    def test_get_nonexistent(self, ctrl: ArtifactController):
        assert ctrl.get("does_not_exist") is None

    def test_list_all(self, ctrl: ArtifactController):
        ctrl.upsert("alpha", "A")
        ctrl.upsert("beta", "B")
        names = ctrl.list_all()
        assert "alpha" in names
        assert "beta" in names
        # defaults too
        assert "tasks" in names


class TestHistory:
    def test_history_preserved_on_update(self, ctrl: ArtifactController):
        ctrl.upsert("log", "Entry 1")
        ctrl.upsert("log", "Entry 2")
        ctrl.upsert("log", "Entry 3")
        h = ctrl.history("log")
        # Should have history entries for the two overwrites
        assert len(h) >= 2
        contents = [entry["content"] for entry in h]
        assert "Entry 2" in contents
        assert "Entry 1" in contents

    def test_history_limit(self, ctrl: ArtifactController):
        for i in range(10):
            ctrl.upsert("counter", f"v{i}")
        h = ctrl.history("counter", limit=3)
        assert len(h) == 3

    def test_history_nonexistent(self, ctrl: ArtifactController):
        h = ctrl.history("ghost")
        assert h == []
