"""Artifact Controller — virtual document store backed by SQLite.

Artifacts are never files on disk in the user's workdir.  They live
exclusively in ``~/.superior_agent/sessions/<session_id>.db``.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".superior_agent" / "sessions"


class ArtifactController:
    """CRUD interface for the virtual artifact store."""

    def __init__(self, session_id: str, root: Path | None = None) -> None:
        self.session_id = session_id
        root = root or _DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self._db_path = root / f"{session_id}.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._ensure_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, name: str, content: str) -> None:
        """Create or update an artifact.  Previous content is preserved
        in ``artifact_history`` before overwriting."""
        cur = self._conn.execute(
            "SELECT id, content FROM artifacts WHERE name = ?", (name,)
        )
        row = cur.fetchone()

        now = datetime.now(timezone.utc).isoformat()

        if row is not None:
            art_id, old_content = row
            # Save previous version
            self._conn.execute(
                "INSERT INTO artifact_history (artifact_id, content, saved_at) VALUES (?, ?, ?)",
                (art_id, old_content, now),
            )
            self._conn.execute(
                "UPDATE artifacts SET content = ?, updated_at = ? WHERE id = ?",
                (content, now, art_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO artifacts (name, content, updated_at) VALUES (?, ?, ?)",
                (name, content, now),
            )
        self._conn.commit()

    def get(self, name: str) -> str | None:
        """Retrieve the current content of an artifact by name."""
        cur = self._conn.execute(
            "SELECT content FROM artifacts WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def history(self, name: str, limit: int = 10) -> list[dict[str, str]]:
        """Retrieve past versions of an artifact."""
        cur = self._conn.execute(
            """
            SELECT ah.content, ah.saved_at
              FROM artifact_history ah
              JOIN artifacts a ON a.id = ah.artifact_id
             WHERE a.name = ?
             ORDER BY ah.saved_at DESC
             LIMIT ?
            """,
            (name, limit),
        )
        return [{"content": r[0], "saved_at": r[1]} for r in cur.fetchall()]

    def list_all(self) -> list[str]:
        """Return the names of all stored artifacts."""
        cur = self._conn.execute("SELECT name FROM artifacts ORDER BY name")
        return [r[0] for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                content     TEXT NOT NULL,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS artifact_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id INTEGER NOT NULL REFERENCES artifacts(id),
                content     TEXT NOT NULL,
                saved_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._conn.commit()

    def _ensure_defaults(self) -> None:
        """Create the two built-in artifacts on first run."""
        if self.get("tasks") is None:
            self.upsert("tasks", "# Tasks\n\n_No tasks yet._\n")
        if self.get("implementation_plan") is None:
            self.upsert("implementation_plan", "# Implementation Plan\n\n_No active plan._\n")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
