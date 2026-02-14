"""SQLite-backed persistence for design sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from modules.session import DesignSession
from modules.vision import VisionSystem


class SessionStorage:
    """Persist/restore DesignSession state in SQLite to survive server restarts."""

    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save(self, session: DesignSession) -> None:
        payload = {
            "keyframes": session.keyframes,
            "user_anchors": session.user_anchors,
            "detected_supports": session.detected_supports,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, status, payload, updated_at)
                VALUES(?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id)
                DO UPDATE SET status=excluded.status, payload=excluded.payload, updated_at=CURRENT_TIMESTAMP
                """,
                (session.session_id, session.status, json.dumps(payload, ensure_ascii=False)),
            )

    def load(self, session_id: str, vision_system: VisionSystem) -> Optional[DesignSession]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, status, payload FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        payload = json.loads(row["payload"])
        return DesignSession(
            session_id=row["session_id"],
            vision_system=vision_system,
            status=row["status"],
            keyframes=payload.get("keyframes", []),
            user_anchors=payload.get("user_anchors", []),
            detected_supports=payload.get("detected_supports", []),
        )
