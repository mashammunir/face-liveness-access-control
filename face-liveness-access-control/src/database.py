"""
src/database.py

SQLite-backed storage for registered users and access logs.

Design notes:
  - Only face EMBEDDINGS are stored, never raw face images, per the
    project's privacy requirements (prefer embeddings over raw images,
    avoid unnecessary storage of biometric imagery).
  - No passwords or secrets are stored anywhere in this database or in
    source code.
  - Two lightweight tables: `users` (registered people) and
    `access_log` (every access attempt, granted or denied).
  - All SQLite operations are wrapped so a locked/corrupt database
    file or a disk-full condition raises a clear DatabaseError instead
    of an unhandled sqlite3.Error reaching the UI and crashing the app.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config


class DatabaseError(Exception):
    """Raised when a database operation fails (locked file, corrupt
    file, disk full, permissions, etc). Callers (e.g. the Streamlit UI)
    should catch this and show the user a clear message rather than
    letting a raw sqlite3.Error surface."""


@contextmanager
def _connect(db_path: Path):
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not open database at {db_path}: {e}") from e

    try:
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Database operation failed on {db_path}: {e}") from e
    finally:
        conn.close()


class UserDatabase:
    """Stores registered users: id, name, embedding (as JSON), created_at, status."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else config.DATABASE_PATH
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise DatabaseError(f"Could not create directory for {self.db_path}: {e}") from e
        self._init_schema()

    def _init_schema(self):
        with _connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                )
            """)

    def add_user(self, name: str, embedding: np.ndarray) -> int:
        """Register a new user. Returns the new user's id.

        Raises ValueError for invalid input (empty name, missing
        embedding) and DatabaseError if the underlying database
        operation itself fails.
        """
        if not name or not name.strip():
            raise ValueError("User name must not be empty.")
        if embedding is None:
            raise ValueError("Embedding must not be None.")

        embedding_json = json.dumps(embedding.astype(float).tolist())
        created_at = datetime.now(timezone.utc).isoformat()

        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO users (name, embedding, created_at, status) VALUES (?, ?, ?, 'active')",
                (name.strip(), embedding_json, created_at),
            )
            return cur.lastrowid

    def get_all_active_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Returns {name: embedding} for all active users, suitable for
        src.face_recognition.match_embedding().

        Note: if multiple users share the same name, only the most
        recently added one's embedding will be kept under that name in
        this dict. Use list_users() for the full record including ids.

        Raises DatabaseError if the query fails, or if a stored
        embedding is corrupt (unparseable JSON) — better to fail loudly
        than silently drop a user from the recognition pool.
        """
        result = {}
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT name, embedding FROM users WHERE status = 'active' ORDER BY id ASC"
            ).fetchall()
            for row in rows:
                try:
                    emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
                except (json.JSONDecodeError, ValueError) as e:
                    raise DatabaseError(
                        f"Corrupt embedding stored for user '{row['name']}': {e}"
                    ) from e
                result[row["name"]] = emb
        return result

    def list_users(self) -> List[dict]:
        """Returns all users (active and inactive) without their raw embeddings."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, name, created_at, status FROM users ORDER BY id ASC"
            ).fetchall()
            return [dict(row) for row in rows]

    def deactivate_user(self, user_id: int):
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE users SET status = 'inactive' WHERE id = ?", (user_id,))

    def delete_user(self, user_id: int):
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


class AccessLogDatabase:
    """Stores every access attempt for later review in the UI."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else config.ACCESS_LOG_PATH
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise DatabaseError(f"Could not create directory for {self.db_path}: {e}") from e
        self._init_schema()

    def _init_schema(self):
        with _connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_name TEXT,
                    liveness_result TEXT NOT NULL,
                    liveness_confidence REAL,
                    recognition_result TEXT NOT NULL,
                    recognition_confidence REAL,
                    access_result TEXT NOT NULL
                )
            """)

    def log_access(
        self,
        user_name: Optional[str],
        liveness_result: str,
        liveness_confidence: float,
        recognition_result: str,
        recognition_confidence: float,
        access_result: str,
    ) -> int:
        """Raises DatabaseError if the write fails. Callers in the UI
        should catch this — a logging failure should be shown to the
        user, not silently swallowed, but it also should not be allowed
        to crash the live camera loop."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO access_log
                   (timestamp, user_name, liveness_result, liveness_confidence,
                    recognition_result, recognition_confidence, access_result)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, user_name, liveness_result, liveness_confidence,
                 recognition_result, recognition_confidence, access_result),
            )
            return cur.lastrowid

    def get_recent(self, limit: int = 50) -> List[dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM access_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
