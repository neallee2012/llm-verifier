from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "chat.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES threads(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        # Migration: add images column if missing
        cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "images" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN images TEXT")


def list_threads() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM threads ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_thread(thread_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, created_at FROM threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
        return dict(row) if row else None


def create_thread(title: str | None) -> dict[str, Any]:
    thread_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO threads (id, title, created_at) VALUES (?, ?, ?)",
            (thread_id, title or "New thread", created_at),
        )
    return {"id": thread_id, "title": title or "New thread", "created_at": created_at}


def update_thread_title(thread_id: str, title: str) -> dict[str, Any] | None:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE threads SET title = ? WHERE id = ?",
            (title, thread_id),
        )
        if cur.rowcount == 0:
            return None
    return get_thread(thread_id)


def delete_thread(thread_id: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        cur = conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        return cur.rowcount > 0


def list_messages(thread_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, thread_id, role, content, model, created_at, images
            FROM messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
            """,
            (thread_id,),
        ).fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            raw = msg.get("images")
            msg["images"] = json.loads(raw) if raw else []
            result.append(msg)
        return result


def add_message(
    thread_id: str,
    role: str,
    content: str,
    model: str | None = None,
    images: list[str] | None = None,
) -> dict[str, Any]:
    message_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    images_json = json.dumps(images) if images else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, thread_id, role, content, model, created_at, images)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, thread_id, role, content, model, created_at, images_json),
        )
    return {
        "id": message_id,
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "model": model,
        "created_at": created_at,
        "images": images or [],
    }


def get_setting(key: str, default: str = "") -> str:
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
