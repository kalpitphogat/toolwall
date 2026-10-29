"""SQLite audit log of every scanned tool output.

Stores hashes by default plus, optionally, a short excerpt around the flagged window,
never the full tool output: an audit log should not become a second copy of
everything the agent read.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    channel     TEXT    NOT NULL,          -- 'gateway' or 'api'
    upstream    TEXT,
    tool        TEXT,
    verdict     TEXT    NOT NULL,
    score       REAL    NOT NULL,
    enforced    INTEGER NOT NULL,          -- 0 in monitor mode: verdict logged, output untouched
    latency_ms  REAL    NOT NULL,
    content_sha256 TEXT NOT NULL,
    args_sha256 TEXT,
    chars       INTEGER NOT NULL,
    excerpt     TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS events_verdict ON events(verdict);
"""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def excerpt(text: str, span: tuple[int, int] | None, width: int = 240) -> str:
    if not span:
        return text[:width]
    s, e = span
    mid = (s + e) // 2
    start = max(0, mid - width // 2)
    return ("…" if start else "") + text[start : start + width] + ("…" if start + width < len(text) else "")


@dataclass
class Event:
    channel: str
    verdict: str
    score: float
    enforced: bool
    latency_ms: float
    content: str
    span: tuple[int, int] | None = None
    upstream: str | None = None
    tool: str | None = None
    args_repr: str | None = None


class AuditStore:
    def __init__(self, path: str | Path, store_excerpts: bool = True):
        self.path = str(path)
        self.store_excerpts = store_excerpts
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def record(self, ev: Event) -> int:
        flagged = ev.verdict != "allow"
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO events (ts, channel, upstream, tool, verdict, score, enforced, latency_ms, content_sha256,"
                " args_sha256, chars, excerpt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), ev.channel, ev.upstream, ev.tool, ev.verdict, ev.score, int(ev.enforced), ev.latency_ms,
                 sha256(ev.content), sha256(ev.args_repr) if ev.args_repr is not None else None, len(ev.content),
                 excerpt(ev.content, ev.span) if (self.store_excerpts and flagged) else None),
            )
            return int(cur.lastrowid)

    def events(self, limit: int = 100, verdict: str | None = None, tool: str | None = None) -> list[dict]:
        q, params = "SELECT * FROM events", []
        where = []
        if verdict:
            where.append("verdict = ?")
            params.append(verdict)
        if tool:
            where.append("tool = ?")
            params.append(tool)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, params)]

    def stats(self) -> dict:
        with self._lock:
            by_verdict = {r["verdict"]: r["n"] for r in self._conn.execute(
                "SELECT verdict, COUNT(*) AS n FROM events GROUP BY verdict")}
            by_tool = [dict(r) for r in self._conn.execute(
                "SELECT tool, COUNT(*) AS n, SUM(verdict != 'allow') AS flagged, MAX(score) AS max_score "
                "FROM events WHERE tool IS NOT NULL GROUP BY tool ORDER BY flagged DESC, n DESC LIMIT 20")]
            lat = [r[0] for r in self._conn.execute("SELECT latency_ms FROM events ORDER BY latency_ms")]
        p = (lambda q: lat[min(len(lat) - 1, int(q * (len(lat) - 1)))]) if lat else (lambda q: None)
        return {"events": sum(by_verdict.values()), "by_verdict": by_verdict, "by_tool": by_tool,
                "latency_ms_p50": p(0.5), "latency_ms_p95": p(0.95)}

    def close(self) -> None:
        self._conn.close()
