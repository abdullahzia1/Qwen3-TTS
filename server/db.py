"""Job store and task queue on SQLite.

Local single-worker MVP. The schema and queries are kept plain so the move
to Postgres (SELECT ... FOR UPDATE SKIP LOCKED) is a driver swap, not a redesign.
"""

import datetime as dt
import json
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,              -- 'tts' | 'voice'
    status          TEXT NOT NULL,              -- queued | running | succeeded | partial | failed
    idempotency_key TEXT UNIQUE,
    request         TEXT NOT NULL,
    result          TEXT,
    error           TEXT,
    progress        REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(id),
    op          TEXT NOT NULL,                  -- 'tts.render' | 'tts.regen_chunk' | 'voice.create'
    args        TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'queued', -- queued | running | done | failed
    priority    INTEGER NOT NULL DEFAULT 0,     -- higher runs first
    error       TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_queue ON tasks(status, priority DESC, id);
"""

JSON_FIELDS = ("request", "result")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _row(r: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if r is None:
        return None
    d = dict(r)
    for k in JSON_FIELDS:
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    if "args" in d and d["args"] is not None:
        d["args"] = json.loads(d["args"])
    return d


class Store:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        # A crash mid-task leaves rows 'running'; put them back in the queue.
        self._db.execute("UPDATE tasks SET status='queued' WHERE status='running'")
        self._db.execute("UPDATE jobs SET status='queued' WHERE status='running'")

    # The connection is shared across threads, so rows must be fetched while
    # the lock is still held - a cursor read after release can be clobbered.
    def _q(self, sql: str, args=()) -> None:
        with self._lock:
            self._db.execute(sql, args)

    def _one(self, sql: str, args=()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchone()

    def _all(self, sql: str, args=()) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # --- jobs ---------------------------------------------------------------
    def create_job(self, kind: str, request: Dict[str, Any], op: str, args: Dict[str, Any],
                   idempotency_key: Optional[str] = None, priority: int = 0) -> Dict[str, Any]:
        """Insert a job and its first task atomically. A repeated idempotency key returns the original job."""
        with self._lock:
            if idempotency_key:
                existing = self._db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if existing:
                    return _row(existing)
            job_id = f"{kind}_{uuid.uuid4().hex[:12]}"
            ts = now()
            self._db.execute("BEGIN")
            try:
                self._db.execute(
                    "INSERT INTO jobs(id, kind, status, idempotency_key, request, created_at, updated_at) "
                    "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                    (job_id, kind, idempotency_key, json.dumps(request), ts, ts),
                )
                self._db.execute(
                    "INSERT INTO tasks(job_id, op, args, priority, created_at) VALUES (?, ?, ?, ?, ?)",
                    (job_id, op, json.dumps(args), priority, ts),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            return _row(self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return _row(self._one("SELECT * FROM jobs WHERE id=?", (job_id,)))

    def list_jobs(self, kind: Optional[str] = None, limit: int = 50, tag: Optional[str] = None,
                  untagged: bool = False) -> List[Dict[str, Any]]:
        """Newest first. `tag` keeps only that tag; `untagged` keeps only jobs without one."""
        where, args = [], []
        if kind:
            where.append("kind=?")
            args.append(kind)
        if tag:
            where.append("json_extract(request, '$.tag')=?")
            args.append(tag)
        elif untagged:
            where.append("json_extract(request, '$.tag') IS NULL")
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "")
        rows = self._all(sql + " ORDER BY created_at DESC LIMIT ?", (*args, limit))
        return [_row(r) for r in rows]

    def update_job(self, job_id: str, **fields: Any) -> None:
        for k in JSON_FIELDS:
            if k in fields and fields[k] is not None:
                fields[k] = json.dumps(fields[k])
        fields["updated_at"] = now()
        cols = ", ".join(f"{k}=?" for k in fields)
        self._q(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    # --- tasks --------------------------------------------------------------
    def add_task(self, job_id: str, op: str, args: Dict[str, Any], priority: int = 0) -> None:
        self._q(
            "INSERT INTO tasks(job_id, op, args, priority, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, op, json.dumps(args), priority, now()),
        )

    def claim_task(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM tasks WHERE status='queued' ORDER BY priority DESC, id LIMIT 1"
            ).fetchone()
            if r is None:
                return None
            self._db.execute("UPDATE tasks SET status='running' WHERE id=?", (r["id"],))
            return _row(r)

    def finish_task(self, task_id: int, error: Optional[str] = None) -> None:
        self._q("UPDATE tasks SET status=?, error=? WHERE id=?", ("failed" if error else "done", error, task_id))

    def queue_depth(self) -> int:
        return self._one("SELECT COUNT(*) FROM tasks WHERE status IN ('queued', 'running')")[0]

    def has_pending(self, job_id: str) -> bool:
        return self._one(
            "SELECT 1 FROM tasks WHERE job_id=? AND status IN ('queued', 'running') LIMIT 1", (job_id,)
        ) is not None
