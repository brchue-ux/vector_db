"""Chunk store and BM25 index, on SQLite + FTS5.

Why SQLite/FTS5 and nothing else: FTS5's `bm25()` is Okapi BM25 with k1=1.2 and
b=0.75 - the same parameterisation the study measured (§B.5) - it is in the
Python standard library, it needs no model and no network, and it gives
incremental insert/delete for free. Phase 1 therefore has zero dependencies.

Phase 2 seam (dense index + reciprocal-rank fusion) - deliberately minimal:
`chunks.id` is a stable integer key that a vector file can address by row, and
`retrieve.Retriever` is the one shape a second retriever has to satisfy. There
is no plugin machinery here for a single future retriever.

The index contains the captain's prose. It is exactly as sensitive as the
transcripts (F9), so the database file is created 0600 inside a 0700 directory.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import chunk as chunk_mod
from . import ingest as ingest_mod
from .clean import candidate_lines, trim_boilerplate

SCHEMA_VERSION = "1"

DEFAULT_DB = Path(
    os.environ.get("VDB_DB")
    or (Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "vdb" / "index.sqlite3")
)

# A line counts as invariant template boilerplate when it is long enough to
# matter and it recurs across this many distinct messages (§7: the 34 template
# lines were identified by frequency, not by hand).
BOILERPLATE_MIN_MESSAGES = 50
BOILERPLATE_MIN_LEN = 40

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS files (
    id        INTEGER PRIMARY KEY,
    path      TEXT UNIQUE NOT NULL,
    kind      TEXT NOT NULL,
    sha256    TEXT NOT NULL,
    size      INTEGER NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    n_messages INTEGER NOT NULL DEFAULT 0,
    n_chunks  INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    file_id   INTEGER NOT NULL REFERENCES files(id),
    session_id TEXT,
    project   TEXT,
    role      TEXT,
    ts        TEXT,
    msg_uuid  TEXT,
    msg_seq   INTEGER,
    part      INTEGER NOT NULL DEFAULT 0,
    n_parts   INTEGER NOT NULL DEFAULT 1,
    n_chars   INTEGER NOT NULL DEFAULT 0,
    is_sidechain INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS chunks_by_session ON chunks(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, tokenize='unicode61 remove_diacritics 2');

-- Frequency table behind the boilerplate trim. `h` is a hash of the line so the
-- primary key stays small; the text is kept so `vdb boilerplate` can show what
-- is being trimmed.
CREATE TABLE IF NOT EXISTS line_freq (
    h TEXT PRIMARY KEY, line TEXT NOT NULL, n INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS file_line (
    file_id INTEGER NOT NULL, h TEXT NOT NULL, n INTEGER NOT NULL,
    PRIMARY KEY (file_id, h)
);
"""


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    messages: int = 0
    chunks: int = 0
    chars: int = 0
    trimmed_lines: int = 0
    secrets: dict[str, int] | None = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["secrets"] = self.secrets or {}
        return d


def _line_hash(line: str) -> str:
    return hashlib.sha1(line.encode("utf-8")).hexdigest()[:20]


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(db_path.parent, 0o700)
    except OSError:
        pass
    existed = db_path.exists()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (SCHEMA_VERSION,),
    )
    conn.commit()
    if not existed:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.chmod(str(db_path) + suffix, 0o600)
            except OSError:
                pass
    return conn


def reset(conn: sqlite3.Connection) -> None:
    """Drop every indexed artefact, keeping the schema."""
    conn.executescript(
        "DELETE FROM chunks_fts; DELETE FROM chunks; DELETE FROM file_line;"
        " DELETE FROM line_freq; DELETE FROM files;"
    )
    conn.commit()


def boilerplate_lines(conn: sqlite3.Connection, threshold: int = BOILERPLATE_MIN_MESSAGES) -> set[str]:
    rows = conn.execute("SELECT line FROM line_freq WHERE n >= ?", (threshold,))
    return {r["line"] for r in rows}


def _forget_file(conn: sqlite3.Connection, file_id: int) -> None:
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)",
        (file_id,),
    )
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    for row in conn.execute("SELECT h, n FROM file_line WHERE file_id = ?", (file_id,)).fetchall():
        conn.execute("UPDATE line_freq SET n = n - ? WHERE h = ?", (row["n"], row["h"]))
    conn.execute("DELETE FROM file_line WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM line_freq WHERE n <= 0")


def _count_lines(conn: sqlite3.Connection, file_id: int, messages: list[ingest_mod.Message]) -> None:
    """Record this file's contribution to the corpus-wide line frequencies."""
    tally: dict[str, tuple[str, int]] = {}
    for msg in messages:
        for line in candidate_lines(msg.text, BOILERPLATE_MIN_LEN):
            h = _line_hash(line)
            prev = tally.get(h)
            tally[h] = (line, (prev[1] if prev else 0) + 1)
    for h, (line, n) in tally.items():
        conn.execute(
            "INSERT INTO file_line(file_id, h, n) VALUES(?,?,?) "
            "ON CONFLICT(file_id, h) DO UPDATE SET n = n + excluded.n",
            (file_id, h, n),
        )
        conn.execute(
            "INSERT INTO line_freq(h, line, n) VALUES(?,?,?) "
            "ON CONFLICT(h) DO UPDATE SET n = n + excluded.n",
            (h, line, n),
        )


def _insert_chunks(
    conn: sqlite3.Connection,
    file_id: int,
    messages: list[ingest_mod.Message],
    boilerplate: set[str],
    stats: IndexStats,
) -> int:
    n_chunks = 0
    for msg in messages:
        text = trim_boilerplate(msg.text, boilerplate)
        if not text.strip():
            # Only the invariant lines were present; there is no message body
            # left to index. The message is not "deleted" in the §7 sense -
            # nothing of its own remained.
            continue
        stats.trimmed_lines += len(msg.text.splitlines()) - len(text.splitlines())
        parts = chunk_mod.split_message(text)
        for i, part in enumerate(parts):
            cur = conn.execute(
                "INSERT INTO chunks(file_id, session_id, project, role, ts, msg_uuid,"
                " msg_seq, part, n_parts, n_chars, is_sidechain)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    file_id,
                    msg.session_id,
                    msg.project,
                    msg.role,
                    msg.ts,
                    msg.uuid,
                    msg.seq,
                    i,
                    len(parts),
                    len(part),
                    int(msg.is_sidechain),
                ),
            )
            conn.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)", (cur.lastrowid, part)
            )
            n_chunks += 1
            stats.chars += len(part)
    return n_chunks


def index(
    conn: sqlite3.Connection,
    root: Path = ingest_mod.DEFAULT_ROOT,
    rebuild: bool = False,
    progress=None,
) -> IndexStats:
    """Index (or re-index) the corpus under `root`.

    Incremental by default: a file whose size, mtime and content hash are
    unchanged is skipped; a changed file has its chunks replaced; a file that
    has disappeared has its chunks removed. `rebuild=True` starts from empty.
    """
    stats = IndexStats(secrets={})
    if rebuild:
        reset(conn)

    known = {
        r["path"]: r for r in conn.execute("SELECT id, path, sha256, size, mtime_ns FROM files")
    }
    seen: set[str] = set()
    staged: list[tuple[int, list[ingest_mod.Message]]] = []
    now = datetime.now(timezone.utc).isoformat()

    for src in ingest_mod.scan(root):
        stats.scanned += 1
        rel = ingest_mod.relpath(src.path, root)
        seen.add(rel)
        prior = known.get(rel)
        if prior and prior["size"] == src.size and prior["mtime_ns"] == src.mtime_ns:
            stats.skipped += 1
            continue
        digest = ingest_mod.sha256(src.path)
        if prior and prior["sha256"] == digest:
            # Touched but not changed: refresh the cheap stat fields only.
            conn.execute(
                "UPDATE files SET size=?, mtime_ns=? WHERE id=?",
                (src.size, src.mtime_ns, prior["id"]),
            )
            stats.skipped += 1
            continue

        messages, secret_counts = ingest_mod.read_messages(src)
        for k, v in secret_counts.items():
            stats.secrets[k] = stats.secrets.get(k, 0) + v

        if prior:
            file_id = prior["id"]
            _forget_file(conn, file_id)
            conn.execute(
                "UPDATE files SET kind=?, sha256=?, size=?, mtime_ns=?, n_messages=?,"
                " n_chunks=0, indexed_at=? WHERE id=?",
                (src.kind, digest, src.size, src.mtime_ns, len(messages), now, file_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO files(path, kind, sha256, size, mtime_ns, n_messages,"
                " n_chunks, indexed_at) VALUES(?,?,?,?,?,?,0,?)",
                (rel, src.kind, digest, src.size, src.mtime_ns, len(messages), now),
            )
            file_id = cur.lastrowid

        _count_lines(conn, file_id, messages)
        staged.append((file_id, messages))
        stats.indexed += 1
        stats.messages += len(messages)
        if progress:
            progress(rel, len(messages))

    for rel, row in known.items():
        if rel not in seen:
            _forget_file(conn, row["id"])
            conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
            stats.removed += 1

    conn.commit()

    # Boilerplate is a corpus-wide property, so the trim set is computed after
    # every changed file has contributed its line counts (§7).
    boilerplate = boilerplate_lines(conn)
    for file_id, messages in staged:
        n = _insert_chunks(conn, file_id, messages, boilerplate, stats)
        conn.execute("UPDATE files SET n_chunks = ? WHERE id = ?", (n, file_id))
        stats.chunks += n
    conn.commit()
    conn.execute("INSERT INTO meta(key,value) VALUES('last_index', ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (now,))
    conn.execute("INSERT INTO meta(key,value) VALUES('root', ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(root),))
    conn.commit()
    return stats


def stats(conn: sqlite3.Connection) -> dict:
    def one(sql: str, default=0):
        row = conn.execute(sql).fetchone()
        return (row[0] if row and row[0] is not None else default)

    return {
        "files": one("SELECT COUNT(*) FROM files"),
        "sessions": one("SELECT COUNT(DISTINCT session_id) FROM chunks"),
        "projects": one("SELECT COUNT(DISTINCT project) FROM chunks"),
        "chunks": one("SELECT COUNT(*) FROM chunks"),
        "chars": one("SELECT SUM(n_chars) FROM chunks"),
        "mean_chunk_chars": round(one("SELECT AVG(n_chars) FROM chunks", 0.0) or 0.0, 1),
        "boilerplate_lines": one(
            f"SELECT COUNT(*) FROM line_freq WHERE n >= {BOILERPLATE_MIN_MESSAGES}"
        ),
        "last_index": (conn.execute("SELECT value FROM meta WHERE key='last_index'").fetchone()
                       or [None])[0],
        "root": (conn.execute("SELECT value FROM meta WHERE key='root'").fetchone() or [None])[0],
    }
