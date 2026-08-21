"""Chunk store and BM25 index, on SQLite + FTS5.

Why SQLite/FTS5 and nothing else: FTS5's `bm25()` is Okapi BM25 with k1=1.2 and
b=0.75 - the same parameterisation the study measured (§B.5) - it is in the
Python standard library, it needs no model and no network, and it gives
incremental insert/delete for free. Phase 1 therefore has zero dependencies.

Phase 2 (dense index + reciprocal-rank fusion, `vdb/dense.py` + `vdb/retrieve.py`) reuses
`chunks.id` as the vector index's own key: `chunk_embeddings` (below) is one row per chunk,
addressed by that same stable integer, so the dense side never needs its own id scheme or a
separate vector file. There is no plugin machinery here for a second future retriever - the
seam was, and remains, exactly `retrieve.BM25Retriever.search()`'s `Result`/`Hit` shape.

The index contains the captain's prose. It is exactly as sensitive as the
transcripts (F9), so the database file is created 0600 inside a 0700 directory.

Implicit-feedback learning loop (data/vdbfeedback/report.md, outside this repo):
`query_log`/`feedback_citation`/`feedback_label`/`chunk_feedback` live in this
same database - one artefact to protect, not a second file. See the functions
below `stats()` and `AGENTS.md` for the two-part (global + per-chunk) runtime
gate that keeps the score-affecting nudge inert until it is explicitly earned.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import chunk as chunk_mod
from . import ingest as ingest_mod
from .clean import candidate_lines, trim_boilerplate

SCHEMA_VERSION = "2"

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

-- Implicit-feedback learning loop (data/vdbfeedback/report.md §5.1). Lives in
-- this same gitignored 0600 database, not a new file - runtime state, never
-- committed, exactly like the index itself.

-- One row per `vdb query` call. `id` IS the query_id handed back to the
-- caller. `held_out` is written once, at insert time, from a hash of `id` -
-- decided before any label can exist, so it can never be gamed by which
-- queries happen to get citations (report §5.1, §6.2b).
CREATE TABLE IF NOT EXISTS query_log (
    id              INTEGER PRIMARY KEY,
    ts              TEXT NOT NULL,
    query_text      TEXT NOT NULL,
    filters_json    TEXT NOT NULL,
    k_requested     INTEGER NOT NULL,
    hit_chunk_ids   TEXT NOT NULL,
    hit_scores      TEXT NOT NULL,
    margin          REAL,
    weak_signal     INTEGER NOT NULL,
    caller_session  TEXT,
    held_out        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS query_log_by_ts ON query_log(ts);

-- Written when (if) the calling agent reports which returned chunk(s) it
-- used. `used_chunk_ids = '[]'` is a valid, meaningful "queried, used
-- nothing" value (report §4.2) - a soft miss, not the absence of a signal.
-- One citation per query_id: a later `vdb feedback` call for the same
-- query_id replaces the prior one rather than accumulating duplicates.
CREATE TABLE IF NOT EXISTS feedback_citation (
    query_id        INTEGER NOT NULL UNIQUE REFERENCES query_log(id),
    used_chunk_ids  TEXT NOT NULL,
    ts              TEXT NOT NULL
);

-- Written later, by the background label-extraction pass (`vdb label`).
-- `label` is one of confirm / correction / mixed / none. Only confirm and
-- correction are ever "high-confidence, single-class" (report §6.1) - mixed
-- is recorded but never feeds `chunk_feedback`.
CREATE TABLE IF NOT EXISTS feedback_label (
    query_id         INTEGER NOT NULL UNIQUE REFERENCES query_log(id),
    label            TEXT NOT NULL,
    label_method     TEXT NOT NULL,
    turns_downstream INTEGER NOT NULL,
    ts               TEXT NOT NULL
);

-- The score-affecting state (report §5.3). Populated by the label pass;
-- read by `BM25Retriever.search()` ONLY when `nudge_active()` says the
-- runtime gate (flag + 300-label threshold + recorded regression check,
-- report §6.3) is satisfied. `positive_n`/`negative_n` accumulate
-- unboundedly here; the cap (§6.2a) is applied at read time in
-- `score_nudge()` so raising the cap later needs no backfill.
CREATE TABLE IF NOT EXISTS chunk_feedback (
    chunk_id     INTEGER PRIMARY KEY,
    positive_n   INTEGER NOT NULL DEFAULT 0,
    negative_n   INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT
);

-- Phase 2 dense index (`vdb/dense.py`). One row per chunk per embedding model,
-- addressed by the same stable `chunks.id` the BM25 side already uses - no
-- separate vector file (report `vdbscout` "storage/index options": brute-force
-- over SQLite BLOBs is comparable-or-faster than sqlite-vec at this corpus's
-- scale, with no extra compiled dependency). `vector` is a packed little-endian
-- float32 array (`dense.pack_vector`/`unpack_vector`), L2-normalized at write
-- time so cosine similarity is a plain dot product at search time. `model`
-- exists so a future model change doesn't have to guess which rows are stale.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id     INTEGER PRIMARY KEY REFERENCES chunks(id),
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    indexed_at   TEXT NOT NULL
);
"""

# The GLOBAL half of the nudge gate: `nudge_active()` stays False below this
# many corpus-wide high-confidence, cited, non-held-out labels, regardless of
# the feature flag (report §6.3). The report's own 300 was "a round number in
# the same range vdbqual's smallest reliable family (C, n=60) and
# vdbaccuracy's bootstrap intervals treat as the floor for saying anything
# with confidence" (§6.3/§7) - 300 was chosen when the *only* gate was this
# global count, i.e. it had to carry both "the check is worth running" and
# "the system broadly behaves". Now that a second, PER-CHUNK gate
# (`NUDGE_PER_CHUNK_MIN`, below) separately carries per-passage confidence,
# this global number only needs to answer "is there enough accumulated data
# for the §6.2c regression check to be meaningful at all" - and the report's
# own citation for 300 already names a smaller number that this domain treats
# as trustworthy for that: vdbqual's smallest reliable family, C, n=60. 60 is
# used here rather than re-deriving a new number, because it is not a guess -
# it is a floor this project's own eval methodology already relies on.
NUDGE_LABEL_THRESHOLD = 60

# The PER-CHUNK half of the nudge gate: a chunk's own score nudge is applied
# only once THAT chunk has accumulated at least this many of its own
# high-confidence, cited, non-held-out labels (`positive_n + negative_n`).
# Chunks below this are not nudged even when the global gate above is fully
# satisfied - `score_nudge()` returns 0.0 for them (report's per-passage
# evidence requirement, task brief "Change 2").
#
# This is a confidence floor, not a "wait for any signal at all" gate: under
# `score_nudge()`'s own `NUDGE_WEIGHT * log(1 + min(n, NUDGE_CAP))` formula, a
# single citation (n=1) already produces a small nonzero nudge
# (0.15 * log(2) ~= 0.104) - so the threshold's job is not to suppress a
# first data point, it is to require more than one before treating it as
# reliable. §3.4's hand-validation found that even after the "clean,
# single-class" filter this project already applies, only ~70% of labels
# survive as genuinely clean - i.e. roughly a 30% chance any single
# high-confidence label is still wrong. Three independent citations pointing
# the same direction is the smallest multiplicity where "wrong 30% of the
# time" no longer plausibly explains the accumulated sign of the nudge, while
# staying far below `NUDGE_CAP` (20) so it works purely as a floor, not as a
# second cap.
NUDGE_PER_CHUNK_MIN = 3

# Inside `score_nudge()`'s log(1 + min(n, CAP)): bounds how far a single
# chunk's score can move even under a burst of repeated labels (report §6.2a
# - directly motivated by §3.2's single templated string producing 136
# near-identical raw events).
NUDGE_CAP = 20

# Small enough that the nudge can only reorder near-ties within BM25's own
# top-k, never overwhelm the primary signal (report §5.3's "bounded
# influence is the drift safeguard, not an afterthought").
NUDGE_WEIGHT = 0.15


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
    """Drop every indexed artefact, keeping the schema.

    Clears `chunk_embeddings` too - `chunks.id` is reassigned on a rebuild, so leaving old
    embeddings behind would silently attach a stale vector to whatever chunk reuses that id.
    """
    conn.executescript(
        "DELETE FROM chunks_fts; DELETE FROM chunk_embeddings; DELETE FROM chunks;"
        " DELETE FROM file_line; DELETE FROM line_freq; DELETE FROM files;"
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
    conn.execute(
        "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)",
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
        "citation_compliance": citation_compliance(conn),
        "dense_embedded": one("SELECT COUNT(*) FROM chunk_embeddings"),
    }


CITATION_COMPLIANCE_WINDOW = 200


def citation_compliance(conn: sqlite3.Connection, window: int = CITATION_COMPLIANCE_WINDOW) -> dict:
    """What fraction of recent `vdb query` calls got a matching `vdb feedback` citation.

    Change 1 of the task brief: citation cannot be forced across two separate
    CLI invocations, so this is the visible compliance signal that lets drift
    be caught instead of silently accepted (`vdb stats`). Looks at the most
    recent `window` query_log rows (by id, i.e. by call order) - recent
    behaviour, not all-time history, is what a calling agent's current
    compliance actually looks like.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n, "
        "SUM(CASE WHEN fc.query_id IS NOT NULL THEN 1 ELSE 0 END) AS cited "
        "FROM (SELECT id FROM query_log ORDER BY id DESC LIMIT ?) ql "
        "LEFT JOIN feedback_citation fc ON fc.query_id = ql.id",
        (window,),
    ).fetchone()
    n = row["n"] or 0
    cited = row["cited"] or 0
    return {
        "window": window,
        "queries": n,
        "cited": cited,
        "uncited": n - cited,
        "citation_rate": round(cited / n, 4) if n else None,
    }


# --- Implicit-feedback learning loop (data/vdbfeedback/report.md) ---------


def _held_out(query_id: int) -> int:
    """Deterministic ~10% split, decided by the query_id alone (report §5.1).

    Hashing rather than `query_id % 10` is belt-and-suspenders against any
    future id-assignment scheme (e.g. a UUID) that would make a raw modulo
    non-uniform; either way the point is that no label, citation, or
    downstream event can influence the split.
    """
    h = hashlib.sha1(str(query_id).encode("utf-8")).hexdigest()
    return 1 if int(h[:8], 16) % 10 == 0 else 0


def log_query(
    conn: sqlite3.Connection,
    *,
    query_text: str,
    filters: dict,
    k_requested: int,
    hit_chunk_ids: list[int],
    hit_scores: list[float],
    margin: float | None,
    weak_signal: bool,
    caller_session: str | None = None,
) -> int:
    """Record one `vdb query` call. Returns the new `query_id`.

    Zero effect on retrieval results - this only writes rows (report §8 step 1).
    """
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO query_log(ts, query_text, filters_json, k_requested, hit_chunk_ids,"
        " hit_scores, margin, weak_signal, caller_session, held_out)"
        " VALUES(?,?,?,?,?,?,?,?,?,0)",
        (
            ts,
            query_text,
            json.dumps(filters),
            k_requested,
            json.dumps(hit_chunk_ids),
            json.dumps(hit_scores),
            margin,
            int(weak_signal),
            caller_session,
        ),
    )
    query_id = cur.lastrowid
    conn.execute(
        "UPDATE query_log SET held_out = ? WHERE id = ?", (_held_out(query_id), query_id)
    )
    conn.commit()
    return query_id


def record_citation(conn: sqlite3.Connection, query_id: int, used_chunk_ids: list[int]) -> None:
    """Record which returned chunk(s) the caller used. `[]` is valid (§4.2).

    Idempotent per query_id: citing again for the same query_id replaces the
    prior citation rather than accumulating duplicates.
    """
    if not conn.execute("SELECT 1 FROM query_log WHERE id = ?", (query_id,)).fetchone():
        raise ValueError(f"unknown query_id {query_id}")
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO feedback_citation(query_id, used_chunk_ids, ts) VALUES(?,?,?) "
        "ON CONFLICT(query_id) DO UPDATE SET used_chunk_ids = excluded.used_chunk_ids, ts = excluded.ts",
        (query_id, json.dumps(sorted(set(used_chunk_ids))), ts),
    )
    conn.commit()


def write_label(
    conn: sqlite3.Connection, query_id: int, label: str, label_method: str, turns_downstream: int
) -> None:
    """Record a label found by the background pass (`vdb label`). Not committed here."""
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO feedback_label(query_id, label, label_method, turns_downstream, ts)"
        " VALUES(?,?,?,?,?) "
        "ON CONFLICT(query_id) DO UPDATE SET label=excluded.label, label_method=excluded.label_method,"
        " turns_downstream=excluded.turns_downstream, ts=excluded.ts",
        (query_id, label, label_method, turns_downstream, ts),
    )


def apply_feedback_to_chunks(conn: sqlite3.Connection, chunk_ids: list[int], label: str) -> None:
    """Increment `chunk_feedback` counts for a clean confirm/correction label.

    Only called for held_out=0 queries with a non-empty citation and a clean
    single-class label (§6.1's compounded filters) - callers, not this
    function, enforce that. Counts here are unbounded; the CAP (§6.2a) is
    applied at read time in `score_nudge()`.
    """
    if label not in ("confirm", "correction"):
        raise ValueError(f"label {label!r} is not score-affecting")
    ts = datetime.now(timezone.utc).isoformat()
    col = "positive_n" if label == "confirm" else "negative_n"
    for chunk_id in chunk_ids:
        conn.execute(
            f"INSERT INTO chunk_feedback(chunk_id, {col}, last_updated) VALUES(?, 1, ?) "
            f"ON CONFLICT(chunk_id) DO UPDATE SET {col} = {col} + 1, last_updated = excluded.last_updated",
            (chunk_id, ts),
        )


def chunk_label_count(conn: sqlite3.Connection, chunk_id: int) -> int:
    """This chunk's own accumulated evidence (`positive_n + negative_n`).

    What `NUDGE_PER_CHUNK_MIN` is measured against - exposed separately from
    `score_nudge()` so callers (tests, `vdb nudge`) can inspect the per-chunk
    gate without recomputing the score math.
    """
    row = conn.execute(
        "SELECT positive_n, negative_n FROM chunk_feedback WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    if not row:
        return 0
    return row["positive_n"] + row["negative_n"]


def score_nudge(conn: sqlite3.Connection, chunk_id: int) -> float:
    """The capped additive BM25 score adjustment for one chunk (report §5.3).

    `score' = score + NUDGE_WEIGHT * (log(1 + min(pos, CAP)) - log(1 + min(neg, CAP)))`

    Gated per-chunk (task brief "Change 2"): a chunk with fewer than
    `NUDGE_PER_CHUNK_MIN` of its own labels returns 0.0 here - `score' =
    score`, exactly today's shipped behaviour - even when the caller has
    already confirmed the global `nudge_active()` gate is satisfied. Neither
    gate replaces the other; this function only ever enforces the per-chunk
    half.
    """
    row = conn.execute(
        "SELECT positive_n, negative_n FROM chunk_feedback WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    if not row:
        return 0.0
    if row["positive_n"] + row["negative_n"] < NUDGE_PER_CHUNK_MIN:
        return 0.0
    pos = min(row["positive_n"], NUDGE_CAP)
    neg = min(row["negative_n"], NUDGE_CAP)
    return NUDGE_WEIGHT * (math.log1p(pos) - math.log1p(neg))


def qualifying_label_count(conn: sqlite3.Connection) -> int:
    """High-confidence, cited, score-affecting labels (§6.1, §6.3's 300 threshold).

    Held-out queries are excluded even if labelled - they must stay an
    untouched eval set (§6.2b), so they never count toward "volume to turn
    the nudge on" either.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM feedback_label fl "
        "JOIN feedback_citation fc ON fc.query_id = fl.query_id "
        "JOIN query_log ql ON ql.id = fl.query_id "
        "WHERE fl.label IN ('confirm', 'correction') "
        "AND fc.used_chunk_ids != '[]' AND ql.held_out = 0"
    ).fetchone()
    return row[0]


def nudge_flag_enabled(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM meta WHERE key = 'feedback_nudge_enabled'").fetchone()
    return bool(row and row["value"] == "1")


def set_nudge_flag(conn: sqlite3.Connection, enabled: bool) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('feedback_nudge_enabled', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("1" if enabled else "0",),
    )
    conn.commit()


def regression_check_status(conn: sqlite3.Connection) -> dict | None:
    """The most recently recorded §6.2c regression-check outcome, if any."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'nudge_regression_check'"
    ).fetchone()
    if not row:
        return None
    return json.loads(row["value"])


def record_regression_check(conn: sqlite3.Connection, passed: bool, notes: str = "") -> None:
    """Record a §6.2c regression-check outcome. Never fabricate `passed=True`.

    This does not itself run the check - `vdbqual`/`vdbtray`/`vdbaccuracy`'s
    static eval harnesses live outside this repo (their own Appendix B); an
    operator runs the nudge on vs. off and records the verdict here. See
    `scripts/nudge-regression-check.md`.
    """
    payload = {
        "passed": bool(passed),
        "notes": notes,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('nudge_regression_check', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (json.dumps(payload),),
    )
    conn.commit()


def nudge_active(conn: sqlite3.Connection) -> bool:
    """The real runtime gate (report §6.3) - not documentation asking someone to remember.

    This is the GLOBAL half of the gate: all three must hold - the operator
    flag, the `NUDGE_LABEL_THRESHOLD` corpus-wide volume floor, and an
    explicitly recorded passing regression check. Below any of these,
    `BM25Retriever.search()` must not read `chunk_feedback` at all. Even when
    this returns True, an individual chunk is only nudged once it separately
    clears the PER-CHUNK gate - see `score_nudge()`'s `NUDGE_PER_CHUNK_MIN`
    check. Neither gate substitutes for the other.
    """
    if not nudge_flag_enabled(conn):
        return False
    if qualifying_label_count(conn) < NUDGE_LABEL_THRESHOLD:
        return False
    check = regression_check_status(conn)
    return bool(check and check.get("passed") is True)


def reset_feedback_labels(conn: sqlite3.Connection) -> None:
    """Clear the learning signal without touching `query_log` history or the chunk index.

    `feedback_label` and `chunk_feedback` are derived state; `query_log` and
    `feedback_citation` are the append-only record of what actually happened
    and are not touched here (report §5.1).
    """
    conn.executescript("DELETE FROM feedback_label; DELETE FROM chunk_feedback;")
    conn.commit()
