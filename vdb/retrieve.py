"""Query path: BM25 over the chunk store.

BM25 is the primary retriever, not a straw-man baseline: it reaches MACRO
recall@10 0.843 at pool scale and 0.709 on the full corpus, and every dense
model measured is significantly worse alone (§12 finding 1).

Phase 2 adds a dense retriever and fuses the two with reciprocal-rank fusion.
The seam for that is exactly this module's `Hit` shape and `search()`
signature - a second retriever returns the same `(chunk_id, rank, score)` and a
fusion function combines two ranked lists. Nothing more is built for it here.

The same seam also carries the implicit-feedback score nudge
(data/vdbfeedback/report.md §5.3, outside this repo): a capped additive
adjustment from `store.chunk_feedback`, gated by `store.nudge_active()` and
shipped OFF (see that function's docstring for the three conditions it
requires). Below the gate it never reads `chunk_feedback` at all.

No confidence gate is applied - and there is not a strong enough measured
signal to build one on. Report F6 found absolute similarity carries no usable
signal (AUC 0.551) and the rank-1-to-rank-10 margin does better (AUC 0.724) -
but that was measured for a bounded dense cosine score, not for this
retriever. Re-measured directly for BM25's own unbounded `-bm25()` score (the
`vdbtray` chunk-granularity harness, reused for this question): AUC(top1
score) = 0.530, 95% CI [0.480, 0.582] (still no usable signal, same
conclusion as F6) and AUC(margin) = 0.645, 95% CI [0.598, 0.690] (real, but
distinctly weaker than the dense 0.724 - the hit and miss margin distributions
overlap substantially). `search()` reports the margin as a diagnostic and lets
the caller judge; it is deliberately not thresholded into a verdict, because
0.645 is not strong enough evidence to hang a hard cutoff on without it
misfiring often (`vdbqual` O13 remains open). Retrieval is never injected
automatically (§13.2) - it is asked for.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from . import store as store_mod

_WORD = re.compile(r"[0-9A-Za-z_]+", re.UNICODE)
MAX_QUERY_TERMS = 64


@dataclass
class Hit:
    chunk_id: int
    rank: int
    score: float
    text: str
    session_id: str | None = None
    project: str | None = None
    role: str | None = None
    ts: str | None = None
    part: int = 0
    n_parts: int = 1
    source: str | None = None
    is_sidechain: bool = False

    def provenance(self) -> str:
        when = (self.ts or "")[:19].replace("T", " ")
        part = f" part {self.part + 1}/{self.n_parts}" if self.n_parts > 1 else ""
        side = " (sidechain)" if self.is_sidechain else ""
        return f"{self.project or '?'} · {self.role or '?'} · {when or 'undated'}{part}{side}"


@dataclass
class Result:
    query: str
    k_requested: int = 10
    hits: list[Hit] = field(default_factory=list)
    nudge_applied: bool = False

    @property
    def margin(self) -> float | None:
        """Rank-1 to rank-10 (or rank-1 to the last hit, if fewer) score margin.

        The only signal measured to beat the raw top score - see this module's
        docstring for the BM25-specific AUC numbers. Returned for the caller to
        look at; deliberately not thresholded into a verdict, because the
        measured signal (AUC 0.645) isn't strong enough to hang a hard cutoff
        on without it misfiring often (open question O13).
        """
        if len(self.hits) < 2:
            return None
        last = self.hits[min(9, len(self.hits) - 1)]
        return round(self.hits[0].score - last.score, 4)

    @property
    def weak_signal(self) -> bool:
        """A calibration-free, structural weak-signal flag.

        True when there isn't enough returned material to say anything with
        confidence: no hits, only one hit (the margin is undefined), or fewer
        hits than requested (the index has thin coverage for this query). This
        is deliberately NOT based on a margin or score cutoff - the measured
        AUC (0.645) isn't strong enough evidence to invent a numeric threshold
        that would misfire silently. See this module's docstring.
        """
        return len(self.hits) < 2 or len(self.hits) < self.k_requested


def match_expression(question: str) -> str:
    """Turn a natural-language question into an FTS5 MATCH expression.

    FTS5's implicit operator is AND, which would make any long question return
    nothing; BM25 ranking wants OR semantics with the scoring doing the work.
    Every term is quoted so that punctuation and FTS5 operator words in the
    question cannot be interpreted as syntax.
    """
    terms = [t.lower() for t in _WORD.findall(question)]
    if not terms:
        raise ValueError("query contains no searchable terms")
    seen, uniq = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " OR ".join(f'"{t}"' for t in uniq[:MAX_QUERY_TERMS])


class BM25Retriever:
    """Okapi BM25 (k1=1.2, b=0.75) via SQLite FTS5, over the chunk store."""

    name = "bm25"

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def search(
        self,
        question: str,
        k: int = 10,
        project: str | None = None,
        role: str | None = None,
        session: str | None = None,
        since: str | None = None,
        until: str | None = None,
        include_sidechain: bool = True,
    ) -> Result:
        """Metadata filters narrow the candidate set before BM25 ranks it.

        `project` and `session` are substring matches (a session id is a full
        UUID nobody types from memory); `role` is exact (`user` / `assistant`
        / `memory` - the corpus's speaker dimension); `since` / `until` are
        ISO-8601 timestamp bounds, inclusive. This is the mechanism that makes
        the tool cheaper than reading a whole conversation - it is load-bearing,
        not a convenience (§13.1 "the captain's chapter/page/paragraph").
        """
        where = ["chunks_fts MATCH ?"]
        params: list[object] = [match_expression(question)]
        if project:
            where.append("c.project LIKE ?")
            params.append(f"%{project}%")
        if role:
            where.append("c.role = ?")
            params.append(role)
        if session:
            where.append("c.session_id LIKE ?")
            params.append(f"%{session}%")
        if since:
            where.append("c.ts >= ?")
            params.append(since)
        if until:
            where.append("c.ts <= ?")
            params.append(until)
        if not include_sidechain:
            where.append("c.is_sidechain = 0")
        params.append(k)

        sql = (
            "SELECT c.id, c.session_id, c.project, c.role, c.ts, c.part, c.n_parts,"
            "       c.is_sidechain, f.path AS source, chunks_fts.text AS text,"
            "       -bm25(chunks_fts) AS score "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "JOIN files f ON f.id = c.file_id "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY score DESC LIMIT ?"
        )
        hits = [
            Hit(
                chunk_id=row["id"],
                rank=i,
                score=round(row["score"], 4),
                text=row["text"],
                session_id=row["session_id"],
                project=row["project"],
                role=row["role"],
                ts=row["ts"],
                part=row["part"],
                n_parts=row["n_parts"],
                source=row["source"],
                is_sidechain=bool(row["is_sidechain"]),
            )
            for i, row in enumerate(self.conn.execute(sql, params), start=1)
        ]

        # Implicit-feedback score nudge (data/vdbfeedback/report.md §5.3), the
        # seam this module's own docstring reserves for a second retriever.
        # `nudge_active()` is the real runtime gate (report §6.3): below the
        # 300-label threshold, or with the flag off, or with no recorded
        # passing regression check, this block does not run at all - the
        # nudge is provably inert, not just defaulted off. When it does run,
        # it only reorders the already-fetched top-k (never pulls in
        # candidates BM25 itself did not return), keeping its influence
        # bounded as documented in `store.NUDGE_WEIGHT`/`NUDGE_CAP`.
        nudged = False
        if hits and store_mod.nudge_active(self.conn):
            nudged = True
            for hit in hits:
                hit.score = round(hit.score + store_mod.score_nudge(self.conn, hit.chunk_id), 4)
            hits.sort(key=lambda h: h.score, reverse=True)
            for i, hit in enumerate(hits, start=1):
                hit.rank = i

        return Result(query=question, k_requested=k, hits=hits, nudge_applied=nudged)
