"""Query path: BM25 over the chunk store.

BM25 is the primary retriever, not a straw-man baseline: it reaches MACRO
recall@10 0.843 at pool scale and 0.709 on the full corpus, and every dense
model measured is significantly worse alone (§12 finding 1).

Phase 2 adds a dense retriever and fuses the two with reciprocal-rank fusion.
The seam for that is exactly this module's `Hit` shape and `search()`
signature - a second retriever returns the same `(chunk_id, rank, score)` and a
fusion function combines two ranked lists. Nothing more is built for it here.

No confidence gate is applied. Absolute score carries no usable signal
(AUC 0.551, F6); the rank-1-to-rank-10 margin does (AUC 0.724), so `search()`
reports that margin as a diagnostic and lets the caller judge. Retrieval is
never injected automatically (§13.2) - it is asked for.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

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
    hits: list[Hit] = field(default_factory=list)

    @property
    def margin(self) -> float | None:
        """Rank-1 to rank-10 score margin - the only signal F6 found usable.

        Returned for the caller to look at. It is deliberately not thresholded
        here: nobody has built or validated that classifier (open question O13).
        """
        if len(self.hits) < 2:
            return None
        last = self.hits[min(9, len(self.hits) - 1)]
        return round(self.hits[0].score - last.score, 4)


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
        since: str | None = None,
        include_sidechain: bool = True,
    ) -> Result:
        where = ["chunks_fts MATCH ?"]
        params: list[object] = [match_expression(question)]
        if project:
            where.append("c.project LIKE ?")
            params.append(f"%{project}%")
        if role:
            where.append("c.role = ?")
            params.append(role)
        if since:
            where.append("c.ts >= ?")
            params.append(since)
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
        return Result(query=question, hits=hits)
