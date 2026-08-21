"""Query path: BM25, dense (`multilingual-e5-large`), and reciprocal-rank fusion of the two.

BM25 is the primary retriever, not a straw-man baseline: it reaches MACRO
recall@10 0.843 at pool scale and 0.709 on the full corpus, and every dense
model measured is significantly worse alone (§12 finding 1).

Phase 2 (`data/decisions/vdbqual-decision-embedding-model-cost.md`) adds `DenseRetriever`
(`multilingual-e5-large` via `vdb/dense.py`) and `HybridRetriever`, which fuses BM25's and
the dense retriever's rankings with reciprocal-rank fusion (`reciprocal_rank_fusion`,
`vdbqual` §11.4's exact measured mechanism: RRF, k=60 - not a score-blend, not a learned
combiner, and no reranker, which that same section measured as making the best configuration
*worse*). All three retrievers return the same `Hit`/`Result` shape - the seam this module's
docstring always reserved for a second retriever - and all three apply metadata filters
identically via the shared `_metadata_filters()` (`vdbaccuracy` §3: pre-search filtering is
the single biggest recall lever measured in this project, and it has to narrow every
retriever's candidate set, not just one).

**Two decisions phase 2 makes explicitly, not by default** (see README "Feedback" and
"Confidence" for the full reasoning):
- The feedback nudge stays BM25-only. `DenseRetriever` never reads `chunk_feedback` - its
  evidence was only ever collected against BM25-only rankings. `HybridRetriever` doesn't add
  a nudge of its own either; when the global gate (`store.nudge_active()`) is active, BM25's
  own nudge still applies to the BM25 ranking that feeds into fusion, because that ranking
  comes from an ordinary `BM25Retriever.search()` call - fusion doesn't know or care that the
  scores it's fusing were nudged.
- `Result.confidence` on any non-BM25-only ranking (dense-only or fused/hybrid) is
  `UNCALIBRATED`, never the calibrated bands below - see `Result.confidence`'s docstring.

BM25's own feedback score nudge (data/vdbfeedback/report.md §5.3, outside this repo) lives in
`BM25Retriever.search()` below: a capped additive adjustment from `store.chunk_feedback`,
gated by TWO conditions that must both hold - `store.nudge_active()` (global: flag +
corpus-wide volume floor + recorded regression check) and, per hit, `store.score_nudge()`'s
own `NUDGE_PER_CHUNK_MIN` check (that specific chunk's own accumulated evidence). Shipped OFF
by the global gate (see `store.nudge_active()`'s docstring). Below the global gate this block
never reads `chunk_feedback` at all; above it, `score_nudge()` still returns 0.0 for any chunk
that hasn't individually earned it.

Report F6 found absolute similarity carries no usable signal (AUC 0.551) and
the rank-1-to-rank-10 margin does better (AUC 0.724) - but that was measured
for a bounded dense cosine score, not for this retriever. Re-measured directly
for BM25's own unbounded `-bm25()` score (the `vdbtray` chunk-granularity
harness, reused for this question): AUC(top1 score) = 0.530, 95% CI
[0.480, 0.582] (still no usable signal, same conclusion as F6) and
AUC(margin) = 0.645, 95% CI [0.598, 0.690] (real, but distinctly weaker than
the dense 0.724 - the hit and miss margin distributions overlap
substantially).

`Result.confidence` (vdbconfidence task, O13) turns that margin into a
calibrated three-way label instead of a fixed threshold - see this module's
`confidence_band()` docstring for the calibration measurement (four query
families rebuilt against the live corpus, 565 scored queries) and for the
real family-pooling confound that measurement surfaced along the way. It is a
label on top of the existing ranking, not a change to it - `search()`'s
ordering and scores are unaffected. Retrieval is never injected automatically
(§13.2) - it is asked for.
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
    # Which retriever produced this ranking - "bm25" (default, phase 1's only retriever),
    # "dense", or "hybrid" (RRF-fused). Drives `confidence` below: the calibrated bands were
    # fit on BM25 margins specifically and must not be reused for any other ranking's margin.
    retriever: str = "bm25"

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
        that would misfire silently. See this module's docstring. Independent
        of `confidence` below - a full result set can still be low-confidence,
        and this flag is untouched by that addition.
        """
        return len(self.hits) < 2 or len(self.hits) < self.k_requested

    @property
    def confidence(self) -> str:
        """The calibrated three-way confidence label for this result's shape.

        Thin wrapper around `confidence_band(self.margin)` - see that
        function's docstring for the calibration measurement, the bands'
        measured hit rates, and why three bands rather than a numeric score -
        for `retriever == "bm25"` results ONLY.

        Phase 2 design decision (task brief "B", README "Confidence"): the calibration behind
        `confidence_band()` was fit on raw BM25 margins, on query families Q and C only
        (`CONFIDENCE_LOW_MARGIN`/`CONFIDENCE_HIGH_MARGIN`'s docstring). Once a ranking comes
        from `DenseRetriever` or `HybridRetriever`, `margin` is a different quantity - a
        cosine-similarity gap or an RRF score gap, not a BM25 score gap - and applying the
        BM25-fit thresholds to it would silently mislabel confidence with numbers that were
        never validated for that case (exactly the overselling-reliability failure this
        project's confidence gate exists to prevent). Recalibrating for hybrid mode was
        judged out of scope for this task (README "Confidence" - follow-up work); until then
        this returns `UNCALIBRATED` for anything that isn't a plain BM25 result, never a
        guessed or reused band.
        """
        if self.retriever != "bm25":
            return UNCALIBRATED
        return confidence_band(self.margin)


CONFIDENT = "confident"
UNCERTAIN = "uncertain"
LOW_CONFIDENCE = "low_confidence"

# Phase 2 (task brief design decision B): the label for any Result whose ranking did not come
# from BM25 alone (`Result.retriever in {"dense", "hybrid"}`). Deliberately not one of the
# three calibrated bands above - see `Result.confidence`'s docstring.
UNCALIBRATED = "uncalibrated"

# Calibrated on real per-query top-10 score shapes (vdbconfidence task, O13):
# vdbqual Appendix B's four query families (M/Q/C/T) were rebuilt against the
# live corpus at the shipped msg1000 BM25 config and scored - 565 queries,
# pooled hit@10 rate 68.5% (M 137/1.000, Q 228/0.465, C 60/0.333, T 140/0.886;
# family sizes replicate vdbtray's harness almost exactly - vdbtray got
# 137/259/60/140 on its own snapshot of this same corpus).
#
# Several shape descriptors were compared by AUC before picking one - not a
# rubber stamp of `Result.margin`: the plain rank1-rank10 margin already used
# there (AUC 0.647), the raw rank1-rank2 gap (0.612), that gap normalised by
# the top score (0.560), a normalised score-decay-curve area (0.611), and the
# rank-1 z-score against the rank2-10 tail's own mean/sd (0.462, WORSE than
# chance). Plain margin won; nothing tried beat it.
#
# That AUC was NOT computed on all 565 queries pooled - doing so inverts the
# sign (pooled AUC 0.385, margin appearing to predict a MISS). Diagnosed, not
# papered over: family M is always a hit (no negative examples, contributes
# no discriminative information) and family T's gold is session-level ("any
# chunk from this session counts") - generous enough that a genuine hit does
# not need a peaked score curve. T's own within-family AUC is a strong 0.850,
# but its typical hit margin sits far below Q/C's typical MISS margin, so
# mixing the four families flips the pooled sign even though the relationship
# is positive within every one of them. This is Simpson's paradox from
# combining populations with different baseline score scales, not a bug in
# the harness - and it is itself the reason this gate is calibrated on
# families Q and C only (n=288, hit rate 43.8%): the two families whose gold
# means "this specific passage is the answer", not "something from the right
# session showed up" - vdbqual §2.2 names Q the family it weights most when
# families disagree, for exactly that specific-passage-gold reason; C's own
# gold (§2.3) is built the same way even though §2.2 doesn't say so about C
# by name. AUC(margin) on that population = 0.647, 95% CI
# [0.584, 0.709] - matching vdbtray's own pooled BM25 margin AUC (0.645,
# n=596) closely, which is reassuring given the two measurements used
# different query samples on a corpus that kept growing between them.
#
# Practical upshot: this gate was calibrated on, and is most trustworthy for,
# specific-answer-recall-shaped queries. A broad "have we talked about X"
# query is closer to family T's shape, where this same margin threshold does
# not carry the same meaning - stated plainly in the README/CLI help, not
# just here.
#
# Cut points are the terciles of that Q+C margin distribution, not round
# numbers - each band's measured hit rate (3,000-resample bootstrap 95% CI):
#   margin <  75.0            -> low_confidence:  28.1% hit  [18.9%, 37.4%]
#   75.0 <= margin < 126.0    -> uncertain:        44.7% hit  [34.7%, 54.4%]
#   margin >= 126.0           -> confident:        58.2% hit  [48.1%, 67.7%]
# Adjacent bands' intervals overlap - this is a real but noisy signal, same
# conclusion as the AUC. Three bands, not a numeric score: the calibration
# does not support finer distinctions without manufacturing false precision
# (see README "Confidence" before trusting a number this measurement can't
# back up).
CONFIDENCE_LOW_MARGIN = 75.0
CONFIDENCE_HIGH_MARGIN = 126.0


def confidence_band(margin: float | None) -> str:
    """Calibrated three-way confidence label for one query's score margin.

    Not a verdict and not a substitute for reading the passages - even the
    `confident` band's measured hit rate (58.2%) is well under certainty. See
    the module-level comment above `CONFIDENCE_LOW_MARGIN` for the
    calibration this is built on, including the family-pooling confound it
    surfaced and why it is fit on families Q and C rather than the whole
    corpus of query shapes.

    `margin=None` (fewer than two hits - the same condition `weak_signal`
    already flags) has no shape to compare and is `low_confidence` by
    definition, not by extrapolation.
    """
    if margin is None or margin < CONFIDENCE_LOW_MARGIN:
        return LOW_CONFIDENCE
    if margin < CONFIDENCE_HIGH_MARGIN:
        return UNCERTAIN
    return CONFIDENT


# The measured hit rate behind each band, for callers to print alongside the
# label so nobody has to take "confident" on faith - the whole point of
# calibrating this rather than shipping an unqualified word.
CONFIDENCE_EXPLANATION = {
    CONFIDENT: (
        "measured hit rate 58% (95% CI 48-68%) on the calibration set - real "
        "signal, still well under certainty; read the passage, don't skip it"
    ),
    UNCERTAIN: (
        "measured hit rate 45% (95% CI 35-54%) - close to a coin flip; the "
        "score shape doesn't clearly look like a hit or a miss"
    ),
    LOW_CONFIDENCE: (
        "measured hit rate 28% (95% CI 19-37%) on queries with this shape - "
        "usually a miss, but not certain; still worth a glance if nothing else "
        "turned up. (With fewer than two results there is no shape to measure "
        "at all, and this band applies by definition, not by that measurement.)"
    ),
    UNCALIBRATED: (
        "this ranking involved the dense retriever (dense-only or fused/hybrid) - the "
        "confident/uncertain/low_confidence gate was calibrated on raw BM25 margins only "
        "and has not been re-derived for this score shape, so no confidence label is given; "
        "read the passages yourself (see README 'Confidence')"
    ),
}


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


def _metadata_filters(
    *,
    project: str | None,
    role: str | None,
    session: str | None,
    since: str | None,
    until: str | None,
    include_sidechain: bool,
) -> tuple[list[str], list[object]]:
    """The metadata WHERE clauses/params every retriever filters candidates by.

    `project` and `session` are substring matches (a session id is a full UUID nobody types
    from memory); `role` is exact (`user` / `assistant` / `memory` - the corpus's speaker
    dimension); `since` / `until` are ISO-8601 timestamp bounds, inclusive. This is the
    mechanism that makes the tool cheaper than reading a whole conversation - it is
    load-bearing, not a convenience (§13.1 "the captain's chapter/page/paragraph"), and
    `vdbaccuracy` §3 measured it as the single biggest recall lever in this whole project.

    `BM25Retriever`, `DenseRetriever`, and `HybridRetriever` all build their WHERE clause from
    this one function - the task brief's requirement that filtering narrow both retrievers'
    candidate sets *identically* before fusion is enforced by construction, not by keeping two
    copies of this logic in sync by hand.
    """
    where: list[str] = []
    params: list[object] = []
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
    return where, params


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
        """Metadata filters (`_metadata_filters`) narrow the candidate set before BM25 ranks it."""
        where, params = _metadata_filters(
            project=project,
            role=role,
            session=session,
            since=since,
            until=until,
            include_sidechain=include_sidechain,
        )
        where = ["chunks_fts MATCH ?"] + where
        params = [match_expression(question)] + params
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
        # `nudge_active()` is the GLOBAL runtime gate (report §6.3): below
        # the corpus-wide label threshold, or with the flag off, or with no
        # recorded passing regression check, this block does not run at all
        # - the nudge is provably inert, not just defaulted off. When it does
        # run, `score_nudge()` still applies the PER-CHUNK gate itself
        # (`store.NUDGE_PER_CHUNK_MIN`), so a hit whose own chunk hasn't
        # individually earned it gets +0.0 even while `nudge_applied` is
        # True. It only ever reorders the already-fetched top-k (never pulls
        # in candidates BM25 itself did not return), keeping its influence
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


class DenseRetriever:
    """`multilingual-e5-large` cosine search over `chunk_embeddings` (`vdb/dense.py`).

    Brute-force, not ANN: `vdbscout`'s "storage/index options" finding is that a plain scan is
    comparable-or-faster than `sqlite-vec` at this corpus's scale, so there is no index
    structure here beyond the SQLite BLOB rows themselves. Vectors are stored L2-normalized
    (`dense.Embedder`), so cosine similarity is a plain dot product.

    No feedback nudge (decision A - see this module's docstring and README "Feedback"): this
    retriever never reads `chunk_feedback`. A chunk missing from `chunk_embeddings` (the dense
    index build hasn't reached it yet, or is still running - see README "Dense index") is
    simply absent from this retriever's candidates; it is not an error.
    """

    name = "dense"

    def __init__(self, conn: sqlite3.Connection, embedder=None):
        self.conn = conn
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from . import dense as dense_mod

            self._embedder = dense_mod.Embedder()
        return self._embedder

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
        """Metadata filters (`_metadata_filters`) narrow the candidate set before cosine ranks it.

        The exact same filter function `BM25Retriever.search()` uses, applied here to the
        `chunk_embeddings` join instead of `chunks_fts` - the two retrievers' candidate sets
        differ only in which chunks have been dense-indexed so far, never in which metadata
        filters were applied.
        """
        from . import dense as dense_mod

        where, params = _metadata_filters(
            project=project,
            role=role,
            session=session,
            since=since,
            until=until,
            include_sidechain=include_sidechain,
        )
        where = ["ce.model = ?"] + where
        params = [dense_mod.MODEL_NAME] + params

        sql = (
            "SELECT c.id, c.session_id, c.project, c.role, c.ts, c.part, c.n_parts,"
            "       c.is_sidechain, f.path AS source, t.text AS text, ce.vector AS vector "
            "FROM chunk_embeddings ce "
            "JOIN chunks c ON c.id = ce.chunk_id "
            "JOIN files f ON f.id = c.file_id "
            "JOIN chunks_fts t ON t.rowid = c.id "
            f"WHERE {' AND '.join(where)}"
        )
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return Result(query=question, k_requested=k, hits=[], retriever="dense")

        import numpy as np

        matrix = np.frombuffer(
            b"".join(row["vector"] for row in rows), dtype="<f4"
        ).reshape(len(rows), -1)
        query_vec = np.asarray(self.embedder.embed_query(question), dtype=np.float32)
        scores = matrix @ query_vec
        order = np.argsort(-scores)[:k]

        hits = [
            Hit(
                chunk_id=rows[i]["id"],
                rank=rank,
                score=round(float(scores[i]), 4),
                text=rows[i]["text"],
                session_id=rows[i]["session_id"],
                project=rows[i]["project"],
                role=rows[i]["role"],
                ts=rows[i]["ts"],
                part=rows[i]["part"],
                n_parts=rows[i]["n_parts"],
                source=rows[i]["source"],
                is_sidechain=bool(rows[i]["is_sidechain"]),
            )
            for rank, i in enumerate(order, start=1)
        ]
        return Result(query=question, k_requested=k, hits=hits, retriever="dense")


# vdbqual §11.4's exact measured fusion mechanism: "Hybrid = reciprocal-rank fusion (k=60) of
# the dense and BM25 rankings." Not re-derived here - the brief for this task is explicit that
# this is the mechanism to implement, not a novel scheme.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Standard reciprocal-rank fusion over N ranked lists of chunk ids.

    `score(chunk) = sum, over every ranking that contains it, of 1 / (k + rank)` (rank
    1-indexed) - `vdbqual` §11.4's exact formula, k=60. A chunk absent from one ranking simply
    contributes no term for that ranking, rather than a penalty - the report does not specify
    tie/absence handling explicitly, so this is the standard RRF convention, not a measured
    choice. Returns `(chunk_id, fused_score)` pairs sorted by score descending; ties are
    broken by first-seen order across `rankings` (Python dicts preserve insertion order, and
    `sorted` is stable), so output is deterministic for the same inputs.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# How many candidates each retriever is asked for before fusion. `vdbqual` §11.4 does not
# document a candidate-pool cap for RRF itself (only the now-rejected reranker had one,
# top-20/30) - this is an engineering choice, not a re-derived research constant: large enough
# that a requested top-k has almost always survived in both single-retriever rankings, small
# enough to keep both the FTS5 query and the cosine scan cheap. Stated as such in the PR.
FUSION_POOL_MULTIPLIER = 5
FUSION_POOL_MIN = 50


class HybridRetriever:
    """BM25 + dense (`multilingual-e5-large`), combined by reciprocal-rank fusion.

    The measured configuration (`vdbqual` §10.3/§11.4): fuse BM25's ranking with e5-large's
    dense ranking by RRF (`reciprocal_rank_fusion`, k=60); no reranker (§11.4 measured a
    cross-encoder reranker making the best configuration *worse* on every metric but one -
    rejected, not merely unimplemented). Filters are applied identically to both retrievers
    before either ranks anything (`_metadata_filters`) - `vdbaccuracy` §3 found pre-search
    filtering the single biggest recall lever in this project, so it must narrow both
    candidate sets, not just one.

    See this module's docstring for the two phase-2 decisions this class embodies: no nudge
    for the dense side (decision A), and `Result.confidence == UNCALIBRATED` rather than the
    BM25-fit bands (decision B, enforced by `Result.retriever == "hybrid"`).
    """

    name = "hybrid"

    def __init__(self, conn: sqlite3.Connection, embedder=None):
        self.conn = conn
        self.bm25 = BM25Retriever(conn)
        self.dense = DenseRetriever(conn, embedder=embedder)

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
        pool = max(k * FUSION_POOL_MULTIPLIER, FUSION_POOL_MIN)
        filters = dict(
            project=project,
            role=role,
            session=session,
            since=since,
            until=until,
            include_sidechain=include_sidechain,
        )
        bm25_result = self.bm25.search(question, k=pool, **filters)
        dense_result = self.dense.search(question, k=pool, **filters)

        by_id: dict[int, Hit] = {}
        for hit in (*bm25_result.hits, *dense_result.hits):
            by_id.setdefault(hit.chunk_id, hit)

        fused = reciprocal_rank_fusion(
            [[h.chunk_id for h in bm25_result.hits], [h.chunk_id for h in dense_result.hits]]
        )

        hits = [
            Hit(
                chunk_id=chunk_id,
                rank=rank,
                score=round(score, 6),
                text=by_id[chunk_id].text,
                session_id=by_id[chunk_id].session_id,
                project=by_id[chunk_id].project,
                role=by_id[chunk_id].role,
                ts=by_id[chunk_id].ts,
                part=by_id[chunk_id].part,
                n_parts=by_id[chunk_id].n_parts,
                source=by_id[chunk_id].source,
                is_sidechain=by_id[chunk_id].is_sidechain,
            )
            for rank, (chunk_id, score) in enumerate(fused[:k], start=1)
        ]

        return Result(
            query=question,
            k_requested=k,
            hits=hits,
            nudge_applied=bm25_result.nudge_applied,
            retriever="hybrid",
        )
