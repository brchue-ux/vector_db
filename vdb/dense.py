"""Dense retrieval: `multilingual-e5-large` embeddings, brute-force cosine search.

Phase 2 (`data/decisions/vdbqual-decision-embedding-model-cost.md`, outside this repo):
`multilingual-e5-large` is the only model in the whole research line that beats BM25 alone
on any metric once fused (`vdbqual` §10.3: hybrid(e5-large) +0.033 MRR over BM25 alone, CI
[+0.004,+0.063]) and its ranking advantage becomes a real recall gain once results are cut
to top 3-5 (`vdbaccuracy` §4: MACRO recall@3 +0.062, CI [+0.006,+0.119], concentrated in
family Q). See `vdb/retrieve.py` for the reciprocal-rank fusion that combines this
retriever's ranking with BM25's.

**Local, dependency-lazy, on purpose.** This repo has been stdlib-only through phase 1
(`AGENTS.md` "Sharp edges"); dense retrieval is the first module that needs anything else.
`fastembed` (ONNX Runtime under the hood, no PyTorch, no network calls at query time beyond
the one-time model download) is imported lazily, inside `Embedder._get()` - importing this
module, or running `vdb query` without `--hybrid`, never requires fastembed/onnxruntime to
be installed. Install with `pip install -e .[dense]`.

**e5's prefix convention is not optional.** Unlike most fastembed models, e5-family models
were trained with `"query: "` / `"passage: "` instruction prefixes distinguishing the two
encoding roles (fastembed's own model registry: "Prefixes for queries/documents:
necessary") - and fastembed does not add these automatically (checked against the installed
package: `PooledEmbedding`, the class backing this model, does no prefixing of its own).
Omitting them measurably degrades e5 retrieval quality in the wider literature this domain's
research draws on; this module always adds them.

**Vectors are stored L2-normalized** so cosine similarity reduces to a plain dot product at
search time (`fastembed`'s `PooledEmbedding` - unlike `PooledNormalizedEmbedding` - does not
normalize its output; this module does it once, at index time, rather than on every query).

**Storage: SQLite BLOB + brute-force numpy scan, not `sqlite-vec`.** `vdbscout`'s "storage/
index options" finding is that both are fast and cheap at this corpus's scale, with
brute-force comparable-or-faster and no extra compiled-extension dependency; `sqlite-vec`
buys ANN semantics this corpus is too small to need. Vectors live in `chunk_embeddings`
(`vdb/store.py`), one row per chunk, addressed by the same stable `chunks.id` the BM25 side
already uses - no separate vector file, one artefact to protect either way.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from typing import Callable, Sequence

MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_DIM = 1024

# e5's own training convention (see module docstring) - required, not cosmetic.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# fastembed's own default batch size for `.embed()`; chosen here mainly so
# `build_index`'s progress callback fires at a human-legible cadence, not for
# any measured throughput reason.
DEFAULT_BATCH_SIZE = 32


def _load_model():
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise RuntimeError(
            "dense retrieval needs the optional 'fastembed' dependency - install with "
            "`pip install -e .[dense]` (pulls in onnxruntime; no PyTorch, no network calls "
            "at query time beyond the one-time model download)"
        ) from exc
    return TextEmbedding(model_name=MODEL_NAME)


def _normalize(vector) -> list[float]:
    import numpy as np

    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


class Embedder:
    """Lazy wrapper around fastembed's ONNX `multilingual-e5-large` model.

    Takes an already-constructed model (or a plain callable) for tests, so unit tests never
    need the real ~2.24GB model or onnxruntime installed - see `tests/test_dense.py`'s
    `FakeEmbedder`. Production code (`build_index`, `DenseRetriever`) gets a real one lazily
    via `Embedder()` with no arguments.
    """

    def __init__(self, model=None):
        self._model = model

    def _get(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get().embed([PASSAGE_PREFIX + t for t in texts])
        return [_normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = list(self._get().query_embed(QUERY_PREFIX + text))
        return _normalize(vectors[0])


def pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


@dataclass
class DenseIndexStats:
    candidates: int = 0
    embedded: int = 0
    skipped: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def build_index(
    conn: sqlite3.Connection,
    embedder: Embedder | None = None,
    rebuild: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> DenseIndexStats:
    """Embed every chunk that doesn't already have a `chunk_embeddings` row.

    Incremental by default, same shape as `store.index()`: a chunk already embedded under
    `MODEL_NAME` is left alone. `rebuild=True` clears `chunk_embeddings` first (e.g. after a
    model change). `progress(embedded_so_far, candidates_total)` fires after every batch, so a
    long-running real-corpus build (see README "Dense index") can report cadence without this
    function knowing anything about logging.
    """
    embedder = embedder or Embedder()
    stats = DenseIndexStats()
    if rebuild:
        conn.execute("DELETE FROM chunk_embeddings")
        conn.commit()

    rows = conn.execute(
        "SELECT c.id AS id, f.text AS text FROM chunks c "
        "JOIN chunks_fts f ON f.rowid = c.id "
        "WHERE c.id NOT IN (SELECT chunk_id FROM chunk_embeddings)"
        " ORDER BY c.id"
    ).fetchall()
    stats.candidates = len(rows)

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embedder.embed_passages([r["text"] for r in batch])
        ts = _now()
        for row, vector in zip(batch, vectors):
            conn.execute(
                "INSERT INTO chunk_embeddings(chunk_id, model, dim, vector, indexed_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT(chunk_id) DO UPDATE SET "
                "model=excluded.model, dim=excluded.dim, vector=excluded.vector, "
                "indexed_at=excluded.indexed_at",
                (row["id"], MODEL_NAME, len(vector), pack_vector(vector), ts),
            )
            stats.embedded += 1
        conn.commit()
        if progress:
            progress(stats.embedded, stats.candidates)

    return stats


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def coverage(conn: sqlite3.Connection) -> dict:
    """How much of the current chunk store has a dense embedding, for `vdb dense-index --status`."""
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    return {
        "model": MODEL_NAME,
        "chunks": total,
        "embedded": embedded,
        "remaining": max(total - embedded, 0),
        "complete": embedded >= total,
    }
