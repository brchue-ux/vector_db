"""Phase 2: dense retrieval (`vdb/dense.py`) and reciprocal-rank fusion (`vdb/retrieve.py`).

All fixtures are constructed, same discipline as tests/fixture.py - no real embedding model is
loaded here. `FakeEmbedder` is a deterministic stand-in with the same `embed_passages`/
`embed_query` shape `dense.Embedder` has, so `DenseRetriever`/`dense.build_index` can be
exercised without the real ~2.24GB `multilingual-e5-large` model or onnxruntime installed.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from tests import fixture as fx
from vdb import dense, retrieve, store

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

_needs_numpy = unittest.skipUnless(
    HAS_NUMPY, "DenseRetriever/HybridRetriever.search() need numpy - install the 'dense' extra"
)


class FakeEmbedder:
    """Deterministic bag-of-words embeddings - no fastembed/onnxruntime needed.

    Hashes each lowercase word into one of `dim` buckets (stable hashlib digest, not the
    per-process-randomized builtin `hash()`) and L2-normalizes, mirroring what `dense.Embedder`
    does to real model output. Two texts that share more words score a higher cosine similarity
    - close enough to a real embedding's behaviour to make ranking assertions meaningful,
    without downloading or running any model.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            bucket = int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class DenseBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "projects"
        self.root.mkdir()
        self.db = base / "index.sqlite3"
        self.conn = store.connect(self.db)
        self.addCleanup(self.conn.close)
        self.embedder = FakeEmbedder()

    def build_dense(self, **kw):
        return dense.build_index(self.conn, embedder=self.embedder, **kw)


class RRFFusion(unittest.TestCase):
    """`reciprocal_rank_fusion` against constructed rank lists - no index needed."""

    def test_matches_the_measured_formula_exactly(self):
        # vdbqual §11.4: score(chunk) = sum of 1/(k+rank) across every ranking it appears in.
        fused = retrieve.reciprocal_rank_fusion([[1, 2, 3]], k=60)
        self.assertEqual(
            fused,
            [(1, 1 / 61), (2, 1 / 62), (3, 1 / 63)],
        )

    def test_default_k_is_60(self):
        with_default = retrieve.reciprocal_rank_fusion([[1]])
        with_explicit = retrieve.reciprocal_rank_fusion([[1]], k=60)
        self.assertEqual(with_default, with_explicit)
        self.assertEqual(retrieve.RRF_K, 60)

    def test_agreement_across_both_rankings_outranks_a_single_top_hit(self):
        # chunk 9 is rank 1 in one list only; chunk 1 is rank 2 in both - RRF's whole point is
        # that consistent moderate agreement can beat a single strong disagreement.
        bm25 = [9, 1, 2]
        dense_ranks = [1, 3, 4]
        fused = dict(retrieve.reciprocal_rank_fusion([bm25, dense_ranks]))
        self.assertGreater(fused[1], fused[9])

    def test_chunk_absent_from_one_ranking_gets_no_penalty_term(self):
        # present in both at rank 1 vs present in only one at rank 1: the latter is just the
        # single term, not a term plus some penalty for the missing ranking.
        fused = dict(retrieve.reciprocal_rank_fusion([[1], [2]]))
        self.assertAlmostEqual(fused[1], 1 / 61)
        self.assertAlmostEqual(fused[2], 1 / 61)

    def test_empty_rankings_produce_no_fused_results(self):
        self.assertEqual(retrieve.reciprocal_rank_fusion([[], []]), [])

    def test_output_is_sorted_descending_by_score(self):
        fused = retrieve.reciprocal_rank_fusion([[5, 1, 9], [9, 5, 1]])
        scores = [s for _, s in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_deterministic_for_identical_inputs(self):
        rankings = [[3, 1, 4, 1, 5], [9, 2, 6, 5, 3]]
        self.assertEqual(
            retrieve.reciprocal_rank_fusion(rankings), retrieve.reciprocal_rank_fusion(rankings)
        )


class VectorPacking(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        vector = [0.1, -0.2, 0.375, 1.0, -1.0]
        blob = dense.pack_vector(vector)
        restored = dense.unpack_vector(blob)
        for a, b in zip(vector, restored):
            self.assertAlmostEqual(a, b, places=6)

    def test_blob_length_matches_dimension(self):
        blob = dense.pack_vector([0.0] * dense.EMBED_DIM)
        self.assertEqual(len(blob), dense.EMBED_DIM * 4)


class BuildIndex(DenseBase):
    def _add_chunk(self, text="a distinctive message about deploy pipelines"):
        fx.write_transcript(self.root, "p", "s", [fx.user_entry(text)])
        store.index(self.conn, root=self.root)

    def test_embeds_every_chunk_once(self):
        self._add_chunk()
        st = self.build_dense()
        self.assertEqual(st.candidates, 1)
        self.assertEqual(st.embedded, 1)
        cov = dense.coverage(self.conn)
        self.assertEqual(cov, {"model": dense.MODEL_NAME, "chunks": 1, "embedded": 1,
                                "remaining": 0, "complete": True})

    def test_incremental_skips_already_embedded_chunks(self):
        self._add_chunk()
        self.build_dense()
        second = self.build_dense()
        self.assertEqual(second.candidates, 0)
        self.assertEqual(second.embedded, 0)

    def test_new_chunk_after_a_build_is_picked_up_incrementally(self):
        self._add_chunk("first message here")
        self.build_dense()
        fx.write_transcript(self.root, "p", "s2", [fx.user_entry("second message here")])
        store.index(self.conn, root=self.root)
        second = self.build_dense()
        self.assertEqual(second.candidates, 1)
        self.assertEqual(dense.coverage(self.conn)["embedded"], 2)

    def test_rebuild_clears_and_reembeds_everything(self):
        self._add_chunk()
        self.build_dense()
        st = self.build_dense(rebuild=True)
        self.assertEqual(st.candidates, 1)
        self.assertEqual(st.embedded, 1)

    def test_stored_vectors_are_l2_normalized(self):
        self._add_chunk()
        self.build_dense()
        row = self.conn.execute("SELECT vector FROM chunk_embeddings").fetchone()
        vec = dense.unpack_vector(row["vector"])
        norm = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_index_embeds_nothing(self):
        st = self.build_dense()
        self.assertEqual(st.candidates, 0)
        self.assertEqual(st.embedded, 0)


class ForgetCascadesToEmbeddings(DenseBase):
    """Deleting/rebuilding chunks must not leave orphaned or mismatched vectors."""

    def test_removed_file_drops_its_chunk_embeddings(self):
        fx.write_transcript(self.root, "p", "gone", [fx.user_entry("will be removed")])
        store.index(self.conn, root=self.root)
        self.build_dense()
        self.assertEqual(dense.coverage(self.conn)["embedded"], 1)

        import shutil

        shutil.rmtree(self.root / "p")
        store.index(self.conn, root=self.root)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0], 0)

    def test_rebuild_clears_stale_embeddings_too(self):
        fx.write_transcript(self.root, "p", "s", [fx.user_entry("some content")])
        store.index(self.conn, root=self.root)
        self.build_dense()
        store.index(self.conn, root=self.root, rebuild=True)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0], 0)


@_needs_numpy
class DenseRetrieverRanking(DenseBase):
    def test_ranks_the_more_similar_passage_first(self):
        fx.write_transcript(
            self.root,
            "p",
            "s",
            [
                fx.user_entry("deploy pipeline docker build timeout", uuid="u1"),
                fx.user_entry("what should we have for lunch today", uuid="u2"),
            ],
        )
        store.index(self.conn, root=self.root)
        self.build_dense()

        result = retrieve.DenseRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline"
        )
        self.assertEqual(result.retriever, "dense")
        self.assertGreater(len(result.hits), 0)
        top_text = result.hits[0].text
        self.assertIn("docker", top_text)

    def test_no_embeddings_yet_returns_no_hits_not_an_error(self):
        fx.write_transcript(self.root, "p", "s", [fx.user_entry("nothing embedded yet")])
        store.index(self.conn, root=self.root)
        result = retrieve.DenseRetriever(self.conn, embedder=self.embedder).search("anything")
        self.assertEqual(result.hits, [])
        self.assertEqual(result.retriever, "dense")


@_needs_numpy
class FiltersNarrowBothRetrieversIdentically(DenseBase):
    """`vdbaccuracy` §3: pre-search filtering must narrow every retriever's candidates alike."""

    def setUp(self):
        super().setUp()
        fx.write_transcript(
            self.root,
            "alpha",
            "s1",
            [
                fx.user_entry("sharedterm alpha message one", uuid="a1"),
                fx.user_entry("sharedterm alpha message two", uuid="a2"),
            ],
        )
        fx.write_transcript(
            self.root,
            "beta",
            "s2",
            [
                fx.user_entry("sharedterm beta message one", uuid="b1"),
                fx.user_entry("sharedterm beta message two", uuid="b2"),
            ],
        )
        store.index(self.conn, root=self.root)
        self.build_dense()
        self.alpha_ids = {
            r["id"] for r in self.conn.execute("SELECT id FROM chunks WHERE project='alpha'")
        }
        self.beta_ids = {
            r["id"] for r in self.conn.execute("SELECT id FROM chunks WHERE project='beta'")
        }

    def test_project_filter_narrows_bm25_candidates(self):
        result = retrieve.BM25Retriever(self.conn).search("sharedterm", k=10, project="alpha")
        got = {h.chunk_id for h in result.hits}
        self.assertEqual(got, self.alpha_ids)

    def test_project_filter_narrows_dense_candidates_identically(self):
        result = retrieve.DenseRetriever(self.conn, embedder=self.embedder).search(
            "sharedterm", k=10, project="alpha"
        )
        got = {h.chunk_id for h in result.hits}
        self.assertEqual(got, self.alpha_ids)

    def test_bm25_and_dense_candidate_sets_match_for_the_same_filter(self):
        bm25 = retrieve.BM25Retriever(self.conn).search("sharedterm", k=10, project="beta")
        dense_r = retrieve.DenseRetriever(self.conn, embedder=self.embedder).search(
            "sharedterm", k=10, project="beta"
        )
        self.assertEqual({h.chunk_id for h in bm25.hits}, {h.chunk_id for h in dense_r.hits})
        self.assertEqual({h.chunk_id for h in bm25.hits}, self.beta_ids)

    def test_hybrid_result_never_includes_a_filtered_out_chunk(self):
        result = retrieve.HybridRetriever(self.conn, embedder=self.embedder).search(
            "sharedterm", k=10, project="alpha"
        )
        got = {h.chunk_id for h in result.hits}
        self.assertTrue(got.issubset(self.alpha_ids))
        self.assertFalse(got & self.beta_ids)


@_needs_numpy
class HybridFusionCorrectness(DenseBase):
    def setUp(self):
        super().setUp()
        # Three chunks; word choice deliberately makes BM25 and the fake dense embedder
        # disagree about ranking, so fusion is doing real work, not just echoing one side.
        fx.write_transcript(
            self.root,
            "p",
            "s",
            [
                fx.user_entry("docker docker docker build pipeline timeout", uuid="u1"),
                fx.user_entry("docker build pipeline succeeded quickly", uuid="u2"),
                fx.user_entry("completely unrelated lunch conversation topic", uuid="u3"),
            ],
        )
        store.index(self.conn, root=self.root)
        self.build_dense()

    def test_fused_score_matches_rrf_of_the_two_component_rankings(self):
        bm25_result = retrieve.BM25Retriever(self.conn).search("docker build pipeline", k=10)
        dense_result = retrieve.DenseRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline", k=10
        )
        expected = dict(
            retrieve.reciprocal_rank_fusion(
                [[h.chunk_id for h in bm25_result.hits], [h.chunk_id for h in dense_result.hits]]
            )
        )

        hybrid = retrieve.HybridRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline", k=10
        )
        self.assertEqual(hybrid.retriever, "hybrid")
        for hit in hybrid.hits:
            self.assertAlmostEqual(hit.score, round(expected[hit.chunk_id], 6))

    def test_hybrid_hits_are_ranked_by_descending_fused_score(self):
        hybrid = retrieve.HybridRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline", k=10
        )
        scores = [h.score for h in hybrid.hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([h.rank for h in hybrid.hits], list(range(1, len(hybrid.hits) + 1)))

    def test_truncates_to_requested_k(self):
        hybrid = retrieve.HybridRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline", k=1
        )
        self.assertEqual(len(hybrid.hits), 1)


@_needs_numpy
class DecisionA_NudgeStaysBM25Only(DenseBase):
    """Task brief decision A: the feedback nudge must never affect the dense retriever."""

    def _activate_nudge_and_boost_chunk(self, chunk_id):
        store.set_nudge_flag(self.conn, True)
        for i in range(store.NUDGE_LABEL_THRESHOLD):
            qid = store.log_query(
                self.conn, query_text="q", filters={}, k_requested=10,
                hit_chunk_ids=[1000 + i], hit_scores=[1.0], margin=None, weak_signal=True,
            )
            self.conn.execute("UPDATE query_log SET held_out=0 WHERE id=?", (qid,))
            store.record_citation(self.conn, qid, [1000 + i])
            store.write_label(self.conn, qid, "confirm", "heuristic_casual_v1", 1)
            store.apply_feedback_to_chunks(self.conn, [1000 + i], "confirm")
        store.record_regression_check(self.conn, passed=True, notes="ok")
        self.conn.commit()
        self.assertTrue(store.nudge_active(self.conn))
        # A large, unmistakable per-chunk boost if it ever leaked into the dense score.
        for _ in range(store.NUDGE_PER_CHUNK_MIN + 10):
            store.apply_feedback_to_chunks(self.conn, [chunk_id], "confirm")
        self.conn.commit()

    def test_dense_score_is_unaffected_by_an_active_and_boosted_nudge(self):
        fx.write_transcript(
            self.root, "p", "s", [fx.user_entry("docker build pipeline timeout error")]
        )
        store.index(self.conn, root=self.root)
        self.build_dense()
        chunk_id = self.conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]

        before = retrieve.DenseRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline"
        ).hits[0].score

        self._activate_nudge_and_boost_chunk(chunk_id)

        after_result = retrieve.DenseRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline"
        )
        self.assertAlmostEqual(after_result.hits[0].score, before)

    def test_hybrid_still_carries_bm25s_own_nudge_but_dense_side_is_untouched(self):
        fx.write_transcript(
            self.root, "p", "s", [fx.user_entry("docker build pipeline timeout error")]
        )
        store.index(self.conn, root=self.root)
        self.build_dense()
        chunk_id = self.conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]

        self._activate_nudge_and_boost_chunk(chunk_id)

        bm25_alone = retrieve.BM25Retriever(self.conn).search("docker build pipeline")
        self.assertTrue(bm25_alone.nudge_applied)

        hybrid = retrieve.HybridRetriever(self.conn, embedder=self.embedder).search(
            "docker build pipeline"
        )
        # HybridRetriever reports whatever its internal BM25 sub-search reported - the nudge
        # is real for the BM25 half, it just never touches the dense half (previous test).
        self.assertTrue(hybrid.nudge_applied)


class DecisionB_ConfidenceUncalibratedForNonBM25(unittest.TestCase):
    """Task brief decision B: never reuse the BM25-fit bands on a dense/hybrid margin."""

    def _hits(self, scores, retriever):
        hits = [
            retrieve.Hit(chunk_id=i, rank=i + 1, score=s, text="x") for i, s in enumerate(scores)
        ]
        return retrieve.Result(query="q", k_requested=10, hits=hits, retriever=retriever)

    def test_dense_result_is_uncalibrated_even_with_an_extreme_margin(self):
        # This exact score shape would be `confident` under the BM25-fit bands - proving the
        # dense/hybrid path does not fall through to them.
        res = self._hits([500.0] + [5.0] * 9, retriever="dense")
        self.assertEqual(res.confidence, retrieve.UNCALIBRATED)

    def test_hybrid_result_is_uncalibrated_even_with_an_extreme_margin(self):
        res = self._hits([500.0] + [5.0] * 9, retriever="hybrid")
        self.assertEqual(res.confidence, retrieve.UNCALIBRATED)

    def test_bm25_result_still_uses_the_calibrated_bands(self):
        res = self._hits([500.0] + [5.0] * 9, retriever="bm25")
        self.assertEqual(res.confidence, retrieve.CONFIDENT)

    def test_default_retriever_is_bm25(self):
        res = retrieve.Result(query="q", hits=[])
        self.assertEqual(res.retriever, "bm25")

    def test_uncalibrated_has_an_explanation(self):
        self.assertIn(retrieve.UNCALIBRATED, retrieve.CONFIDENCE_EXPLANATION)
        self.assertTrue(retrieve.CONFIDENCE_EXPLANATION[retrieve.UNCALIBRATED])

    def test_uncalibrated_is_not_one_of_the_three_calibrated_bands(self):
        self.assertNotIn(
            retrieve.UNCALIBRATED, (retrieve.CONFIDENT, retrieve.UNCERTAIN, retrieve.LOW_CONFIDENCE)
        )

    def test_margin_and_weak_signal_are_unaffected_by_retriever(self):
        # Only `confidence` special-cases retriever - the underlying diagnostics stay generic.
        hits = [retrieve.Hit(chunk_id=i, rank=i + 1, score=s, text="x") for i, s in enumerate([10.0, 9.0])]
        res = retrieve.Result(query="q", k_requested=2, hits=hits, retriever="hybrid")
        self.assertIsNotNone(res.margin)
        self.assertFalse(res.weak_signal)


class CliHybridFlag(unittest.TestCase):
    """CLI wiring for `vdb query --hybrid`, with an empty dense index (no model needed)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "projects"
        self.root.mkdir()
        self.db = base / "index.sqlite3"
        fx.write_transcript(
            self.root, "p", "s", [fx.user_entry("a distinctive query term appears here")]
        )
        conn = store.connect(self.db)
        store.index(conn, root=self.root)
        conn.close()

    def run_cli(self, argv):
        import contextlib
        import io

        from vdb import cli

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--db", str(self.db)] + argv)
        return code, out.getvalue(), err.getvalue()

    def test_hybrid_with_no_dense_index_warns_but_still_returns_bm25_candidates(self):
        code, out, err = self.run_cli(["query", "distinctive query term", "--hybrid", "--json"])
        self.assertEqual(code, 0)
        self.assertIn("dense index is empty", err)
        import json

        payload = json.loads(out)
        self.assertEqual(payload["retriever"], "hybrid")
        self.assertEqual(payload["confidence"], retrieve.UNCALIBRATED)
        self.assertGreater(payload["n_hits"], 0)

    def test_plain_query_is_still_bm25_and_calibrated(self):
        code, out, err = self.run_cli(["query", "distinctive query term", "--json"])
        self.assertEqual(code, 0)
        import json

        payload = json.loads(out)
        self.assertEqual(payload["retriever"], "bm25")
        self.assertIn(
            payload["confidence"],
            (retrieve.CONFIDENT, retrieve.UNCERTAIN, retrieve.LOW_CONFIDENCE),
        )

    def test_dense_index_status_reports_zero_coverage(self):
        code, out, _ = self.run_cli(["dense-index", "--status", "--json"])
        self.assertEqual(code, 0)
        import json

        payload = json.loads(out)
        self.assertEqual(payload["embedded"], 0)
        self.assertFalse(payload["complete"])


if __name__ == "__main__":
    unittest.main()
