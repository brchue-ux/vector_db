"""The calibrated confidence gate (`retrieve.confidence_band`, `Result.confidence`).

Covers: the classification logic against constructed score shapes (clear
hit-shaped, clear miss-shaped, ambiguous), the band boundaries themselves, and
that this is a labelling layer only - ranking and scores are unchanged by it.
See `vdb/retrieve.py`'s module comment above `CONFIDENCE_LOW_MARGIN` for the
calibration measurement this is built on.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests import fixture as fx
from vdb import cli, retrieve, store


def _hits(scores):
    return [
        retrieve.Hit(chunk_id=i, rank=i + 1, score=s, text="x") for i, s in enumerate(scores)
    ]


class ConfidenceBandBoundaries(unittest.TestCase):
    """Unit tests against constructed margins - no index needed."""

    def test_no_margin_is_low_confidence(self):
        self.assertEqual(retrieve.confidence_band(None), retrieve.LOW_CONFIDENCE)

    def test_zero_margin_is_low_confidence(self):
        self.assertEqual(retrieve.confidence_band(0.0), retrieve.LOW_CONFIDENCE)

    def test_clearly_miss_shaped_margin_is_low_confidence(self):
        # a flat top-10, nothing stands out - the miss-shaped case the
        # calibration set's low band was measured on
        self.assertEqual(retrieve.confidence_band(10.0), retrieve.LOW_CONFIDENCE)

    def test_ambiguous_margin_is_uncertain(self):
        midpoint = (retrieve.CONFIDENCE_LOW_MARGIN + retrieve.CONFIDENCE_HIGH_MARGIN) / 2
        self.assertEqual(retrieve.confidence_band(midpoint), retrieve.UNCERTAIN)

    def test_clearly_hit_shaped_margin_is_confident(self):
        # a sharply peaked top-10 - the hit-shaped case the calibration set's
        # high band was measured on
        self.assertEqual(retrieve.confidence_band(500.0), retrieve.CONFIDENT)

    def test_low_boundary_is_inclusive_on_the_uncertain_side(self):
        self.assertEqual(
            retrieve.confidence_band(retrieve.CONFIDENCE_LOW_MARGIN), retrieve.UNCERTAIN
        )
        self.assertEqual(
            retrieve.confidence_band(retrieve.CONFIDENCE_LOW_MARGIN - 0.001),
            retrieve.LOW_CONFIDENCE,
        )

    def test_high_boundary_is_inclusive_on_the_confident_side(self):
        self.assertEqual(
            retrieve.confidence_band(retrieve.CONFIDENCE_HIGH_MARGIN), retrieve.CONFIDENT
        )
        self.assertEqual(
            retrieve.confidence_band(retrieve.CONFIDENCE_HIGH_MARGIN - 0.001), retrieve.UNCERTAIN
        )

    def test_every_band_has_an_explanation(self):
        for band in (retrieve.CONFIDENT, retrieve.UNCERTAIN, retrieve.LOW_CONFIDENCE):
            self.assertIn(band, retrieve.CONFIDENCE_EXPLANATION)
            self.assertTrue(retrieve.CONFIDENCE_EXPLANATION[band])


class ResultConfidence(unittest.TestCase):
    """`Result.confidence` on constructed Hit lists - no index needed."""

    def test_no_hits_is_low_confidence(self):
        res = retrieve.Result(query="q", k_requested=10, hits=[])
        self.assertEqual(res.confidence, retrieve.LOW_CONFIDENCE)

    def test_single_hit_is_low_confidence_margin_undefined(self):
        res = retrieve.Result(query="q", k_requested=10, hits=_hits([10.0]))
        self.assertEqual(res.confidence, retrieve.LOW_CONFIDENCE)

    def test_sharply_peaked_top10_is_confident(self):
        # rank 1 stands far apart from a flat, low tail
        scores = [500.0] + [5.0] * 9
        res = retrieve.Result(query="q", k_requested=10, hits=_hits(scores))
        self.assertEqual(res.confidence, retrieve.CONFIDENT)

    def test_flat_top10_is_low_confidence(self):
        # every score close together - nothing stands out
        scores = [10.0, 9.5, 9.2, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0, 7.8]
        res = retrieve.Result(query="q", k_requested=10, hits=_hits(scores))
        self.assertEqual(res.confidence, retrieve.LOW_CONFIDENCE)

    def test_moderate_gap_is_uncertain(self):
        scores = [110.0] + [10.0] * 9  # margin=100, between the two cut points
        res = retrieve.Result(query="q", k_requested=10, hits=_hits(scores))
        self.assertEqual(res.confidence, retrieve.UNCERTAIN)

    def test_confidence_is_independent_of_weak_signal(self):
        # thin coverage (fewer hits than requested) but the two available
        # hits are still sharply separated - weak_signal and confidence are
        # different axes, not aliases of each other
        res = retrieve.Result(query="q", k_requested=10, hits=_hits([500.0, 5.0]))
        self.assertTrue(res.weak_signal)
        self.assertEqual(res.confidence, retrieve.CONFIDENT)


class ConfidenceDoesNotChangeRanking(unittest.TestCase):
    """A labelling layer, not a ranking change (task brief's explicit constraint)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "projects"
        self.root.mkdir()
        self.db = base / "index.sqlite3"
        fx.write_transcript(
            self.root,
            "p",
            "s",
            [
                fx.user_entry(
                    "the deploy pipeline keeps timing out on the docker build step", uuid="u1"
                ),
                fx.user_entry("what should we have for lunch", uuid="u2"),
            ],
        )
        self.conn = store.connect(self.db)
        self.addCleanup(self.conn.close)
        store.index(self.conn, root=self.root)

    def test_hits_and_scores_identical_whether_or_not_confidence_is_read(self):
        res = retrieve.BM25Retriever(self.conn).search("docker build timing out")
        before = [(h.chunk_id, h.rank, h.score) for h in res.hits]
        _ = res.confidence  # touching the property must not mutate anything
        after = [(h.chunk_id, h.rank, h.score) for h in res.hits]
        self.assertEqual(before, after)

    def test_two_identical_searches_get_the_same_confidence(self):
        r1 = retrieve.BM25Retriever(self.conn).search("docker build timing out")
        r2 = retrieve.BM25Retriever(self.conn).search("docker build timing out")
        self.assertEqual(r1.confidence, r2.confidence)
        self.assertEqual([h.chunk_id for h in r1.hits], [h.chunk_id for h in r2.hits])


class CliConfidenceOutput(unittest.TestCase):
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
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--db", str(self.db)] + argv)
        return code, out.getvalue()

    def test_human_output_carries_a_confidence_line(self):
        code, out = self.run_cli(["query", "distinctive query term"])
        self.assertEqual(code, 0)
        self.assertIn("confidence:", out)

    def test_json_output_carries_confidence_and_a_note(self):
        code, out = self.run_cli(["query", "distinctive query term", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn(payload["confidence"], (retrieve.CONFIDENT, retrieve.UNCERTAIN, retrieve.LOW_CONFIDENCE))
        self.assertEqual(payload["confidence_note"], retrieve.CONFIDENCE_EXPLANATION[payload["confidence"]])

    def test_no_hits_still_reports_a_confidence(self):
        code, out = self.run_cli(["query", "zqxvbnkjplomfghqz"])
        self.assertEqual(code, 1)
        self.assertIn("confidence:", out)

    def test_no_hits_json_still_reports_a_confidence(self):
        code, out = self.run_cli(["query", "zqxvbnkjplomfghqz", "--json"])
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertEqual(payload["confidence"], retrieve.LOW_CONFIDENCE)


if __name__ == "__main__":
    unittest.main()
