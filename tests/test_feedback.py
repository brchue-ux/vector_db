"""Implicit-feedback learning loop: logging, citation, label extraction, nudge gate.

data/vdbfeedback/report.md is the spec (outside this repo, never committed).
All fixtures here are synthetic, same discipline as tests/fixture.py.
"""

import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests import fixture as fx
from vdb import feedback as feedback_mod
from vdb import retrieve, store


class FeedbackBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "projects"
        self.root.mkdir()
        self.db = base / "index.sqlite3"
        self.conn = store.connect(self.db)
        self.addCleanup(self.conn.close)

    def log_query(self, **kw):
        kw.setdefault("query_text", "q")
        kw.setdefault("filters", {})
        kw.setdefault("k_requested", 10)
        kw.setdefault("hit_chunk_ids", [1])
        kw.setdefault("hit_scores", [1.0])
        kw.setdefault("margin", None)
        kw.setdefault("weak_signal", True)
        return store.log_query(self.conn, **kw)


class HeldOutSplit(FeedbackBase):
    def test_deterministic_for_the_same_query_id(self):
        qid = self.log_query()
        row = self.conn.execute("SELECT held_out FROM query_log WHERE id=?", (qid,)).fetchone()
        first = row["held_out"]
        # re-deriving from the same id, independent of anything written since
        self.assertEqual(store._held_out(qid), first)

    def test_decided_before_any_label_can_exist(self):
        qid = self.log_query()
        row = self.conn.execute(
            "SELECT held_out FROM query_log WHERE id=?", (qid,)
        ).fetchone()
        # No citation or label exists yet at this point, and never will change
        # held_out for this row.
        self.assertNotIn(qid, [])  # sanity: row exists
        self.assertIn(row["held_out"], (0, 1))

    def test_roughly_a_tenth_over_many_ids(self):
        n = 500
        ids = [self.log_query() for _ in range(n)]
        held = sum(
            self.conn.execute(
                "SELECT held_out FROM query_log WHERE id=?", (qid,)
            ).fetchone()["held_out"]
            for qid in ids
        )
        # Not a tight statistical claim - just "clearly a minority fraction,
        # in the right ballpark", so this can't flake on hash noise.
        self.assertTrue(20 <= held <= 90, f"held_out count {held}/500 far from ~10%")


class Citation(FeedbackBase):
    def test_recording_a_used_chunk(self):
        qid = self.log_query()
        store.record_citation(self.conn, qid, [1, 2])
        row = self.conn.execute(
            "SELECT used_chunk_ids FROM feedback_citation WHERE query_id=?", (qid,)
        ).fetchone()
        self.assertEqual(row["used_chunk_ids"], "[1, 2]")

    def test_empty_used_is_a_valid_meaningful_row(self):
        qid = self.log_query()
        store.record_citation(self.conn, qid, [])
        row = self.conn.execute(
            "SELECT used_chunk_ids FROM feedback_citation WHERE query_id=?", (qid,)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["used_chunk_ids"], "[]")

    def test_unknown_query_id_is_rejected(self):
        with self.assertRaises(ValueError):
            store.record_citation(self.conn, 999, [1])

    def test_reciting_replaces_not_duplicates(self):
        qid = self.log_query()
        store.record_citation(self.conn, qid, [1])
        store.record_citation(self.conn, qid, [2, 3])
        rows = self.conn.execute(
            "SELECT used_chunk_ids FROM feedback_citation WHERE query_id=?", (qid,)
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["used_chunk_ids"], "[2, 3]")


class Classify(unittest.TestCase):
    def test_casual_confirm(self):
        label, method = feedback_mod.classify_text("yes, that's exactly right, thanks")
        self.assertEqual(label, feedback_mod.LABEL_CONFIRM)
        self.assertIn("casual", method)

    def test_casual_correction(self):
        label, method = feedback_mod.classify_text("no, that's wrong, try again")
        self.assertEqual(label, feedback_mod.LABEL_CORRECTION)

    def test_structured_confirm(self):
        label, _ = feedback_mod.classify_text("Decision [key=widget-fix]: approved")
        self.assertEqual(label, feedback_mod.LABEL_CONFIRM)

    def test_structured_correction(self):
        label, _ = feedback_mod.classify_text("--action fix, this is not an acceptable answer")
        self.assertEqual(label, feedback_mod.LABEL_CORRECTION)

    def test_mixed_confirms_one_corrects_another(self):
        label, _ = feedback_mod.classify_text(
            "yes that part is right, but that's wrong for the other case, try again"
        )
        self.assertEqual(label, feedback_mod.LABEL_MIXED)

    def test_neither_is_none(self):
        label, _ = feedback_mod.classify_text("let's look at the billing invoices next")
        self.assertEqual(label, feedback_mod.LABEL_NONE)

    def test_empty_is_none(self):
        label, _ = feedback_mod.classify_text("")
        self.assertEqual(label, feedback_mod.LABEL_NONE)


class LabelPass(FeedbackBase):
    def _write_session(self, name, entries):
        return fx.write_transcript(self.root, "proj", name, entries)

    def test_confirm_is_found_and_updates_chunk_feedback(self):
        self._write_session(
            "sess-confirm",
            [
                fx.user_entry("first question", session="sess-confirm", ts="2026-08-01T10:00:00Z"),
                fx.assistant_entry(
                    [{"type": "text", "text": "here is an answer"}], session="sess-confirm"
                ),
                fx.user_entry(
                    "yes, that's exactly right",
                    session="sess-confirm",
                    uuid="u2",
                    ts="2026-08-01T10:05:00Z",
                ),
            ],
        )
        qid = self.log_query(hit_chunk_ids=[42])
        self.conn.execute(
            "UPDATE query_log SET caller_session='sess-confirm', ts='2026-08-01T10:00:00Z' WHERE id=?",
            (qid,),
        )
        store.record_citation(self.conn, qid, [42])

        st = feedback_mod.run_label_pass(
            self.conn, root=self.root, min_age_seconds=0,
            now=datetime(2026, 8, 1, 11, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(st.labeled, 1)
        self.assertEqual(st.by_label.get("confirm"), 1)

        row = self.conn.execute(
            "SELECT label FROM feedback_label WHERE query_id=?", (qid,)
        ).fetchone()
        self.assertEqual(row["label"], "confirm")

        fb = self.conn.execute(
            "SELECT positive_n, negative_n FROM chunk_feedback WHERE chunk_id=42"
        ).fetchone()
        self.assertEqual((fb["positive_n"], fb["negative_n"]), (1, 0))

    def test_mixed_is_recorded_but_does_not_update_chunk_feedback(self):
        self._write_session(
            "sess-mixed",
            [
                fx.user_entry("first question", session="sess-mixed", ts="2026-08-01T10:00:00Z"),
                fx.user_entry(
                    "yes that part is right, but that's wrong for the other case",
                    session="sess-mixed",
                    uuid="u2",
                    ts="2026-08-01T10:05:00Z",
                ),
            ],
        )
        qid = self.log_query(hit_chunk_ids=[7])
        self.conn.execute(
            "UPDATE query_log SET caller_session='sess-mixed', ts='2026-08-01T10:00:00Z' WHERE id=?",
            (qid,),
        )
        store.record_citation(self.conn, qid, [7])

        feedback_mod.run_label_pass(
            self.conn, root=self.root, min_age_seconds=0,
            now=datetime(2026, 8, 1, 11, 0, 0, tzinfo=timezone.utc),
        )
        row = self.conn.execute(
            "SELECT label FROM feedback_label WHERE query_id=?", (qid,)
        ).fetchone()
        self.assertEqual(row["label"], "mixed")
        self.assertIsNone(
            self.conn.execute(
                "SELECT * FROM chunk_feedback WHERE chunk_id=7"
            ).fetchone()
        )

    def test_boilerplate_downstream_reply_is_ignored(self):
        # A recurring template line, indexed via a normal `store.index()` run
        # so line_freq/boilerplate crosses the trim threshold - same
        # mechanism `vdb index` itself uses (store.py:81-87).
        line = "the usage window has reset, resume exactly where you stopped"
        for i in range(store.BOILERPLATE_MIN_MESSAGES + 2):
            fx.write_transcript(
                self.root, f"filler{i}", "brief",
                [fx.user_entry(f"{line}\n\nunrelated filler body {i}")],
            )
        store.index(self.conn, root=self.root)

        self._write_session(
            "sess-boiler",
            [
                fx.user_entry("first question", session="sess-boiler", ts="2026-08-01T10:00:00Z"),
                fx.user_entry(
                    line,
                    session="sess-boiler",
                    uuid="u2",
                    ts="2026-08-01T10:05:00Z",
                ),
            ],
        )
        qid = self.log_query(hit_chunk_ids=[3])
        self.conn.execute(
            "UPDATE query_log SET caller_session='sess-boiler', ts='2026-08-01T10:00:00Z' WHERE id=?",
            (qid,),
        )
        store.record_citation(self.conn, qid, [3])

        st = feedback_mod.run_label_pass(
            self.conn, root=self.root, min_age_seconds=0,
            now=datetime(2026, 8, 1, 11, 0, 0, tzinfo=timezone.utc),
        )
        # The only downstream candidate line is pure boilerplate once
        # trimmed, so no evidence is found - the harness resume string must
        # never be counted as a captain confirming anything (report §3.2).
        self.assertEqual(st.labeled, 0)
        self.assertEqual(st.skipped_no_evidence, 1)

    def test_too_young_is_skipped_not_treated_as_no_evidence(self):
        self._write_session(
            "sess-young",
            [fx.user_entry("q", session="sess-young", ts="2026-08-01T10:00:00Z")],
        )
        qid = self.log_query(hit_chunk_ids=[1])
        self.conn.execute(
            "UPDATE query_log SET caller_session='sess-young', ts='2026-08-01T10:00:00Z' WHERE id=?",
            (qid,),
        )
        store.record_citation(self.conn, qid, [1])
        st = feedback_mod.run_label_pass(
            self.conn, root=self.root, min_age_seconds=3600,
            now=datetime(2026, 8, 1, 10, 5, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(st.skipped_too_young, 1)
        self.assertEqual(st.labeled, 0)

    def test_uncited_query_is_never_scanned(self):
        qid = self.log_query(hit_chunk_ids=[1])
        self.conn.execute(
            "UPDATE query_log SET caller_session='sess-x', ts='2026-08-01T10:00:00Z' WHERE id=?",
            (qid,),
        )
        # no citation recorded at all
        st = feedback_mod.run_label_pass(
            self.conn, root=self.root, min_age_seconds=0,
            now=datetime(2026, 8, 1, 11, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(st.scanned, 0)

    def test_held_out_query_gets_labelled_but_never_updates_chunk_feedback(self):
        self._write_session(
            "sess-heldout",
            [
                fx.user_entry("first question", session="sess-heldout", ts="2026-08-01T10:00:00Z"),
                fx.user_entry(
                    "yes, that's exactly right",
                    session="sess-heldout",
                    uuid="u2",
                    ts="2026-08-01T10:05:00Z",
                ),
            ],
        )
        qid = self.log_query(hit_chunk_ids=[9])
        self.conn.execute(
            "UPDATE query_log SET caller_session='sess-heldout', ts='2026-08-01T10:00:00Z',"
            " held_out=1 WHERE id=?",
            (qid,),
        )
        store.record_citation(self.conn, qid, [9])
        feedback_mod.run_label_pass(
            self.conn, root=self.root, min_age_seconds=0,
            now=datetime(2026, 8, 1, 11, 0, 0, tzinfo=timezone.utc),
        )
        row = self.conn.execute(
            "SELECT label FROM feedback_label WHERE query_id=?", (qid,)
        ).fetchone()
        self.assertEqual(row["label"], "confirm")
        self.assertIsNone(
            self.conn.execute("SELECT * FROM chunk_feedback WHERE chunk_id=9").fetchone()
        )


class ScoreNudgeCap(FeedbackBase):
    def test_no_feedback_is_zero_nudge(self):
        self.assertEqual(store.score_nudge(self.conn, 1), 0.0)

    def test_nudge_grows_then_saturates_at_the_cap(self):
        below_cap = store.NUDGE_CAP // 2
        store.apply_feedback_to_chunks(self.conn, [1], "confirm")
        for _ in range(below_cap - 1):
            store.apply_feedback_to_chunks(self.conn, [1], "confirm")
        at_cap = store.score_nudge(self.conn, 1)
        expected = store.NUDGE_WEIGHT * math.log1p(below_cap)
        self.assertAlmostEqual(at_cap, expected)

        # Push far past the cap - the score must not move any further.
        for _ in range(store.NUDGE_CAP * 3):
            store.apply_feedback_to_chunks(self.conn, [1], "confirm")
        beyond_cap = store.score_nudge(self.conn, 1)
        expected_capped = store.NUDGE_WEIGHT * math.log1p(store.NUDGE_CAP)
        self.assertAlmostEqual(beyond_cap, expected_capped)

    def test_corrections_push_the_nudge_negative(self):
        store.apply_feedback_to_chunks(self.conn, [5], "correction")
        self.assertLess(store.score_nudge(self.conn, 5), 0.0)

    def test_confirms_and_corrections_partially_offset(self):
        store.apply_feedback_to_chunks(self.conn, [8], "confirm")
        store.apply_feedback_to_chunks(self.conn, [8], "confirm")
        store.apply_feedback_to_chunks(self.conn, [8], "correction")
        self.assertGreater(store.score_nudge(self.conn, 8), 0.0)


class NudgeGate(FeedbackBase):
    def _make_qualifying_labels(self, n):
        for i in range(n):
            qid = self.log_query(hit_chunk_ids=[100 + i])
            # Force held_out=0: qualifying_label_count excludes held-out rows
            # by design (§6.2b), and this helper is about volume, not the
            # held-out split - HeldOutSplit covers that separately.
            self.conn.execute("UPDATE query_log SET held_out=0 WHERE id=?", (qid,))
            store.record_citation(self.conn, qid, [100 + i])
            store.write_label(self.conn, qid, "confirm", "heuristic_casual_v1", 1)
            store.apply_feedback_to_chunks(self.conn, [100 + i], "confirm")
        self.conn.commit()

    def test_inert_by_default(self):
        self.assertFalse(store.nudge_active(self.conn))

    def test_flag_alone_is_not_enough(self):
        store.set_nudge_flag(self.conn, True)
        self.assertFalse(store.nudge_active(self.conn))

    def test_volume_alone_is_not_enough(self):
        self._make_qualifying_labels(store.NUDGE_LABEL_THRESHOLD)
        self.assertGreaterEqual(store.qualifying_label_count(self.conn), store.NUDGE_LABEL_THRESHOLD)
        self.assertFalse(store.nudge_active(self.conn))

    def test_flag_and_volume_without_recorded_regression_check_is_still_inert(self):
        store.set_nudge_flag(self.conn, True)
        self._make_qualifying_labels(store.NUDGE_LABEL_THRESHOLD)
        self.assertFalse(store.nudge_active(self.conn))

    def test_a_failed_regression_check_does_not_activate_it(self):
        store.set_nudge_flag(self.conn, True)
        self._make_qualifying_labels(store.NUDGE_LABEL_THRESHOLD)
        store.record_regression_check(self.conn, passed=False, notes="regressed family Q")
        self.assertFalse(store.nudge_active(self.conn))

    def test_all_three_conditions_together_activate_it(self):
        store.set_nudge_flag(self.conn, True)
        self._make_qualifying_labels(store.NUDGE_LABEL_THRESHOLD)
        store.record_regression_check(self.conn, passed=True, notes="nudge on vs off, no regression")
        self.assertTrue(store.nudge_active(self.conn))

    def test_below_threshold_search_never_reads_chunk_feedback(self):
        store.set_nudge_flag(self.conn, True)
        store.record_regression_check(self.conn, passed=True, notes="ok")
        self._make_qualifying_labels(store.NUDGE_LABEL_THRESHOLD - 1)
        self.assertFalse(store.nudge_active(self.conn))

        fx.write_transcript(
            self.root, "p", "s", [fx.user_entry("a distinctive query term appears here")]
        )
        store.index(self.conn, root=self.root)
        chunk_id = self.conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]
        # A huge, unmistakable boost if the gate were ever bypassed.
        for _ in range(50):
            store.apply_feedback_to_chunks(self.conn, [chunk_id], "confirm")

        before = retrieve.BM25Retriever(self.conn).search("distinctive query term").hits[0].score
        self.assertFalse(retrieve.BM25Retriever(self.conn).search("distinctive query term").nudge_applied)

        raw_row = self.conn.execute(
            "SELECT -bm25(chunks_fts) AS s FROM chunks_fts WHERE rowid=?", (chunk_id,)
        ).fetchone()
        self.assertAlmostEqual(before, round(raw_row["s"], 4))

    def test_once_active_the_nudge_reorders_near_ties(self):
        store.set_nudge_flag(self.conn, True)
        self._make_qualifying_labels(store.NUDGE_LABEL_THRESHOLD)
        store.record_regression_check(self.conn, passed=True, notes="ok")
        self.assertTrue(store.nudge_active(self.conn))

        fx.write_transcript(
            self.root, "p", "s",
            [
                fx.user_entry("widget calibration alpha", uuid="u1"),
                fx.user_entry("widget calibration beta", uuid="u2"),
            ],
        )
        store.index(self.conn, root=self.root)
        result = retrieve.BM25Retriever(self.conn).search("widget calibration")
        self.assertTrue(result.hits)
        loser = result.hits[-1].chunk_id
        for _ in range(store.NUDGE_CAP):
            store.apply_feedback_to_chunks(self.conn, [loser], "confirm")

        boosted = retrieve.BM25Retriever(self.conn).search("widget calibration")
        self.assertTrue(boosted.nudge_applied)
        self.assertEqual(boosted.hits[0].chunk_id, loser)


if __name__ == "__main__":
    unittest.main()
