"""Index build, incremental re-index, and query with provenance."""

import os
import tempfile
import unittest
from pathlib import Path

from tests import fixture as fx
from vdb import ingest, retrieve, store


class IndexBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "projects"
        self.root.mkdir()
        self.db = base / "index.sqlite3"
        self.conn = store.connect(self.db)
        self.addCleanup(self.conn.close)

    def index(self, **kw):
        return store.index(self.conn, root=self.root, **kw)

    def search(self, q, **kw):
        return retrieve.BM25Retriever(self.conn).search(q, **kw)


class ColdStart(IndexBase):
    def test_empty_corpus_is_not_an_error(self):
        st = self.index()
        self.assertEqual((st.scanned, st.chunks), (0, 0))
        self.assertEqual(store.stats(self.conn)["chunks"], 0)

    def test_query_on_empty_index_returns_nothing(self):
        self.index()
        self.assertEqual(self.search("anything").hits, [])

    def test_index_and_query_with_provenance(self):
        fx.write_transcript(
            self.root,
            "proj-alpha",
            "sess1",
            [
                fx.user_entry(
                    "how do we stop the flaky websocket reconnect loop?",
                    session="sess-1",
                    ts="2026-07-04T09:00:00Z",
                ),
                fx.assistant_entry([{"type": "text", "text": "add exponential backoff to the "
                                                             "reconnect handler"}]),
            ],
        )
        self.index()
        res = self.search("websocket reconnect loop")
        self.assertTrue(res.hits)
        top = res.hits[0]
        self.assertIn("websocket", top.text)
        self.assertEqual(top.session_id, "sess-1")
        self.assertEqual(top.project, "proj-alpha")
        self.assertEqual(top.ts, "2026-07-04T09:00:00Z")
        self.assertTrue(top.source.endswith("sess1.jsonl"))
        self.assertIn("proj-alpha", top.provenance())

    def test_memory_cards_are_indexed(self):
        fx.write_memory(self.root, "proj-alpha", "card.md",
                        "the captain prefers hyphenated flags over camelCase")
        self.index()
        hits = self.search("hyphenated flags").hits
        self.assertEqual(hits[0].role, "memory")

    def test_tool_output_is_not_searchable(self):
        fx.write_transcript(
            self.root,
            "proj-alpha",
            "sess1",
            [
                fx.user_entry("run the migration"),
                fx.tool_result_entry("ERROR: relation 'widgets' already exists"),
            ],
        )
        self.index()
        self.assertEqual(self.search("widgets relation exists").hits, [])

    def test_ranking_prefers_the_relevant_passage(self):
        fx.write_transcript(
            self.root,
            "p",
            "s",
            [
                fx.user_entry("the deploy pipeline keeps timing out on the docker build step",
                              uuid="u1"),
                fx.user_entry("what should we have for lunch", uuid="u2"),
            ],
        )
        self.index()
        self.assertIn("docker", self.search("docker build timing out").hits[0].text)

    def test_secret_never_reaches_the_index(self):
        fx.write_transcript(
            self.root, "p", "s",
            [fx.user_entry("key AKIAIOSFODNN7EXAMPLE is in the env file")],
        )
        st = self.index()
        self.assertEqual(st.secrets.get("AWS-KEY"), 1)
        self.assertEqual(self.search("AKIAIOSFODNN7EXAMPLE").hits, [])
        rows = list(self.conn.execute("SELECT text FROM chunks_fts"))
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", "\n".join(r[0] for r in rows))

    def test_db_file_is_not_world_readable(self):
        self.index()
        self.assertEqual(os.stat(self.db).st_mode & 0o077, 0)

    def test_query_with_no_searchable_terms_raises(self):
        self.index()
        with self.assertRaises(ValueError):
            self.search("!!! ???")

    def test_fts_operator_words_are_not_syntax(self):
        fx.write_transcript(self.root, "p", "s", [fx.user_entry("we chose postgres not mysql")])
        self.index()
        self.assertTrue(self.search("postgres NOT mysql AND (something)").hits)


class Incremental(IndexBase):
    def setUp(self):
        super().setUp()
        self.p1 = fx.write_transcript(
            self.root, "p1", "s1", [fx.user_entry("first session about parser tokens")]
        )
        self.p2 = fx.write_transcript(
            self.root, "p2", "s2", [fx.user_entry("second session about billing invoices")]
        )
        first = self.index()
        self.assertEqual(first.indexed, 2)

    def test_unchanged_corpus_reindexes_nothing(self):
        st = self.index()
        self.assertEqual((st.indexed, st.skipped, st.removed), (0, 2, 0))
        self.assertEqual(store.stats(self.conn)["chunks"], 2)

    def test_appended_session_is_picked_up_without_touching_the_rest(self):
        fx.append_entries(
            self.p1, [fx.user_entry("later turn about retry backoff", uuid="u9")]
        )
        os.utime(self.p1, ns=(0, 2_000_000_000))
        st = self.index()
        self.assertEqual((st.indexed, st.skipped), (1, 1))
        self.assertTrue(self.search("retry backoff").hits)
        # the untouched file's chunk is still there exactly once
        self.assertEqual(len(self.search("billing invoices").hits), 1)

    def test_changed_file_does_not_leave_stale_chunks(self):
        self.p2.write_text("", encoding="utf-8")
        os.utime(self.p2, ns=(0, 3_000_000_000))
        self.index()
        self.assertEqual(self.search("billing invoices").hits, [])
        self.assertEqual(store.stats(self.conn)["chunks"], 1)

    def test_new_file_is_added(self):
        fx.write_transcript(self.root, "p3", "s3", [fx.user_entry("a brand new topic: kerning")])
        st = self.index()
        self.assertEqual(st.indexed, 1)
        self.assertTrue(self.search("kerning").hits)

    def test_deleted_file_is_removed_from_the_index(self):
        self.p2.unlink()
        st = self.index()
        self.assertEqual(st.removed, 1)
        self.assertEqual(self.search("billing invoices").hits, [])
        self.assertEqual(store.stats(self.conn)["files"], 1)

    def test_touched_but_unchanged_file_is_not_reindexed(self):
        os.utime(self.p1, ns=(0, 9_000_000_000))
        st = self.index()
        self.assertEqual((st.indexed, st.skipped), (0, 2))

    def test_rebuild_produces_the_same_index(self):
        before = store.stats(self.conn)["chunks"]
        st = self.index(rebuild=True)
        self.assertEqual(st.indexed, 2)
        self.assertEqual(store.stats(self.conn)["chunks"], before)

    def test_no_duplicate_chunks_after_repeated_runs(self):
        for _ in range(3):
            self.index()
        self.assertEqual(len(self.search("parser tokens").hits), 1)


class Boilerplate(IndexBase):
    LINE = "You are a crewmate: an autonomous worker agent managed by firstmate, working alone."

    def test_repeated_template_line_is_trimmed_but_messages_survive(self):
        for i in range(store.BOILERPLATE_MIN_MESSAGES + 5):
            fx.write_transcript(
                self.root,
                f"proj{i}",
                "brief",
                [fx.user_entry(f"{self.LINE}\n\n# Task\nunique body {i} about widget calibration")],
            )
        self.index()
        self.assertGreaterEqual(store.stats(self.conn)["boilerplate_lines"], 1)
        texts = [r[0] for r in self.conn.execute("SELECT text FROM chunks_fts")]
        self.assertEqual(len(texts), store.BOILERPLATE_MIN_MESSAGES + 5)
        for t in texts:
            self.assertNotIn("autonomous worker agent", t)
        # §7/§11.5: the templated messages themselves are still indexed.
        self.assertTrue(self.search("widget calibration unique body 7").hits)

    def test_rare_long_line_is_not_trimmed(self):
        fx.write_transcript(
            self.root, "p", "s",
            [fx.user_entry("this long line appears exactly once in the whole corpus, so it stays")],
        )
        self.index()
        self.assertTrue(self.search("appears exactly once").hits)


class Seam(unittest.TestCase):
    def test_retriever_shape_is_what_phase_two_fuses(self):
        # Phase 2 adds a dense retriever and RRF over two ranked lists. The only
        # contract that needs to hold is: search() -> Result of ranked Hits with
        # stable chunk ids.
        self.assertTrue(hasattr(retrieve.BM25Retriever, "search"))
        hit = retrieve.Hit(chunk_id=7, rank=1, score=1.0, text="x")
        self.assertEqual((hit.chunk_id, hit.rank), (7, 1))
        res = retrieve.Result(query="q", hits=[hit])
        self.assertIsNone(res.margin)

    def test_margin_is_reported_not_thresholded(self):
        hits = [retrieve.Hit(chunk_id=i, rank=i, score=10.0 - i, text="x") for i in range(1, 11)]
        res = retrieve.Result(query="q", hits=hits)
        self.assertAlmostEqual(res.margin, 9.0)


class Defaults(unittest.TestCase):
    def test_default_corpus_root(self):
        self.assertEqual(ingest.DEFAULT_ROOT.name, "projects")
        self.assertIn(".claude", str(ingest.DEFAULT_ROOT))

    def test_index_lives_outside_the_repo(self):
        self.assertNotIn(str(Path.cwd()), str(store.DEFAULT_DB))


if __name__ == "__main__":
    unittest.main()
