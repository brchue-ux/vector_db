"""CLI-surfaced pieces of the implicit-feedback loop (task brief "Change 1").

Covers: `vdb query`'s citation reminder appearing in both human and `--json`
output, and the citation-compliance visibility (`store.citation_compliance`,
surfaced through `vdb stats`).
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests import fixture as fx
from vdb import cli, store


class CliBase(unittest.TestCase):
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


class QueryCitationReminder(CliBase):
    def test_human_output_states_citation_is_expected_with_the_exact_command(self):
        code, out = self.run_cli(["query", "distinctive query term"])
        self.assertEqual(code, 0)
        self.assertIn("query_id:", out)
        self.assertIn("citation is expected", out)
        self.assertIn("vdb feedback", out)
        self.assertIn("--used <chunk_id>", out)
        self.assertIn('--used ""', out)

    def test_json_output_carries_a_citation_reminder_field(self):
        code, out = self.run_cli(["query", "distinctive query term", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["citation_expected"])
        self.assertIn(f"vdb feedback {payload['query_id']} --used", payload["citation_command"])
        self.assertIn("citation is expected", payload["citation_reminder"])
        self.assertIn(str(payload["query_id"]), payload["citation_command"])

    def test_no_hits_still_reminds_to_cite_used_nothing(self):
        code, out = self.run_cli(["query", "zqxvbnkjplomfghqz"])
        self.assertEqual(code, 1)
        self.assertIn("query_id:", out)
        self.assertIn('--used ""', out)

    def test_no_hits_json_still_carries_reminder(self):
        code, out = self.run_cli(["query", "zqxvbnkjplomfghqz", "--json"])
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertTrue(payload["citation_expected"])
        self.assertIn("citation_command_used_nothing", payload)


class CitationComplianceVisibility(CliBase):
    def test_store_function_counts_cited_vs_uncited(self):
        conn = store.connect(self.db)
        qid1 = store.log_query(
            conn, query_text="a", filters={}, k_requested=10, hit_chunk_ids=[1],
            hit_scores=[1.0], margin=None, weak_signal=True,
        )
        qid2 = store.log_query(
            conn, query_text="b", filters={}, k_requested=10, hit_chunk_ids=[2],
            hit_scores=[1.0], margin=None, weak_signal=True,
        )
        store.record_citation(conn, qid1, [1])
        # qid2 left uncited on purpose.
        cc = store.citation_compliance(conn)
        self.assertEqual(cc["queries"], 2)
        self.assertEqual(cc["cited"], 1)
        self.assertEqual(cc["uncited"], 1)
        self.assertAlmostEqual(cc["citation_rate"], 0.5)
        conn.close()

    def test_window_limits_to_most_recent_queries(self):
        conn = store.connect(self.db)
        qid_old = store.log_query(
            conn, query_text="old", filters={}, k_requested=10, hit_chunk_ids=[1],
            hit_scores=[1.0], margin=None, weak_signal=True,
        )
        store.record_citation(conn, qid_old, [1])
        for i in range(3):
            store.log_query(
                conn, query_text=f"new{i}", filters={}, k_requested=10, hit_chunk_ids=[1],
                hit_scores=[1.0], margin=None, weak_signal=True,
            )
        cc = store.citation_compliance(conn, window=3)
        self.assertEqual(cc["queries"], 3)
        self.assertEqual(cc["cited"], 0)
        conn.close()

    def test_empty_log_has_no_rate(self):
        conn = store.connect(self.db)
        cc = store.citation_compliance(conn)
        self.assertEqual(cc["queries"], 0)
        self.assertIsNone(cc["citation_rate"])
        conn.close()

    def test_vdb_stats_surfaces_compliance(self):
        code, out = self.run_cli(["query", "distinctive query term"])
        self.assertEqual(code, 0)
        code, out = self.run_cli(["stats"])
        self.assertEqual(code, 0)
        self.assertIn("citation compliance", out)
        self.assertIn("0/1", out)

        code, out = self.run_cli(["stats", "--json"])
        payload = json.loads(out)
        self.assertIn("citation_compliance", payload)
        self.assertEqual(payload["citation_compliance"]["queries"], 1)
        self.assertEqual(payload["citation_compliance"]["cited"], 0)


if __name__ == "__main__":
    unittest.main()
