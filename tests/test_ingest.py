"""Ingest: what gets in and what stays out (report §1.1, §1.2, §6.3-§6.4)."""

import tempfile
import unittest
from pathlib import Path

from tests import fixture as fx
from vdb import ingest


class Ingest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def read(self, path):
        src = next(s for s in ingest.scan(self.root) if s.path == path)
        return ingest.read_messages(src)

    def test_prose_kept_tool_io_excluded(self):
        path = fx.write_transcript(
            self.root,
            "proj-a",
            "sess",
            [
                fx.user_entry("why is the linker failing?"),
                fx.assistant_entry(
                    [
                        {"type": "text", "text": "because libssl is missing"},
                        fx.tool_use_block(cmd="apt-get install libssl-dev"),
                    ]
                ),
                fx.tool_result_entry("Reading package lists... Done"),
                fx.thinking_entry(),
            ],
        )
        msgs, _ = self.read(path)
        texts = [m.text for m in msgs]
        self.assertEqual(texts, ["why is the linker failing?", "because libssl is missing"])
        for t in texts:
            self.assertNotIn("apt-get", t)
            self.assertNotIn("package lists", t)

    def test_harness_only_user_turn_is_dropped(self):
        path = fx.write_transcript(
            self.root,
            "proj-a",
            "sess",
            [
                fx.user_entry("<system-reminder>be nice</system-reminder>"),
                fx.user_entry("a real question", uuid="u2"),
            ],
        )
        msgs, _ = self.read(path)
        self.assertEqual([m.text for m in msgs], ["a real question"])

    def test_provenance_is_captured(self):
        path = fx.write_transcript(
            self.root, "proj-b", "sess", [fx.user_entry("hello", session="abc-123")]
        )
        msgs, _ = self.read(path)
        m = msgs[0]
        self.assertEqual(m.session_id, "abc-123")
        self.assertEqual(m.project, "proj-b")
        self.assertEqual(m.role, "user")
        self.assertEqual(m.ts, "2026-08-01T10:00:00Z")

    def test_sidechain_flagged_not_dropped(self):
        path = fx.write_transcript(
            self.root, "proj-a", "sess", [fx.user_entry("subagent work", sidechain=True)]
        )
        msgs, _ = self.read(path)
        self.assertTrue(msgs[0].is_sidechain)

    def test_secret_filter_runs_on_ingest(self):
        path = fx.write_transcript(
            self.root,
            "proj-a",
            "sess",
            [fx.user_entry("token is ghp_1234567890abcdefghijklmnopqrstuvwxyz ok?")],
        )
        msgs, tally = self.read(path)
        self.assertNotIn("ghp_1234567890", msgs[0].text)
        self.assertEqual(tally.get("GITHUB-TOKEN"), 1)

    def test_memory_card_gets_its_own_session(self):
        path = fx.write_memory(self.root, "proj-a", "prefers-tabs.md", "---\nname: x\n---\nbody")
        msgs, _ = self.read(path)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, "memory")
        self.assertEqual(msgs[0].session_id, "memory:proj-a/prefers-tabs.md")
        self.assertEqual(msgs[0].project, "proj-a")

    def test_scan_finds_transcripts_and_memory_only(self):
        fx.write_transcript(self.root, "proj-a", "sess", [fx.user_entry("hi")])
        fx.write_memory(self.root, "proj-a", "card.md", "body")
        (self.root / "proj-a" / "notes.md").write_text("not a memory card")
        (self.root / "proj-a" / "data.json").write_text("{}")
        kinds = sorted(s.kind for s in ingest.scan(self.root))
        self.assertEqual(kinds, ["memory", "transcript"])

    def test_missing_root_is_not_an_error(self):
        self.assertEqual(ingest.scan(self.root / "nope"), [])

    def test_malformed_json_line_is_skipped(self):
        path = fx.write_transcript(self.root, "proj-a", "sess", [fx.user_entry("good line")])
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        msgs, _ = self.read(path)
        self.assertEqual(len(msgs), 1)

    def test_corpus_is_never_modified(self):
        path = fx.write_transcript(self.root, "proj-a", "sess", [fx.user_entry("hi")])
        before = path.read_bytes(), path.stat().st_mtime_ns
        self.read(path)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)


if __name__ == "__main__":
    unittest.main()
