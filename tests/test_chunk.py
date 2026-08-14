"""Message-boundary chunking (report §11.1-§11.3, §13.1).

The chunking contract is the single biggest measured quality lever in the study,
so it is tested as a contract: one chunk per message, split at paragraph
boundaries near 1,800 chars, no overlap, message text only.
"""

import unittest

from vdb.chunk import LIMIT, raw_pieces, split_message


def para(n_chars: int, ch: str = "x") -> str:
    return (ch * 9 + " ") * (n_chars // 10)


class ChunkContract(unittest.TestCase):
    def test_short_message_is_exactly_one_chunk(self):
        text = "a question\n\nwith two paragraphs"
        self.assertEqual(split_message(text), [text])

    def test_limit_default_is_1800(self):
        self.assertEqual(LIMIT, 1800)

    def test_message_at_the_limit_is_not_split(self):
        text = "y" * LIMIT
        self.assertEqual(len(split_message(text)), 1)

    def test_long_message_splits_on_paragraph_boundaries(self):
        paras = [f"P{i} " + para(900) for i in range(4)]
        text = "\n\n".join(paras)
        chunks = split_message(text)
        self.assertGreater(len(chunks), 1)
        # every chunk starts at a paragraph start, i.e. never mid-sentence
        for c in chunks:
            self.assertRegex(c, r"^P\d ")

    def test_no_chunk_exceeds_the_limit(self):
        text = "\n\n".join(para(1500) for _ in range(6))
        for c in split_message(text):
            self.assertLessEqual(len(c), LIMIT)

    def test_no_overlap_between_consecutive_chunks(self):
        text = "\n\n".join(f"P{i} " + para(900) for i in range(6))
        chunks = split_message(text)
        for a, b in zip(chunks, chunks[1:]):
            self.assertNotIn(b[:40], a)
        # and the concatenation is not longer than the source: nothing repeated
        self.assertLessEqual(sum(len(c) for c in chunks), len(text))

    def test_long_paragraph_falls_back_to_line_boundaries(self):
        text = "\n".join(f"line {i} " + para(200) for i in range(20))
        chunks = split_message(text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(c.startswith("line "))

    def test_single_unbroken_run_falls_back_to_hard_split(self):
        text = "z" * (LIMIT * 2 + 50)
        chunks = split_message(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual("".join(chunks), text)

    def test_split_is_lossless(self):
        text = "\n\n".join(f"P{i}\n" + para(1400) for i in range(5))
        self.assertEqual("".join(raw_pieces(text)), text)

    def test_chunk_holds_only_the_message_text(self):
        # §11.3: a provenance header costs -0.041 recall@10 and neighbouring
        # context halves cross-session recall. Chunk text is the message, alone.
        text = "the message body"
        self.assertEqual(split_message(text), ["the message body"])

    def test_empty_and_whitespace(self):
        self.assertEqual(split_message(""), [])
        self.assertEqual(split_message("   \n\n  "), [])


if __name__ == "__main__":
    unittest.main()
