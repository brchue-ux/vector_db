# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## What this project is

Local retrieval over the captain's own Claude Code history — **for an agent to call, not a human
to search** (`vdb query`; no UI is planned, ever). Ingest, cleaning, message-boundary chunking,
BM25, metadata pre-filtering, and a background indexer are built; see `README.md` for what it
does and how to run it, and `vdb/*.py` module docstrings for why each rule exists.

## The rules that are not negotiable by taste

The design is dictated by two measured studies, both outside this repo (they describe the
captain's private corpus and must never be committed): `vdbqual` (380 real queries, retrieval
quality generally) and `vdbtray` (596 real queries, specifically the chunk split threshold).
**Do not "improve" any of the following without re-running the relevant eval — each was measured,
and the intuitive alternative lost:**

- **One chunk per message**, paragraph splits near **1,000 chars** (`vdb/chunk.py:LIMIT` —
  `vdbtray`; not `vdbqual`'s untested 1,800 default), no overlap, nothing in the chunk but the
  message's own text. Message-boundary chunking is the largest effect in `vdbqual`; the 1,000
  threshold is the lowest one `vdbtray` found with no measured recall/MRR cost on any query
  family. Metadata headers and neighbouring context both measurably hurt recall.
- **No automatic context injection.** Retrieval cannot tell when it has found nothing (raw-score
  AUC 0.551 dense / 0.530 BM25, both ≈ chance). Delivery is always an explicit command. The margin
  (rank1-to-rank10 score gap) carries real but modest signal for BM25 (AUC 0.645, weaker than the
  dense 0.724) — reported as a diagnostic (`Result.margin`, `Result.weak_signal`), never
  thresholded into a verdict. See `vdb/retrieve.py`'s docstring before citing an AUC number for
  this system — the dense-model figures do not transfer to BM25's unbounded score.
- **No reranker.** Measured: it made the best configuration worse.
- **Templated messages are trimmed, never deleted.** Deleting them collapses
  session-topic recall.
- Prose only; tool I/O stays out (size, not content, is the reason).
- **Metadata (project/session/date/role) is a filter (`WHERE`), never embedded in chunk text.**
  Embedding it measurably hurts recall; filtering is what makes this cheaper than reading a whole
  conversation.

## Sharp edges

- The corpus (`~/.claude/projects`) is read-only. Never write there.
- The index file is exactly as sensitive as the transcripts. It is gitignored,
  lives outside the repo by default, and is created 0600. Never commit corpus
  content, index files, or real transcript excerpts — test fixtures are
  synthesised in `tests/fixture.py`.
- Stdlib only, deliberately: BM25 comes from SQLite FTS5 (`bm25()` is Okapi with
  k1=1.2, b=0.75). Keep phase 1 dependency-free.
- Tests: `python -m unittest discover -s tests -t .` (no pytest on this box).
- Background indexing is a `systemd --user` timer (`scripts/install-timer.sh` /
  `uninstall-timer.sh`, units in `systemd/`), not a daemon of its own. Installing/testing it for
  real writes to `~/.config/systemd/user` and enables a real timer — don't run the install script
  from a disposable worktree whose path won't survive; render/verify the unit files instead.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
