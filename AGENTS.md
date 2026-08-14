# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## What this project is

Local retrieval over the captain's own Claude Code history. Phase 1 (ingest,
cleaning, message-boundary chunking, BM25) is built; see `README.md` for what it
does and how to run it, and `vdb/*.py` module docstrings for why each rule exists.

## The rules that are not negotiable by taste

The design is dictated by the `vdbqual` retrieval-quality study, measured on 380
real queries with confidence intervals. **Do not "improve" any of the following
without re-running that eval — each was measured, and the intuitive alternative
lost:**

- **One chunk per message**, paragraph splits near 1,800 chars, no overlap,
  nothing in the chunk but the message's own text. Largest effect in the study.
  Metadata headers and neighbouring context both measurably hurt.
- **No automatic context injection.** Retrieval cannot tell when it has found
  nothing (AUC 0.551). Delivery is always an explicit command.
- **No reranker.** Measured: it made the best configuration worse.
- **Templated messages are trimmed, never deleted.** Deleting them collapses
  session-topic recall.
- Prose only; tool I/O stays out (size, not content, is the reason).

## Sharp edges

- The corpus (`~/.claude/projects`) is read-only. Never write there.
- The index file is exactly as sensitive as the transcripts. It is gitignored,
  lives outside the repo by default, and is created 0600. Never commit corpus
  content, index files, or real transcript excerpts — test fixtures are
  synthesised in `tests/fixture.py`.
- Stdlib only, deliberately: BM25 comes from SQLite FTS5 (`bm25()` is Okapi with
  k1=1.2, b=0.75). Keep phase 1 dependency-free.
- Tests: `python -m unittest discover -s tests -t .` (no pytest on this box).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
