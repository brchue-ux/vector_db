# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## What this project is

Local retrieval over the captain's own Claude Code history — **for an agent to call, not a human
to search** (`vdb query`; no UI is planned, ever). Ingest, cleaning, message-boundary chunking,
BM25, metadata pre-filtering, a background indexer, implicit-feedback logging/citation/label
extraction, and a calibrated confidence gate on every result are built; see `README.md` for what
it does and how to run it, and `vdb/*.py` module docstrings for why each rule exists.

## The implicit-feedback learning loop

Spec: `data/vdbfeedback/report.md` (outside this repo, never committed — describes the captain's
private corpus). Logging (`query_log`), citation (`vdb feedback`), and heuristic label extraction
(`vdb label`, a second systemd timer) are shipped and score-neutral.

**Citation is expected after every `vdb query` call that gets acted on**, not an optional
afterthought: `vdb query`'s human and `--json` output both print the exact `vdb feedback
<query_id> --used ...` follow-up on every call (this cannot be enforced across two separate CLI
invocations, so the tool insists loudly instead). Check whether that's actually happening with
`vdb stats` (`store.citation_compliance()`) — the fraction of the most recent 200 `query_log` rows
with a matching `feedback_citation`. There is no inference-based fallback for a missing citation
and there will not be one — the report's §4.3 already measured diluted/guessed credit as worse
than no signal.

The score-affecting nudge (`vdb/store.py:score_nudge`, plugged into `BM25Retriever.search()`) is
implemented but **must stay off**, gated by two independent conditions that must *both* hold:

- **Global** (`store.nudge_active()`, a property of the whole retrieval system): the operator flag
  (`vdb nudge --enable`), **60** high-confidence, cited, non-held-out labels corpus-wide
  (`store.NUDGE_LABEL_THRESHOLD` — reduced from the report's original 300 now that a second,
  per-chunk gate also exists; 60 is not a new guess, it is the same floor the report's own §6.3
  already names as trustworthy — `vdbqual`'s smallest reliable family, C, n=60 — for "there is
  enough data for the regression check to mean something"), and a regression check against the
  frozen `vdbqual`/`vdbtray`/`vdbaccuracy` static eval sets explicitly recorded as passed (`vdb
  nudge --record-check`, procedure in `scripts/nudge-regression-check.md`).
- **Per-chunk** (`store.score_nudge()`'s own check): a given `chunk_id` only gets nudged once
  *that chunk* has earned at least **3** of its own high-confidence, cited, non-held-out labels
  (`store.NUDGE_PER_CHUNK_MIN`) — see that constant's docstring for why 3 (a single citation
  already produces a small nonzero nudge under the capped-log formula, so the threshold's job is
  requiring basic multiplicity, not blocking the first data point). Below its own threshold, a
  chunk is not nudged at all — `score' = score` — even once the global gate is satisfied.
  `vdb nudge --chunk <id>` inspects one chunk's own gate.

**Do not flip `--enable` or record a `pass` without actually running that check** — recording a
fabricated pass defeats the entire safeguard. `query_log.held_out` (~10% of queries, decided by
hashing `query_id` at insert time) never contributes to `chunk_feedback` and is meant to become
the running eval set for that check.

Attribution uses `$CLAUDE_CODE_SESSION_ID` (verified to equal the transcript's own
`sessionId`/filename stem — report §4.4's open question, resolved: yes, it's available), captured
automatically by `vdb query` into `caller_session` unless `--caller-session` overrides it.

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
- **Confidence gate cut points (margin 75.0 / 126.0, `vdb/retrieve.py:CONFIDENCE_LOW_MARGIN`/
  `CONFIDENCE_HIGH_MARGIN`) are fit on query families Q and C only, not the whole corpus of query
  shapes.** Pooling all four families inverts the signal (family T's session-level gold lets a
  genuine hit have a flat score curve, which Simpson's-paradoxes the pooled AUC negative — see that
  constant's docstring for the full measurement). Don't re-tune these by eyeballing hit rates or
  widen the calibration population to "more data" without re-checking this confound first.
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
- Background indexing and label extraction are two `systemd --user` timers
  (`scripts/install-timer.sh` / `uninstall-timer.sh`, units in `systemd/`: `vdb-index` and
  `vdb-label`), not a daemon of its own. Installing/testing it for real writes to
  `~/.config/systemd/user` and enables real timers — don't run the install script from a
  disposable worktree whose path won't survive; render/verify the unit files instead.
- `vdb label`'s session-transcript lookup (`vdb/feedback.py:_find_session_transcript`) is an exact
  `rglob` for `<session_id>.jsonl` under the corpus root — never "most recently modified file in
  the project directory" (report Appendix A item 4: this corpus routinely has multiple concurrent
  sessions per project directory, so that heuristic misattributes).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
