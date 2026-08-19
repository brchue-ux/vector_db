# Regression check before flipping the score nudge on (report §6.2c)

The implicit-feedback score nudge (`vdb/store.py:score_nudge`, plugged in at
`vdb/retrieve.py`'s `BM25Retriever.search()`) must never move from off to on,
and its weight `w` (`store.NUDGE_WEIGHT`) or cap (`store.NUDGE_CAP`) must
never change, without first re-running it against the frozen static eval
sets this domain already has, and rejecting the change if it measurably
regresses them. This is not optional and there is no shortcut around
actually running it — do not record a "pass" without doing so.

## Where the eval sets/harnesses live

Outside this repo, per each report's own Appendix B (do not commit anything
from them into `vector_db`):

- `vdbqual` (380 real queries, retrieval quality generally) — Appendix B.7.
- `vdbtray` (596 real queries, chunk-granularity) — its own Appendix B.
- `vdbaccuracy` (pre-search narrowing / filtering) — Appendix B.

Find the current location of each lab's scratch directory and harness script
from whichever report is closest at hand; they are not reproduced here
because they are outside this repo by the same rule everything else in this
domain follows (`AGENTS.md` "Sharp edges").

## What "run the check" means

1. Build (or reuse) an index with a real `chunk_feedback` population — this
   only makes sense once `vdb nudge --json` reports
   `qualifying_labels >= threshold`.
2. Run each static eval harness twice against the *same* index: once with
   `vdb/store.py`'s `NUDGE_WEIGHT` effectively zeroed out (nudge off — the
   default, pre-this-check state) and once with the nudge as it will ship
   (on, with its real `w`/`CAP`).
3. Compare recall@10 / MRR per family, the same metrics those reports
   already report. "Measurably regresses" means outside the harness's own
   noise floor (each report states its own confidence intervals — use those,
   not a fresh judgment call).

## Recording the result

```sh
python -m vdb nudge --record-check pass --notes "vdbqual/vdbtray/vdbaccuracy re-run 2026-XX-XX, no regression on any family, see <path to your run's notes>"
# or, if it regressed something:
python -m vdb nudge --record-check fail --notes "family Q recall@10 dropped 0.56 -> 0.51, rejected"
```

This writes to the `meta` table in the index database (report §6.2c is a
runtime gate, not documentation) and is one of three conditions in the
GLOBAL half of the gate that `store.nudge_active()` requires — see that
function's docstring (a PER-CHUNK gate, `store.NUDGE_PER_CHUNK_MIN`, applies
on top of this and is unaffected by it). Recording `fail` is exactly as
valid an outcome as `pass`; it is what stops a bad change from ever reaching
`nudge_active()`.

`vdb nudge --enable` (the separate operator flag), a `pass`-recorded
regression check, and `store.NUDGE_LABEL_THRESHOLD`+ (60+) qualifying labels
corpus-wide are all three required for the GLOBAL gate — missing any one of
them keeps the nudge inert, provably, at query time. A specific chunk also
needs its own `store.NUDGE_PER_CHUNK_MIN` (3+) labels before it is nudged,
independently of the global gate being satisfied — see `AGENTS.md` and
`vdb/store.py`'s docstrings for both numbers.
