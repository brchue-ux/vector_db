# vector_db

Local retrieval over your own Claude Code history: ask a question, get back passages
from past sessions with enough provenance to go read the rest.

**Status: phase 1 shipped.** Ingest, cleaning, message-boundary chunking, a BM25
keyword index, and an explicit query command. No model, no vectors, no network,
no dependencies beyond the Python standard library. Phase 2 adds a dense index
and reciprocal-rank fusion.

Everything here implements the retrieval-quality study (`vdbqual` report §13),
which measured 380 real queries against this corpus. The design decisions below
are its findings, not preferences.

## What it does

* **Ingests** `~/.claude/projects/**/*.jsonl` transcripts and `**/memory/*.md`
  cards, strictly read-only.
* **Cleans** harness machinery out of user turns — `<system-reminder>`,
  `<task-notification>`, slash-command echoes, interrupt markers (§1.1). A turn
  that was nothing but machinery is dropped.
* **Excludes tool I/O.** Tool calls, tool results and attachments stay out. Not
  because command output is noisy — measured, it is *less* distracting per chunk
  than more prose — but because it is 22.6× the volume, and index size is what
  costs recall (§6.3–§6.4).
* **Trims invariant template lines** that recur across ≥50 messages, without
  deleting the templated messages. Deleting them measurably hurts: the task body
  inside a launch brief is the best summary of that session that exists
  (§7, §11.5).
* **Chunks one message per chunk**, splitting long messages at paragraph
  boundaries near 1,800 chars, no overlap, nothing in the chunk but the
  message's own text. This is the single largest measured effect in the study:
  +0.085 recall@10 over 600-char windows *at matched chunk size*
  (CI [+0.044, +0.126]). Metadata headers cost −0.041; neighbouring context
  halves cross-session recall (§11.1–§11.3).
* **Redacts secret-shaped strings on ingest** — private keys, cloud and API
  tokens, JWTs, bearer headers, OAuth callback codes, long high-entropy tokens
  (§13.1, F9).
* **Indexes with Okapi BM25** (k1=1.2, b=0.75) via SQLite FTS5, incrementally.

## Install

Nothing to install. Python 3.11+ with `sqlite3` compiled with FTS5, which is the
default.

## Index

```sh
python -m vdb index                 # build, or update only what changed
python -m vdb index --rebuild       # start from empty
python -m vdb index --root /path    # a different corpus root
python -m vdb stats                 # what is in the index
python -m vdb boilerplate           # the template lines currently being trimmed
```

Incremental by default: a file whose size, mtime and SHA-256 are unchanged is
skipped; a changed file has its chunks replaced; a deleted file has its chunks
removed. Cold start on the full corpus takes about 40 seconds and ~150 MB of
RAM; an update with nothing to do takes about 0.2 seconds.

The index lives at `$XDG_DATA_HOME/vdb/index.sqlite3` (default
`~/.local/share/vdb/index.sqlite3`), created 0600 inside a 0700 directory.
Override with `--db` or `$VDB_DB`.

> The index contains your prose. It is exactly as sensitive as the transcripts.
> Do not commit it, sync it, or copy it anywhere you would not copy `~/.claude`.
> The secret filter reduces the exposure; it does not eliminate it (open
> question O14 leaves its false-negative rate on this material unmeasured).

## Query

```sh
python -m vdb query "why did we pick sqlite over duckdb"
python -m vdb query "flaky websocket reconnect" -k 5 --full
python -m vdb query "auth redesign" --project firstmate --since 2026-06-01
python -m vdb query "auth redesign" --json          # for programmatic callers
```

Each hit carries project, role, timestamp, session id and source file, so a
passage can always be traced back to the session it came from.

Retrieval is **asked for, never injected.** The study's load-bearing negative
finding is that this system cannot tell when it has found nothing: the top hit's
score is statistically indistinguishable whether or not the answer is in the
corpus (AUC 0.551, F6). Anything that pastes top-k into every session will paste
confident nonsense a large fraction of the time with no signal that it did. The
rank-1-to-rank-10 margin does carry signal (AUC 0.724) and is printed as a
diagnostic — it is deliberately not thresholded, because nobody has built or
validated that classifier yet (O13).

## Tests

```sh
python -m unittest discover -s tests -t .
```

All fixtures are constructed in `tests/fixture.py`. No real transcript content
is committed to this repository, ever.

## Layout

| module | what it owns |
|---|---|
| `vdb/ingest.py` | walking the corpus, parsing transcripts and memory cards, prose extraction |
| `vdb/clean.py` | harness-block stripping and boilerplate trimming |
| `vdb/secrets.py` | the secret-shaped-string filter |
| `vdb/chunk.py` | message-boundary chunking |
| `vdb/store.py` | SQLite chunk store, FTS5 index, incremental update |
| `vdb/retrieve.py` | BM25 query path, `Hit`/`Result` shapes |
| `vdb/cli.py` | `index` / `query` / `stats` / `boilerplate` |

## Phase 2

A dense index (`bge-small-en-v1.5` to start) fused with BM25 by reciprocal-rank
fusion — the only configuration measured to beat BM25 alone, and the one that
degrades most slowly as the corpus grows (§12 finding 4, §11.7). The seam for it
is small on purpose: `chunks.id` is a stable integer key a vector file can
address by row, and `retrieve.BM25Retriever.search()` is the shape a second
retriever has to match so two ranked lists can be fused. There is no plugin
architecture here for one future retriever.

Not planned, and measured rather than assumed: **no reranker** (it made the best
configuration worse, §11.4) and **no automatic context injection** (§13.2).
