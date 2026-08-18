# vector_db

Local retrieval over your own Claude Code history — **for an agent to call, not a human to
search.** A Claude Code session invokes `vdb query` mid-task when it decides it needs past
context, and gets back a handful of short, provenance-tagged passages instead of loading a whole
conversation. There is deliberately no UI and no visual app here, and none is planned — the
consumer is an agent, not a human doing his own searching.

**Status: phase 1 + the agent-facing tool + implicit-feedback logging shipped.** Ingest,
cleaning, message-boundary chunking, a BM25 keyword index, metadata pre-filtering, a
deterministic agent-callable query command, a background indexer, and query/citation/label
logging for an implicit-feedback loop (`vdb feedback`, `vdb label`) — with its
score-affecting nudge built but shipped off (see "Feedback" below). No model, no vectors, no
network, no dependencies beyond the Python standard library. Phase 2 adds a dense index and
reciprocal-rank fusion.

Everything here implements the retrieval-quality study (`vdbqual` report §13) plus the follow-up
`vdbtray` chunk-granularity experiment that set the split threshold below. Both measured real
queries against this corpus with confidence intervals; the design decisions below are their
findings, not preferences.

**How reliable this actually is, stated plainly:** on the fullest-scale measurement available
(`vdbqual` §11.7), the best configuration finds a labelled correct passage in the top 10 for
roughly **three queries in four**. It cannot reliably tell you when it has found nothing (see
"Retrieval is asked for, never injected" below) — a returned passage is a candidate to read and
judge, not a verified answer. Read the passages; don't act on the ranking alone.

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
  boundaries near 1,000 chars, no overlap, nothing in the chunk but the
  message's own text. Message-boundary chunking itself is the single largest
  measured effect in the study: +0.085 recall@10 over 600-char windows *at
  matched chunk size* (CI [+0.044, +0.126]). Metadata headers cost −0.041;
  neighbouring context halves cross-session recall (§11.1–§11.3). The 1,000
  threshold is a separate, later finding (`vdbtray` chunk-granularity
  experiment, outside this repo) — see `vdb/chunk.py`'s docstring: it's the
  lowest threshold with no measured recall/MRR cost on any query family, and
  it nearly halves the size of the chunk that answers a query versus the
  untested 1,800 default `vdbqual` shipped with.
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
python -m vdb stats                 # what is in the index, plus citation compliance (see Feedback)
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

## Query — for an agent to call

```sh
python -m vdb query "why did we pick sqlite over duckdb"
python -m vdb query "flaky websocket reconnect" -k 5 --full
python -m vdb query "auth redesign" --project firstmate --since 2026-06-01 --until 2026-07-01
python -m vdb query "auth redesign" --session a3e6769e --role user
python -m vdb query "auth redesign" --json          # for programmatic callers — use this
```

### Filters — narrow before searching, not after

The corpus carries the captain's own "chapter/page/paragraph" as structure; filtering on it is
what makes this cheaper than reading a whole conversation, so it runs *before* BM25 ranks
anything, as a SQL `WHERE`, not as a post-hoc trim of the top-k:

| filter | flag | matches |
|---|---|---|
| project ("chapter") | `--project SUBSTR` | project directory name contains `SUBSTR` |
| session ("page") | `--session SUBSTR` | session id (a UUID) contains `SUBSTR` |
| date range | `--since ISO`, `--until ISO` | timestamp bounds, inclusive, either or both |
| speaker | `--role {user,assistant,memory}` | exact match |
| sidechain transcripts | `--no-sidechain` | excludes subagent transcripts (included by default) |

**A rough date-window filter is not a safe default for "have we discussed this before"
queries.** `vdbaccuracy` §3.5/§5.1 measured that for the cross-session recall family
(the gold passage displaced in time, not near the question), date-window filtering
*hurts* — the whole premise of that family is that the answer isn't where you'd guess
to narrow to. Date filtering helps other families (a known, recent window). Reach for
`--since`/`--until` when you have a real reason to believe the answer is in that
window, not as a default "narrow it down" move.

Report §11.3 measured that embedding this same information *into the chunk text* makes retrieval
worse (a −0.041 recall@10 cost for a two-line header alone) — so it is never in the indexed text.
It is stored as columns on the `chunks` table and applied as a metadata filter instead, which is
this module's whole reason filtering is a `WHERE` clause and not a second-pass re-rank.

### Output — deterministic, for a machine reader

`--json` is the contract for a calling agent: a fixed shape, never prose that has to be parsed.

```json
{
  "query_id": 42, "query": "...", "k_requested": 10, "n_hits": 7,
  "margin": 12.4, "weak_signal": false, "nudge_applied": false,
  "hits": [ {"chunk_id": 1, "rank": 1, "score": 0.0, "text": "...",
             "session_id": "...", "project": "...", "role": "...", "ts": "...",
             "part": 0, "n_parts": 1, "source": "...", "is_sidechain": false }, ... ],
  "citation_expected": true,
  "citation_command": "vdb feedback 42 --used <chunk_id>[,<chunk_id>...]",
  "citation_command_used_nothing": "vdb feedback 42 --used \"\"",
  "citation_reminder": "citation is expected after every query you act on: ..."
}
```

Every hit carries project, role, timestamp, session id and source file — enough to cite the
passage or go read the surrounding session deliberately. `query_id` is what you pass to `vdb
feedback` (see "Feedback" below). **Citation is expected after every query you act on** — both
the human-readable and `--json` output say so on every call, every time, with the exact follow-up
command already filled in (`citation_command`/`citation_command_used_nothing` in JSON); this is
not a one-time thing to remember, it is printed again on every `vdb query` call because there is
no way for this tool to enforce it across two separate CLI invocations. Check whether that's
actually happening any time with `vdb stats` (below).

Exit codes are stable and mean the same thing in every mode: **0** = the query ran and found at
least one passage; **1** = the query ran and found nothing; **2** = the query could not run at
all (no index yet, an empty index, or a question with no searchable terms). A calling agent can
branch on the exit code alone without parsing stderr.

### Retrieval is asked for, never injected

The study's load-bearing negative finding is that this system cannot reliably tell when it has
found nothing: the top hit's raw score is statistically near-indistinguishable whether or not the
answer is in the corpus (dense AUC 0.551, F6; re-measured for this system's own BM25 score,
AUC 0.530, 95% CI [0.480, 0.582] — same conclusion). Anything that pastes top-k into every session
would paste confident nonsense a large fraction of the time with no signal that it did — which is
exactly why this is a command an agent calls, never a hook that runs automatically.

`margin` (the score gap between rank 1 and rank 10, or the last hit if fewer) carries more signal
than the raw score — but re-measured for this system specifically, that signal is real and modest
(AUC 0.645, 95% CI [0.598, 0.690]), not the strong 0.724 a prior study found for a dense retriever
on a different score scale. **It is deliberately not thresholded into a verdict.** Instead,
`weak_signal` is a calibration-free structural flag — true when there are fewer than 2 hits (the
margin is undefined) or fewer hits than requested (the index has thin coverage for this query) —
and `false` otherwise. It does *not* mean "trust this"; it only means "there was enough returned
material to have an opinion at all." Reading the actual passages, not just the score or the flag,
is still the caller's job. Do not build a numeric confidence gate on top of `margin` without
re-measuring it (O13 is still open) — the honest answer today is that a good gate does not exist
yet.

## Feedback — how this index learns from real use

Implements `data/vdbfeedback/report.md` (outside this repo). The short version: **cite
what you use, and this system slowly gets better at ranking it; don't, and nothing about
retrieval changes.**

1. Every `vdb query` writes a `query_log` row and prints a `query_id`, whether or not
   you cite anything. This alone changes nothing about ranking.
2. **If you act on a returned passage, cite it — every time, not just when convenient:**
   ```sh
   vdb feedback <query_id> --used <chunk_id>[,<chunk_id>...]
   vdb feedback <query_id> --used ""       # queried, used nothing — also a real signal
   ```
   This is the single highest-leverage thing a calling agent can do here. `vdb query`
   prints the exact command above (with `<query_id>` already filled in) on *every* call,
   human and `--json` alike — a calling agent should not have to already know to cite,
   it is told every time. An uncited query is logged, never scored — it is not diluted
   as partial credit across the results (report §4.2–§4.3: with recall@10 well under
   100% even in the best filtered condition, most of a typical top-k didn't matter even
   on a successful query, so smearing credit across all of it would reward passages that
   merely rode along). There is no inference-based fallback and there will not be one —
   §4.3 already measured that diluted/guessed credit is worse than no signal.

   **Check compliance any time with `vdb stats`** — it reports what fraction of the most
   recent 200 `query_log` rows have a matching `feedback_citation` (`citation_compliance`
   in `--json`). This is the visible signal that catches citation drift instead of
   silently accepting it; a future session should check this before assuming the loop is
   actually accumulating signal.
3. A background pass (`vdb label`, installed as a second systemd timer alongside the
   indexer — see below) reads cited queries old enough for downstream evidence to exist,
   scans your own session's transcript forward for a confirm/correction signal (two
   heuristic tiers: casual chat, and this environment's structured
   supervisor/decision-relay register), and writes `feedback_label` rows. Only clean,
   single-class labels ("mixed" — confirms one thing, corrects another — is recorded but
   never scored) update the per-chunk feedback counts.
4. **The score-affecting nudge (a small, capped additive adjustment to BM25 scores) is
   implemented but ships OFF**, gated by two conditions that must *both* hold:
   - **Global**: the operator flag, **60** high-confidence cited labels corpus-wide
     (`store.NUDGE_LABEL_THRESHOLD` — reduced from the original 300; see that constant's
     docstring for why 60 is the right floor, not just a smaller guess), and a regression
     check against the static eval sets explicitly recorded as passed.
   - **Per-chunk**: a given chunk's own score nudge only applies once *that chunk* has
     earned at least **3** of its own high-confidence, cited labels
     (`store.NUDGE_PER_CHUNK_MIN`) — a passage used constantly no longer has to wait on
     the whole corpus's volume, but it still needs more than one citation before its own
     evidence is trusted (see that constant's docstring for why 3, not 1 or 20).

   Both gates apply independently; neither replaces the other. `vdb nudge --json` (add
   `--chunk <id>` to inspect one chunk's own gate) reports live status;
   `scripts/nudge-regression-check.md` documents the regression check. See `AGENTS.md`
   before ever touching this.

Attribution to "your own session" uses `$CLAUDE_CODE_SESSION_ID` (confirmed to match the
transcript's own filename/`sessionId` — see `AGENTS.md`), captured automatically by
`vdb query` unless you pass `--caller-session` yourself.

## Background indexing — so nobody has to remember to run it

```sh
scripts/install-timer.sh      # install both systemd --user timers, and start them now
scripts/uninstall-timer.sh    # remove them (does not touch the index database)
```

This installs `~/.config/systemd/user/vdb-{index,label}.{service,timer}`, rendered from
`systemd/vdb-{index,label}.service.in` with this checkout's absolute path (it does not
assume the package is `pip install`-ed), then `systemctl --user enable --now`s both
timers. `vdb-index` runs `vdb index --quiet` ~2 minutes after boot/login and every 15
minutes after that; `vdb-label` runs `vdb label --quiet` ~10 minutes after boot/login and
every 30 minutes after that (labelling evidence can be several turns downstream of the
query, report §4.1, so it doesn't need the indexer's tighter cadence). Both use
`Persistent=true`, so a run missed while the machine was off or asleep still happens on
the next wake. An incremental no-op index run costs ~0.2s (see Index, above); a label
pass with nothing newly eligible is comparably cheap.

**For it to run without anyone being logged in** (the actual "just works" requirement on a
headless homeserver), the user needs systemd lingering enabled once:
`sudo loginctl enable-linger $(whoami)`. The install script checks this and prints the command if
it's missing, rather than silently leaving the timer only running while someone's logged in.

This is a `systemd --user` timer, not a bespoke daemon — no framework, no supervisor process of
its own, nothing that needs babysitting beyond what `systemctl --user status vdb-index.timer` and
`journalctl --user -u vdb-index.service` already give you.

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
| `vdb/retrieve.py` | BM25 query path, metadata filters, `Hit`/`Result` shapes, the margin/weak-signal diagnostic, the (gated-off) score nudge |
| `vdb/feedback.py` | the background label-extraction pass (`vdb label`): heuristic confirm/correction classification, boilerplate-aware transcript scanning |
| `vdb/cli.py` | `index` / `query` / `feedback` / `label` / `nudge` / `stats` / `boilerplate`; deterministic output and exit codes for an agent caller |
| `systemd/`, `scripts/install-timer.sh`, `scripts/uninstall-timer.sh` | the background indexer + label pass (two systemd `--user` timers) |
| `scripts/nudge-regression-check.md` | the §6.2c procedure to run before ever flipping the score nudge on |

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
