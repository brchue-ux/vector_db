"""Command line: `vdb index`, `vdb dense-index`, `vdb query` (`--hybrid` for phase 2's
BM25+dense fusion), `vdb feedback`, `vdb label`, `vdb nudge`, `vdb stats`, `vdb boilerplate`.

The primary caller is an agent, not a human at a terminal - so `query --json`
output is deterministic and machine-parseable, and exit codes are stable:
0 = hits returned, 1 = the query ran but found nothing, 2 = the query could
not run at all (no index, an empty index, or an unparseable question). A
human reading the plain-text output gets the identical information with
headers and diagnostics attached.

Delivery is an explicit command, by design. Report §13.2 makes this the
load-bearing negative finding: retrieval cannot tell when it has found nothing
(F6; re-measured for this retriever's own score in `vdb/retrieve.py`'s
docstring), so it must be asked, never injected silently into a session.

Every `query` also prints a calibrated `confidence` label (`confident` /
`uncertain` / `low_confidence`, `retrieve.confidence_band()`) alongside the
results. It is a label on the existing ranking, not a change to it - a
measured but noisy signal (see that function's docstring for the calibration
and its own honest limits), never a substitute for reading the passages.

Every `query` prints a `query_id` AND the exact citation follow-up command,
every single time, in both human and `--json` output - this cannot be
enforced across two separate CLI invocations, so the tool insists loudly
instead of hoping the caller remembers. If you act on a returned passage,
cite it with `vdb feedback <query_id> --used <chunk_id>[,...]` (an empty
`--used` is a valid "queried, used nothing" signal) - this is the only thing
that lets `vdb label`'s background pass (data/vdbfeedback/report.md §8 step
3) ever attribute a confirm/correction to a specific passage. Uncited queries
are still logged, never scored. Check whether citation is actually happening
with `vdb stats` (`citation_compliance`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from . import feedback as feedback_mod
from . import ingest as ingest_mod
from . import retrieve, store


def _fmt_int(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def cmd_index(args) -> int:
    conn = store.connect(args.db)
    seen = [0]

    def progress(rel: str, n_msgs: int) -> None:
        seen[0] += 1
        if not args.quiet:
            print(f"  [{seen[0]:>5}] {rel} ({n_msgs} messages)", file=sys.stderr)

    st = store.index(conn, root=Path(args.root), rebuild=args.rebuild, progress=progress)
    if args.json:
        print(json.dumps({"index": st.as_dict(), "store": store.stats(conn)}, indent=2))
        return 0
    print(
        f"scanned {_fmt_int(st.scanned)} files · indexed {_fmt_int(st.indexed)} · "
        f"unchanged {_fmt_int(st.skipped)} · removed {_fmt_int(st.removed)}"
    )
    print(
        f"{_fmt_int(st.messages)} messages -> {_fmt_int(st.chunks)} chunks, "
        f"{_fmt_int(st.chars)} chars indexed, {_fmt_int(st.trimmed_lines)} boilerplate lines trimmed"
    )
    if st.secrets:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(st.secrets.items()))
        print(f"secret filter redacted: {detail}")
    else:
        print("secret filter redacted: nothing in the files touched this run")
    s = store.stats(conn)
    print(
        f"index now: {_fmt_int(s['chunks'])} chunks over {_fmt_int(s['sessions'])} sessions "
        f"in {_fmt_int(s['projects'])} projects (mean {s['mean_chunk_chars']} chars/chunk)"
    )
    print(f"database: {args.db}  — as sensitive as the transcripts; never commit it")
    return 0


def cmd_dense_index(args) -> int:
    """Build/update the dense (`multilingual-e5-large`) index (`vdb/dense.py`).

    A separate, much more expensive command from `vdb index` on purpose - this needs
    `fastembed`/`onnxruntime` (the `dense` extra, `pip install -e '.[dense]'`) and embeds
    every not-yet-embedded chunk, which is a many-hour job on the full corpus (README "Dense
    index" has the real command and how to run it detached in the background). `--status`
    reports coverage without embedding anything, so a background run can be checked on
    cheaply.
    """
    from . import dense as dense_mod

    conn = store.connect(args.db)
    if args.status:
        cov = dense_mod.coverage(conn)
        if args.json:
            print(json.dumps(cov, indent=2))
        else:
            print(
                f"model: {cov['model']}\n"
                f"embedded {_fmt_int(cov['embedded'])} / {_fmt_int(cov['chunks'])} chunks "
                f"({_fmt_int(cov['remaining'])} remaining)\n"
                f"complete: {cov['complete']}"
            )
        return 0

    def progress(embedded: int, total: int) -> None:
        if not args.quiet:
            print(f"  [{embedded:>6}/{total}] embedded", file=sys.stderr)

    try:
        st = dense_mod.build_index(
            conn, rebuild=args.rebuild, batch_size=args.batch_size, progress=progress
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"dense_index": st.as_dict(), "coverage": dense_mod.coverage(conn)}, indent=2))
        return 0
    print(f"embedded {_fmt_int(st.embedded)} / {_fmt_int(st.candidates)} candidate chunks")
    cov = dense_mod.coverage(conn)
    print(f"dense index now: {_fmt_int(cov['embedded'])} / {_fmt_int(cov['chunks'])} chunks embedded")
    return 0


def cmd_query(args) -> int:
    db = Path(args.db)
    if not db.exists():
        print(
            f"no index at {db}\nrun:  python -m vdb index",
            file=sys.stderr,
        )
        return 2
    conn = store.connect(db)
    if store.stats(conn)["chunks"] == 0:
        print("the index is empty — run `python -m vdb index` first", file=sys.stderr)
        return 2

    if args.hybrid:
        from . import dense as dense_mod

        if dense_mod.coverage(conn)["embedded"] == 0:
            print(
                "warning: the dense index is empty — `--hybrid` results are effectively "
                "BM25-only until `vdb dense-index` has run (see README 'Dense index')",
                file=sys.stderr,
            )
        retriever = retrieve.HybridRetriever(conn)
    else:
        retriever = retrieve.BM25Retriever(conn)

    try:
        result = retriever.search(
            args.question,
            k=args.k,
            project=args.project,
            role=args.role,
            session=args.session,
            since=args.since,
            until=args.until,
            include_sidechain=not args.no_sidechain,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    caller_session = args.caller_session or os.environ.get("CLAUDE_CODE_SESSION_ID")
    query_id = store.log_query(
        conn,
        query_text=result.query,
        filters={
            "project": args.project,
            "role": args.role,
            "session": args.session,
            "since": args.since,
            "until": args.until,
            "include_sidechain": not args.no_sidechain,
            "retriever": result.retriever,
        },
        k_requested=result.k_requested,
        hit_chunk_ids=[h.chunk_id for h in result.hits],
        hit_scores=[h.score for h in result.hits],
        margin=result.margin,
        weak_signal=result.weak_signal,
        caller_session=caller_session,
    )

    citation_cmd = f'vdb feedback {query_id} --used <chunk_id>[,<chunk_id>...]'
    citation_cmd_used_nothing = f'vdb feedback {query_id} --used ""'
    citation_reminder = (
        "citation is expected after every query you act on: run "
        f"`{citation_cmd}` if you used a result, or `{citation_cmd_used_nothing}` "
        "if you queried and used nothing — this is what lets the index learn "
        "from real use (see README 'Feedback'; check compliance any time with "
        "`vdb stats`)"
    )

    if args.json:
        print(
            json.dumps(
                {
                    "query_id": query_id,
                    "query": result.query,
                    "retriever": result.retriever,
                    "k_requested": result.k_requested,
                    "n_hits": len(result.hits),
                    "margin": result.margin,
                    "weak_signal": result.weak_signal,
                    "confidence": result.confidence,
                    "confidence_note": retrieve.CONFIDENCE_EXPLANATION[result.confidence],
                    "nudge_applied": result.nudge_applied,
                    "hits": [h.__dict__ for h in result.hits],
                    "citation_expected": True,
                    "citation_command": citation_cmd,
                    "citation_command_used_nothing": citation_cmd_used_nothing,
                    "citation_reminder": citation_reminder,
                },
                indent=2,
            )
        )
        return 0 if result.hits else 1

    if not result.hits:
        print("no passages matched — this may mean the index has nothing on this, "
              "not that nothing on this exists (this system cannot tell the two apart).")
        if result.retriever != "bm25":
            print(f"retriever: {result.retriever}")
        print(f"\nconfidence: {result.confidence}  — {retrieve.CONFIDENCE_EXPLANATION[result.confidence]}")
        print(f"\nquery_id: {query_id}  — {citation_reminder}")
        return 1
    for hit in result.hits:
        body = hit.text if args.full else textwrap.shorten(
            " ".join(hit.text.split()), width=args.width, placeholder=" …"
        )
        print(f"\n#{hit.rank}  score {hit.score:.3f}  [{hit.provenance()}]")
        print(f"    session {hit.session_id}")
        print(f"    chunk_id {hit.chunk_id}")
        print(f"    {hit.source}")
        print(textwrap.indent(body, "    "))
    if result.margin is not None:
        print(f"\nrank1–rank{min(10, len(result.hits))} margin: {result.margin}")
    if result.retriever != "bm25":
        print(f"retriever: {result.retriever}")
    print(f"confidence: {result.confidence}  — {retrieve.CONFIDENCE_EXPLANATION[result.confidence]}")
    if result.weak_signal:
        print(
            "weak signal: fewer results than usual for this index — treat these "
            "passages with more caution than a full, well-separated result set "
            "(this is a structural flag, independent of confidence above; see README)"
        )
    print(
        "(confidence is a calibrated but noisy signal, not a verdict — read the "
        "passages, don't just trust the label; see README 'Confidence')"
    )
    print(f"\nquery_id: {query_id}  — {citation_reminder}")
    return 0


def cmd_feedback(args) -> int:
    """`vdb feedback <query_id> --used <chunk_id>[,...]` (report §4.2, §8 step 2).

    Citing what you used is what makes every later label mean something: an
    uncited query is logged but never enters the score-affecting path (§4.3).
    """
    conn = store.connect(args.db)
    used_ids = [int(x) for x in args.used.split(",") if x.strip()]
    try:
        store.record_citation(conn, args.query_id, used_ids)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"query_id": args.query_id, "used_chunk_ids": used_ids}))
    elif used_ids:
        print(f"recorded: query {args.query_id} used chunks {used_ids}")
    else:
        print(f"recorded: query {args.query_id} queried, used nothing")
    return 0


def cmd_label(args) -> int:
    """Background pass (`vdb label`, report §8 step 3) - install via the systemd
    --user timer (`scripts/install-timer.sh`), not run by hand as a daemon."""
    conn = store.connect(args.db)
    st = feedback_mod.run_label_pass(
        conn,
        root=Path(args.root),
        min_age_seconds=args.min_age_seconds,
        max_lookahead=args.max_lookahead,
    )
    if args.json:
        print(json.dumps(st.as_dict(), indent=2))
    elif not args.quiet:
        print(
            f"scanned {_fmt_int(st.scanned)} cited queries · labelled {_fmt_int(st.labeled)} "
            f"({st.by_label}) · too young {_fmt_int(st.skipped_too_young)} · "
            f"no session found {_fmt_int(st.skipped_no_session)} · "
            f"no evidence yet {_fmt_int(st.skipped_no_evidence)}"
        )
    return 0


def cmd_nudge(args) -> int:
    """Status and control for the score-affecting nudge's runtime gate (report §6.3).

    Two gates, both required (see `vdb/store.py`'s `NUDGE_LABEL_THRESHOLD` and
    `NUDGE_PER_CHUNK_MIN` docstrings for why each number is what it is):
    - GLOBAL: the operator flag, `store.NUDGE_LABEL_THRESHOLD` corpus-wide
      qualifying labels, and a recorded passing regression check
      (`--record-check`) - `store.nudge_active()`. Setting `--enable` only
      flips the operator flag; the other two conditions still apply. See
      `scripts/nudge-regression-check.md` before ever recording a pass.
    - PER-CHUNK: a specific chunk's own accumulated evidence must separately
      clear `store.NUDGE_PER_CHUNK_MIN` - pass `--chunk <id>` to inspect one.
    """
    conn = store.connect(args.db)
    if args.enable:
        store.set_nudge_flag(conn, True)
    elif args.disable:
        store.set_nudge_flag(conn, False)
    elif args.record_check:
        store.record_regression_check(
            conn, passed=(args.record_check == "pass"), notes=args.notes or ""
        )

    status = {
        "flag_enabled": store.nudge_flag_enabled(conn),
        "qualifying_labels": store.qualifying_label_count(conn),
        "global_threshold": store.NUDGE_LABEL_THRESHOLD,
        "regression_check": store.regression_check_status(conn),
        "active": store.nudge_active(conn),
        "per_chunk_threshold": store.NUDGE_PER_CHUNK_MIN,
    }
    if args.chunk is not None:
        status["chunk_id"] = args.chunk
        status["chunk_label_count"] = store.chunk_label_count(conn, args.chunk)
        status["chunk_qualifies"] = status["chunk_label_count"] >= store.NUDGE_PER_CHUNK_MIN
        status["chunk_nudged"] = status["active"] and status["chunk_qualifies"]

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"flag enabled:          {status['flag_enabled']}")
        print(f"qualifying labels:     {status['qualifying_labels']} / {status['global_threshold']}"
              " (global gate)")
        print(f"regression check:      {status['regression_check']}")
        print(f"NUDGE ACTIVE (global): {status['active']}")
        print(f"per-chunk threshold:   {status['per_chunk_threshold']} labels")
        if args.chunk is not None:
            print(f"chunk {status['chunk_id']} labels:   {status['chunk_label_count']}")
            print(f"chunk {status['chunk_id']} nudged:    {status['chunk_nudged']}")
    return 0


def cmd_stats(args) -> int:
    conn = store.connect(args.db)
    s = store.stats(conn)
    if args.json:
        print(json.dumps(s, indent=2))
        return 0
    for key, value in s.items():
        if key == "citation_compliance":
            continue
        print(f"{key:>20}: {_fmt_int(value)}")
    cc = s["citation_compliance"]
    print(
        f"{'citation compliance':>20}: {cc['cited']}/{cc['queries']} of the last "
        f"{cc['window']} queries cited"
        + (f" ({cc['citation_rate']:.0%})" if cc["citation_rate"] is not None else "")
    )
    if cc["queries"] and cc["citation_rate"] is not None and cc["citation_rate"] < 1.0:
        print(
            "                      every uncited query is logged but never enters the "
            "score-affecting path — see README 'Feedback'"
        )
    return 0


def cmd_boilerplate(args) -> int:
    """Show the invariant template lines currently being trimmed (§7)."""
    conn = store.connect(args.db)
    rows = conn.execute(
        "SELECT line, n FROM line_freq WHERE n >= ? ORDER BY n DESC LIMIT ?",
        (store.BOILERPLATE_MIN_MESSAGES, args.limit),
    ).fetchall()
    if not rows:
        print("no lines meet the boilerplate threshold yet")
        return 0
    for row in rows:
        print(f"{row['n']:>6}  {row['line'][:140]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vdb",
        description="Local BM25 + dense retrieval over your own Claude Code history.",
    )
    p.add_argument("--db", default=str(store.DEFAULT_DB), help="index database path")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index", help="build or incrementally update the BM25 index")
    i.add_argument("--root", default=str(ingest_mod.DEFAULT_ROOT), help="corpus root (read-only)")
    i.add_argument("--rebuild", action="store_true", help="discard the index and start over")
    i.add_argument("--json", action="store_true")
    i.add_argument("-q", "--quiet", action="store_true", help="no per-file progress")
    i.set_defaults(func=cmd_index)

    di = sub.add_parser(
        "dense-index",
        help="build or incrementally update the dense (multilingual-e5-large) index — "
        "needs the 'dense' extra (pip install -e '.[dense]'); a many-hour job on the full "
        "corpus, see README 'Dense index'",
    )
    di.add_argument("--rebuild", action="store_true", help="discard dense embeddings and start over")
    di.add_argument(
        "--batch-size", type=int, default=32, help="chunks embedded per model call (default 32)"
    )
    di.add_argument(
        "--status", action="store_true", help="report embedding coverage and exit, embed nothing"
    )
    di.add_argument("--json", action="store_true")
    di.add_argument("-q", "--quiet", action="store_true", help="no per-batch progress")
    di.set_defaults(func=cmd_dense_index)

    q = sub.add_parser(
        "query",
        help="ask the index a question",
        epilog=(
            "Filter guidance: narrowing by project is close to a universal win. A rough or "
            "guessed date window helps when a query is naturally time-scoped, but can hurt "
            "cross-session 'have we discussed this before' questions because their answers "
            "are displaced in time. If you are not sure which kind of question you are "
            "answering, do not use --since/--until by default."
        ),
    )
    q.add_argument("question")
    q.add_argument("-k", type=int, default=10, help="number of passages (default 10)")
    q.add_argument("--project", help="restrict to projects whose directory name contains this")
    q.add_argument("--role", choices=["user", "assistant", "memory"], help="the speaker")
    q.add_argument("--session", help="restrict to session ids containing this (a session is one transcript)")
    q.add_argument("--since", help="ISO timestamp lower bound, e.g. 2026-01-01")
    q.add_argument("--until", help="ISO timestamp upper bound, e.g. 2026-06-30")
    q.add_argument("--no-sidechain", action="store_true", help="exclude subagent transcripts")
    q.add_argument(
        "--hybrid",
        action="store_true",
        help="fuse BM25 with the dense (multilingual-e5-large) retriever via reciprocal-rank "
        "fusion, instead of BM25 alone — needs `vdb dense-index` to have run; confidence is "
        "reported as 'uncalibrated' for hybrid results, not the BM25-calibrated bands (see "
        "README 'Confidence')",
    )
    q.add_argument("--full", action="store_true", help="print whole passages, not excerpts")
    q.add_argument("--width", type=int, default=400, help="excerpt width (default 400)")
    q.add_argument(
        "--caller-session",
        help="your own Claude Code session id, for feedback attribution (report §4.4); "
        "defaults to $CLAUDE_CODE_SESSION_ID when unset — usually no need to pass this",
    )
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_query)

    f = sub.add_parser(
        "feedback",
        help="report which returned chunk(s) you used — cite this every time you act on a result",
    )
    f.add_argument("query_id", type=int)
    f.add_argument(
        "--used",
        required=True,
        help='comma-separated chunk_ids you used, or "" if you queried and used none',
    )
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_feedback)

    l = sub.add_parser(
        "label",
        help="background pass: extract confirm/correction labels from downstream conversation",
    )
    l.add_argument("--root", default=str(ingest_mod.DEFAULT_ROOT), help="corpus root (read-only)")
    l.add_argument(
        "--min-age-seconds", type=int, default=feedback_mod.DEFAULT_MIN_AGE_SECONDS,
        help="how long to wait for downstream evidence before giving up",
    )
    l.add_argument(
        "--max-lookahead", type=int, default=feedback_mod.DEFAULT_MAX_LOOKAHEAD,
        help="how many of the caller's own user-role turns to scan forward",
    )
    l.add_argument("--json", action="store_true")
    l.add_argument("-q", "--quiet", action="store_true")
    l.set_defaults(func=cmd_label)

    n = sub.add_parser("nudge", help="status/control for the score-affecting feedback nudge (default: inert)")
    n.add_argument("--enable", action="store_true", help="set the operator flag (does not by itself activate the nudge)")
    n.add_argument("--disable", action="store_true", help="clear the operator flag")
    n.add_argument(
        "--record-check", choices=["pass", "fail"], default=None,
        help="record a §6.2c regression-check outcome — see scripts/nudge-regression-check.md; never fabricate 'pass'",
    )
    n.add_argument("--notes", help="notes to attach to --record-check")
    n.add_argument(
        "--chunk", type=int, default=None,
        help="also report this chunk_id's own per-chunk gate status (evidence count, whether it would be nudged)",
    )
    n.add_argument("--json", action="store_true")
    n.set_defaults(func=cmd_nudge)

    s = sub.add_parser("stats", help="what is in the index, plus citation compliance (recent `vdb feedback` follow-through)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    b = sub.add_parser("boilerplate", help="template lines currently trimmed on ingest")
    b.add_argument("--limit", type=int, default=50)
    b.set_defaults(func=cmd_boilerplate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
