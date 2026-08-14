"""Command line: `vdb index`, `vdb query`, `vdb stats`, `vdb boilerplate`.

Delivery is an explicit command, by design. Report §13.2 makes this the
load-bearing negative finding: retrieval cannot tell when it has found nothing
(F6, AUC 0.551), so it must be asked, never injected silently into a session.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

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
    try:
        result = retrieve.BM25Retriever(conn).search(
            args.question,
            k=args.k,
            project=args.project,
            role=args.role,
            since=args.since,
            include_sidechain=not args.no_sidechain,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "query": result.query,
                    "margin": result.margin,
                    "hits": [h.__dict__ for h in result.hits],
                },
                indent=2,
            )
        )
        return 0

    if not result.hits:
        print("no passages matched.")
        return 1
    for hit in result.hits:
        body = hit.text if args.full else textwrap.shorten(
            " ".join(hit.text.split()), width=args.width, placeholder=" …"
        )
        print(f"\n#{hit.rank}  score {hit.score:.3f}  [{hit.provenance()}]")
        print(f"    session {hit.session_id}")
        print(f"    {hit.source}")
        print(textwrap.indent(body, "    "))
    if result.margin is not None:
        print(
            f"\nrank1–rank{min(10, len(result.hits))} margin: {result.margin} "
            "(a signal, not a verdict — this system cannot tell when it has nothing)"
        )
    return 0


def cmd_stats(args) -> int:
    conn = store.connect(args.db)
    s = store.stats(conn)
    if args.json:
        print(json.dumps(s, indent=2))
        return 0
    for key, value in s.items():
        print(f"{key:>20}: {_fmt_int(value)}")
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
        description="Local BM25 retrieval over your own Claude Code history (phase 1).",
    )
    p.add_argument("--db", default=str(store.DEFAULT_DB), help="index database path")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index", help="build or incrementally update the index")
    i.add_argument("--root", default=str(ingest_mod.DEFAULT_ROOT), help="corpus root (read-only)")
    i.add_argument("--rebuild", action="store_true", help="discard the index and start over")
    i.add_argument("--json", action="store_true")
    i.add_argument("-q", "--quiet", action="store_true", help="no per-file progress")
    i.set_defaults(func=cmd_index)

    q = sub.add_parser("query", help="ask the index a question")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=10, help="number of passages (default 10)")
    q.add_argument("--project", help="restrict to projects whose directory name contains this")
    q.add_argument("--role", choices=["user", "assistant", "memory"])
    q.add_argument("--since", help="ISO timestamp lower bound, e.g. 2026-01-01")
    q.add_argument("--no-sidechain", action="store_true", help="exclude subagent transcripts")
    q.add_argument("--full", action="store_true", help="print whole passages, not excerpts")
    q.add_argument("--width", type=int, default=400, help="excerpt width (default 400)")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_query)

    s = sub.add_parser("stats", help="what is in the index")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    b = sub.add_parser("boilerplate", help="template lines currently trimmed on ingest")
    b.add_argument("--limit", type=int, default=50)
    b.set_defaults(func=cmd_boilerplate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
