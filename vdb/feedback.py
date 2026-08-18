"""Background label extraction (data/vdbfeedback/report.md §5.2, §8 step 3).

Heuristics only, deliberately - not a model call. The report's own measurement
says the failure modes here are boilerplate contamination and low pair volume,
not language understanding (§5.2). Two tiers, both measured in the report:

* casual-chat patterns (§2.4 pass 1) - short affirmatives/negatives the brief
  itself gives as examples ("yes", "continue", "that's wrong", "try again").
* structured-decision-register patterns (§3.5, §2.4 pass 2) - this
  environment's actual dominant register: multi-agent supervisor/decision-relay
  traffic ("Decision [key=...]: approved", "--action fix", "not an acceptable
  answer"), more attributable than casual chat because it names the thing
  being judged.

A message that fires both tiers' opposite classes in the same turn is
"mixed" - confirms one thing, corrects another - and is recorded but never
promoted to a score-affecting label (§6.1's safeguard).

Neither pattern set is the report's own `classify.py`/`classify2.py` lab
scripts verbatim (those measured a base rate over a lab-scratch corpus and are
explicitly not proposed as production code, report Appendix A item 7); this is
a from-scratch implementation of the same two registers, tuned for
maintainability rather than reproducing a one-off measurement.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import ingest as ingest_mod
from . import store as store_mod
from .clean import trim_boilerplate

# How long to wait before trusting "no evidence found yet" as opposed to
# "the evidence just hasn't arrived" - report §4.1 notes the confirming or
# correcting turn is often several turns downstream, not immediately next.
DEFAULT_MIN_AGE_SECONDS = 900

# How many of the caller session's own user-role turns after the query to
# scan forward through looking for the first clean signal (report §4.1: not
# "the very next message").
DEFAULT_MAX_LOOKAHEAD = 20

LABEL_METHOD_CASUAL = "heuristic_casual_v1"
LABEL_METHOD_STRUCT = "heuristic_struct_v1"

_CASUAL_CONFIRM_START = re.compile(
    r"^(yes|yep|yeah|yup|correct|right|good|great|perfect|exactly|nice|"
    r"continue|lgtm|approved?|confirmed?)\b",
    re.IGNORECASE,
)
_CASUAL_CONFIRM_MID = re.compile(
    r"\b(sounds good|looks good|that works|that'?s right|that'?s correct|"
    r"makes sense|lgtm)\b",
    re.IGNORECASE,
)
_CASUAL_CORRECTION_START = re.compile(
    r"^(no|nope|wrong|incorrect)\b",
    re.IGNORECASE,
)
_CASUAL_CORRECTION_MID = re.compile(
    r"\b(that'?s wrong|that'?s not right|that'?s incorrect|not quite|"
    r"try again|this needs to be changed|not what i (asked|meant|wanted))\b",
    re.IGNORECASE,
)

# Report §3.5: a decision keyed to a specific finding, machine-parseable by
# construction because it already has to be for the supervision system to
# route it.
_STRUCT_CONFIRM = re.compile(
    r"decision\s*\[key=[^\]]+\]\s*:\s*approved\b"
    r"|--action[= ]approve\b",
    re.IGNORECASE,
)
_STRUCT_CORRECTION = re.compile(
    r"decision\s*\[key=[^\]]+\]\s*:\s*(reject|rejected|denied)\b"
    r"|--action[= ]fix\b"
    r"|not an acceptable answer\b",
    re.IGNORECASE,
)

LABEL_NONE = "none"
LABEL_CONFIRM = "confirm"
LABEL_CORRECTION = "correction"
LABEL_MIXED = "mixed"


def classify_text(text: str) -> tuple[str, str]:
    """Classify one (already boilerplate-trimmed) message.

    Returns `(label, label_method)`. `label_method` names every tier that
    fired, so a later audit can tell which pattern set produced a given row.
    """
    text = (text or "").strip()
    if not text:
        return LABEL_NONE, "none"

    confirm_hit = False
    correction_hit = False
    methods: list[str] = []

    if _CASUAL_CONFIRM_START.match(text) or _CASUAL_CONFIRM_MID.search(text):
        confirm_hit = True
        methods.append(LABEL_METHOD_CASUAL)
    if _CASUAL_CORRECTION_START.match(text) or _CASUAL_CORRECTION_MID.search(text):
        correction_hit = True
        methods.append(LABEL_METHOD_CASUAL)
    if _STRUCT_CONFIRM.search(text):
        confirm_hit = True
        methods.append(LABEL_METHOD_STRUCT)
    if _STRUCT_CORRECTION.search(text):
        correction_hit = True
        methods.append(LABEL_METHOD_STRUCT)

    method = "+".join(sorted(set(methods))) if methods else "none"
    if confirm_hit and correction_hit:
        return LABEL_MIXED, method
    if confirm_hit:
        return LABEL_CONFIRM, method
    if correction_hit:
        return LABEL_CORRECTION, method
    return LABEL_NONE, "none"


@dataclass
class LabelPassStats:
    scanned: int = 0
    labeled: int = 0
    skipped_too_young: int = 0
    skipped_no_session: int = 0
    skipped_no_evidence: int = 0
    by_label: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _find_session_transcript(root: Path, session_id: str) -> Path | None:
    """Locate the caller's own transcript file by session id (report §4.4).

    Session ids are UUIDs used as the transcript's own filename stem
    (verified: `$CLAUDE_CODE_SESSION_ID` matches the `.jsonl` filename under
    `~/.claude/projects/<project>/<session_id>.jsonl` - see AGENTS.md). This
    is an exact lookup, not a guess - deliberately NOT "most recently
    modified file in the project directory", which report Appendix A item 4
    rejects: this corpus routinely has multiple concurrent sessions active
    in the same project directory.
    """
    if not root.exists():
        return None
    matches = list(root.rglob(f"{session_id}.jsonl"))
    return matches[0] if matches else None


def _scan_forward(
    path: Path, since_ts: str, boilerplate: set[str], max_lookahead: int
) -> tuple[str | None, str | None, int]:
    """Walk the caller's own transcript forward from `since_ts` for a label.

    Only main-thread (`is_sidechain=False`) user-role messages are considered
    - the direct evidence a human or supervising agent actually said
    something back, same population the report measures (§2.3). Boilerplate
    lines are trimmed with the same frequency-based set the index itself
    uses (`store.boilerplate_lines`) before classifying, reusing
    `store.py`'s dedup mechanism rather than reinventing it (§5.2) - a pure
    harness-template turn (§3.2) is treated as though it were not there.

    Returns `(label, label_method, turns_downstream)`, or `(None, None, 0)`
    if nothing qualifies within `max_lookahead` candidate turns.
    """
    st = path.stat()
    src = ingest_mod.SourceFile(
        path=path, kind=ingest_mod.TRANSCRIPT, size=st.st_size, mtime_ns=st.st_mtime_ns
    )
    messages, _ = ingest_mod.read_messages(src)

    seen = 0
    for msg in messages:
        if msg.is_sidechain or msg.role != "user":
            continue
        if not msg.ts or msg.ts < since_ts:
            continue
        seen += 1
        trimmed = trim_boilerplate(msg.text, boilerplate)
        if not trimmed.strip():
            continue
        label, method = classify_text(trimmed)
        if label != LABEL_NONE:
            return label, method, seen
        if seen >= max_lookahead:
            break
    return None, None, 0


def run_label_pass(
    conn: sqlite3.Connection,
    root: Path = ingest_mod.DEFAULT_ROOT,
    min_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    max_lookahead: int = DEFAULT_MAX_LOOKAHEAD,
    now: datetime | None = None,
) -> LabelPassStats:
    """Extract labels for cited queries old enough for evidence to exist.

    Only queries with a *non-empty* citation are considered: an uncited query
    has no chunk to attribute a label to, so scanning for one would be
    wasted work in service of a label that could never be score-affecting
    anyway (report §4.3). Queries already labelled are skipped (idempotent
    across repeated timer runs).
    """
    now = now or datetime.now(timezone.utc)
    stats = LabelPassStats()
    boilerplate = store_mod.boilerplate_lines(conn)

    rows = conn.execute(
        "SELECT ql.id AS query_id, ql.ts, ql.caller_session, ql.held_out,"
        "       fc.used_chunk_ids "
        "FROM query_log ql "
        "JOIN feedback_citation fc ON fc.query_id = ql.id "
        "WHERE fc.used_chunk_ids != '[]' "
        "AND NOT EXISTS (SELECT 1 FROM feedback_label fl WHERE fl.query_id = ql.id)"
    ).fetchall()

    for row in rows:
        stats.scanned += 1
        try:
            q_ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
        except ValueError:
            stats.skipped_no_evidence += 1
            continue
        if q_ts.tzinfo is None:
            q_ts = q_ts.replace(tzinfo=timezone.utc)
        if now - q_ts < timedelta(seconds=min_age_seconds):
            stats.skipped_too_young += 1
            continue

        session = row["caller_session"]
        transcript = _find_session_transcript(root, session) if session else None
        if transcript is None:
            stats.skipped_no_session += 1
            continue

        label, method, turns_downstream = _scan_forward(
            transcript, row["ts"], boilerplate, max_lookahead
        )
        if label is None:
            stats.skipped_no_evidence += 1
            continue

        store_mod.write_label(conn, row["query_id"], label, method, turns_downstream)
        stats.labeled += 1
        stats.by_label[label] = stats.by_label.get(label, 0) + 1

        # Only clean, single-class labels, backed by a citation, on a
        # non-held-out query move the score table (§6.1, §6.2b compounded).
        if label in (LABEL_CONFIRM, LABEL_CORRECTION) and not row["held_out"]:
            used_ids = json.loads(row["used_chunk_ids"])
            store_mod.apply_feedback_to_chunks(conn, used_ids, label)

    conn.commit()
    return stats
