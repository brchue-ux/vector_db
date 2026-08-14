"""Read-only ingest of the Claude Code history corpus.

Two sources, per report §13.1 and §B.6:

* transcripts - `~/.claude/projects/**/*.jsonl`
* memory cards - `~/.claude/projects/*/memory/*.md`

Prose only. `tool_use` inputs, `tool_result` blocks, `toolUseResult`,
`attachment` entries and `type=="system"` entries are tool I/O and stay out of
this index (§6.3-§6.4: the cost is the 22.6:1 *size* ratio, not the content, so
if tool I/O is ever wanted it belongs in a separate index queried deliberately).
`thinking` blocks are skipped because their text is empty on disk (§1.2).

Nothing here ever opens a file for writing. The corpus is treated as read-only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import secrets as secrets_mod
from .clean import strip_injected

DEFAULT_ROOT = Path.home() / ".claude" / "projects"

TRANSCRIPT = "transcript"
MEMORY = "memory"


@dataclass(frozen=True)
class Message:
    """One cleaned prose message, before chunking."""

    session_id: str
    project: str
    role: str
    ts: str | None
    uuid: str | None
    seq: int
    text: str
    is_sidechain: bool = False


@dataclass(frozen=True)
class SourceFile:
    path: Path
    kind: str
    size: int
    mtime_ns: int

    @property
    def project(self) -> str:
        if self.kind == MEMORY:
            return self.path.parent.parent.name
        return self.path.parent.name


def scan(root: Path = DEFAULT_ROOT) -> list[SourceFile]:
    """Every ingestable file under `root`, in a stable order."""
    found: list[SourceFile] = []
    if not root.exists():
        return found
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            kind = TRANSCRIPT
        elif path.suffix == ".md" and path.parent.name == "memory":
            kind = MEMORY
        else:
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        found.append(SourceFile(path, kind, st.st_size, st.st_mtime_ns))
    return found


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _blocks_text(content: object) -> str:
    """Prose text out of a message's `content`, tool I/O excluded."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "\n\n".join(p for p in parts if p)


def read_transcript(src: SourceFile) -> Iterator[Message]:
    seq = 0
    fallback_session = src.path.stem
    with src.path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get("type") not in ("user", "assistant"):
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role") or entry.get("type")
            raw = _blocks_text(message.get("content"))
            if not raw:
                continue
            text = strip_injected(raw)
            if not text:
                # The whole turn was harness machinery (§1.1: 3,075 such user
                # messages in the measured corpus).
                continue
            yield Message(
                session_id=entry.get("sessionId") or fallback_session,
                project=src.project,
                role=role,
                ts=entry.get("timestamp"),
                uuid=entry.get("uuid"),
                seq=seq,
                text=text,
                is_sidechain=bool(entry.get("isSidechain")),
            )
            seq += 1


def read_memory(src: SourceFile) -> Iterator[Message]:
    """A memory card is one message with its own synthetic session id (§B.3)."""
    try:
        raw = src.path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    text = strip_injected(raw)
    if not text:
        return
    ts = datetime.fromtimestamp(src.mtime_ns / 1e9, tz=timezone.utc).isoformat()
    yield Message(
        session_id=f"memory:{src.project}/{src.path.name}",
        project=src.project,
        role="memory",
        ts=ts,
        uuid=None,
        seq=0,
        text=text,
        is_sidechain=False,
    )


def read_messages(src: SourceFile, redact: bool = True) -> tuple[list[Message], dict[str, int]]:
    """Cleaned, secret-filtered messages for one source file.

    Returns the messages and a tally of what the secret filter removed.
    """
    reader = read_memory if src.kind == MEMORY else read_transcript
    out: list[Message] = []
    tally: dict[str, int] = {}
    for msg in reader(src):
        text = msg.text
        if redact:
            text, counts = secrets_mod.redact(text)
            for k, v in counts.items():
                tally[k] = tally.get(k, 0) + v
            if not text.strip():
                continue
        out.append(Message(**{**msg.__dict__, "text": text}))
    return out, tally


def relpath(path: Path, root: Path) -> str:
    try:
        return os.fspath(path.relative_to(root))
    except ValueError:
        return os.fspath(path)
