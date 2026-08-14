"""Builders for a synthetic corpus.

Everything the tests read is generated here. Real transcript excerpts are never
committed to this repo - the corpus is private and the index derived from it is
exactly as sensitive (report F9).
"""

from __future__ import annotations

import json
from pathlib import Path


def user_entry(text, session="s1", uuid="u1", ts="2026-08-01T10:00:00Z", sidechain=False):
    return {
        "type": "user",
        "sessionId": session,
        "uuid": uuid,
        "timestamp": ts,
        "isSidechain": sidechain,
        "message": {"role": "user", "content": text},
    }


def assistant_entry(blocks, session="s1", uuid="a1", ts="2026-08-01T10:01:00Z"):
    return {
        "type": "assistant",
        "sessionId": session,
        "uuid": uuid,
        "timestamp": ts,
        "message": {"role": "assistant", "content": blocks},
    }


def tool_use_block(name="Bash", cmd="ls -la"):
    return {"type": "tool_use", "id": "t1", "name": name, "input": {"command": cmd}}


def tool_result_entry(payload, session="s1"):
    return {
        "type": "user",
        "sessionId": session,
        "uuid": "tr1",
        "timestamp": "2026-08-01T10:02:00Z",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": payload}],
        },
    }


def thinking_entry(session="s1"):
    return {
        "type": "assistant",
        "sessionId": session,
        "uuid": "th1",
        "timestamp": "2026-08-01T10:03:00Z",
        "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "",
                                                      "signature": "sig"}]},
    }


def write_transcript(root: Path, project: str, name: str, entries: list[dict]) -> Path:
    d = root / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def append_entries(path: Path, entries: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def write_memory(root: Path, project: str, name: str, body: str) -> Path:
    d = root / project / "memory"
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(body, encoding="utf-8")
    return path
