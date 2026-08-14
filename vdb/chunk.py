"""Message-boundary chunking (report §11.1-§11.3, recommendation §13.1).

One chunk per message. Messages longer than ~1,800 characters are split at
paragraph boundaries, then line boundaries, then hard character boundaries.
No overlap. Nothing in the chunk but the message's own text.

This is the single largest measured effect in the whole study: message
boundaries beat 600-character windows by +0.085 recall@10, CI [+0.044, +0.126],
*at matched chunk size*. Adding a two-line provenance header costs -0.041, and
prepending the previous turn cuts cross-session recall by more than half
(0.533 -> 0.250). Provenance is stored as metadata alongside the chunk and is
deliberately not part of the indexed text.
"""

from __future__ import annotations

import re

LIMIT = 1800

_PARA = re.compile(r"(\n\s*\n)")
_LINE = re.compile(r"(\n)")


def _split_keep(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split into pieces whose concatenation is exactly `text`.

    Separators are glued onto the end of the preceding piece so nothing is lost.
    """
    parts = pattern.split(text)
    pieces: list[str] = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if seg or sep:
            pieces.append(seg + sep)
    return pieces or [text]


def _hard(text: str, limit: int) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [text]


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Greedily pack pieces into runs of at most `limit` characters.

    A single piece longer than the limit is emitted alone; the caller has
    already decided how it gets broken down further.
    """
    out: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        out.append(cur)
    return out


def split_message(text: str, limit: int = LIMIT) -> list[str]:
    """Split one message's text into chunk texts.

    Lossless up to leading/trailing whitespace on each chunk: concatenating the
    raw (unstripped) pieces reproduces the input exactly, which `raw_pieces`
    exposes for tests.
    """
    return [c.strip() for c in raw_pieces(text, limit) if c.strip()]


def raw_pieces(text: str, limit: int = LIMIT) -> list[str]:
    """The same split, without stripping, so `"".join(...) == text` holds."""
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    out: list[str] = []
    for para in _pack(_split_keep(text, _PARA), limit):
        if len(para) <= limit:
            out.append(para)
            continue
        for line in _pack(_split_keep(para, _LINE), limit):
            if len(line) <= limit:
                out.append(line)
            else:
                out.extend(_hard(line, limit))
    return out
