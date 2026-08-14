"""Secret-shaped-string filter, applied on ingest (report §13.1, failure mode F9).

The corpus contains OAuth callback URLs with live authorization codes, pasted
tokens and private keys, and retrieval surfaces clusters of them because that is
exactly what a retriever is for. This redacts the obvious shapes before anything
reaches the index.

It is a filter, not a guarantee. Report O14 explicitly leaves the false-negative
rate on this material unmeasured, so the index file is still to be treated as
exactly as sensitive as the transcripts themselves.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_MARK = "[REDACTED-{}]"


def _mark(kind: str) -> str:
    return _MARK.format(kind)


# (name, pattern, replacement) - replacement may reference groups to keep the
# surrounding context (a URL's parameter name, an assignment's key) readable.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "PRIVATE-KEY",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        _mark("PRIVATE-KEY"),
    ),
    ("SSH-KEY", re.compile(r"\bssh-(?:rsa|ed25519|dss)\s+[A-Za-z0-9+/=]{40,}"), _mark("SSH-KEY")),
    ("AWS-KEY", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"), _mark("AWS-KEY")),
    (
        "GITHUB-TOKEN",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        _mark("GITHUB-TOKEN"),
    ),
    ("SLACK-TOKEN", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"), _mark("SLACK-TOKEN")),
    ("API-KEY", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}\b"), _mark("API-KEY")),
    ("GOOGLE-KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), _mark("GOOGLE-KEY")),
    (
        "JWT",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        _mark("JWT"),
    ),
    (
        "BEARER",
        re.compile(r"(?i)\b(authorization\s*:\s*bearer|bearer)\s+[A-Za-z0-9._~+/=-]{16,}"),
        r"\1 " + _mark("BEARER"),
    ),
    (
        "URL-SECRET",
        re.compile(
            r"(?i)([?&](?:code|access_token|id_token|refresh_token|token|api[_-]?key"
            r"|client_secret|secret|password|pwd|sig|signature)=)[^\s&#\"'<>]{6,}"
        ),
        r"\1" + _mark("URL-SECRET"),
    ),
    (
        "ASSIGNED-SECRET",
        re.compile(
            # The name may be prefixed (DB_PASSWORD, GH_API_KEY), so no \b in front.
            r"(?i)([A-Za-z0-9_.-]*(?:password|passwd|pwd|secret|api[_-]?key|apikey"
            r"|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)\s*[:=]\s*)"
            r"[\"']?([A-Za-z0-9._/+=~-]{8,})[\"']?"
        ),
        r"\1" + _mark("ASSIGNED-SECRET"),
    ),
]

# F9's own yardstick: 1.6% of eval queries contain an unbroken token of >=45
# characters. Length alone would eat long file paths and base64 images, so the
# token must also look high-entropy and mixed.
_LONG_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{45,}(?![A-Za-z0-9+/=_-])")
_ENTROPY_FLOOR = 3.2


def _shannon(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_random(tok: str) -> bool:
    has_digit = any(c.isdigit() for c in tok)
    has_alpha = any(c.isalpha() for c in tok)
    has_upper = any(c.isupper() for c in tok)
    if not (has_digit and has_alpha):
        return False
    if not (has_upper or "=" in tok or "+" in tok or "/" in tok):
        # all-lowercase-with-digits is usually a hash or an id; still redact if
        # it is genuinely high entropy, which the caller checks next.
        pass
    return _shannon(tok) >= _ENTROPY_FLOOR


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, {kind: n_redactions})."""
    counts: dict[str, int] = {}
    if not text:
        return text, counts
    for name, pattern, repl in _RULES:
        text, n = pattern.subn(repl, text)
        if n:
            counts[name] = counts.get(name, 0) + n

    def _long(m: re.Match[str]) -> str:
        tok = m.group(0)
        if tok.startswith("[REDACTED-") or not _looks_random(tok):
            return tok
        counts["LONG-TOKEN"] = counts.get("LONG-TOKEN", 0) + 1
        return _mark("LONG-TOKEN")

    text = _LONG_TOKEN.sub(_long, text)
    return text, counts
