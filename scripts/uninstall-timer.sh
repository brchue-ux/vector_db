#!/usr/bin/env bash
# Remove the vdb background indexer. Does not touch the index database or
# ~/.claude — only the systemd --user units this installed.
set -euo pipefail

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if command -v systemctl >/dev/null 2>&1; then
    systemctl --user disable --now vdb-index.timer 2>/dev/null || true
fi
rm -f "${UNIT_DIR}/vdb-index.service" "${UNIT_DIR}/vdb-index.timer"
if command -v systemctl >/dev/null 2>&1; then
    systemctl --user daemon-reload
fi

echo "removed: ${UNIT_DIR}/vdb-index.{service,timer}"
echo "note: the index database itself was not touched — it still exists and can still be queried."
echo "note: if you enabled lingering just for this (sudo loginctl enable-linger \$(whoami)), that is"
echo "      not undone here; disable it yourself if nothing else on this box needs it:"
echo "      sudo loginctl disable-linger \$(whoami)"
