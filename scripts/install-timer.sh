#!/usr/bin/env bash
# Install the vdb background indexer as a systemd --user timer.
#
# What this does: renders systemd/vdb-index.service.in with this checkout's
# absolute path and the python3 on this PATH, copies both units into
# ~/.config/systemd/user/, and enables+starts the timer. Nothing here needs
# root, and nothing here touches ~/.claude except by running `vdb index`
# itself, exactly as if you'd typed it (read-only against the corpus).
#
# Safe to re-run: it overwrites its own unit files idempotently.
set -euo pipefail

if ! command -v systemctl >/dev/null 2>&1; then
    echo "error: systemctl not found — this installer is systemd-user-timer only" >&2
    echo "       (no daemon framework here; run 'python -m vdb index' by hand or via cron instead)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python3)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$UNIT_DIR"

sed -e "s#@REPO_ROOT@#${REPO_ROOT}#g" -e "s#@PYTHON@#${PYTHON_BIN}#g" \
    "${REPO_ROOT}/systemd/vdb-index.service.in" > "${UNIT_DIR}/vdb-index.service"
cp "${REPO_ROOT}/systemd/vdb-index.timer" "${UNIT_DIR}/vdb-index.timer"

systemctl --user daemon-reload
systemctl --user enable --now vdb-index.timer

echo "installed: ${UNIT_DIR}/vdb-index.{service,timer}"
systemctl --user list-timers vdb-index.timer --no-pager || true

if ! loginctl show-user "$(whoami)" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
    echo
    echo "NOTE: user lingering is not enabled for $(whoami)."
    echo "      Without it, this timer only runs while you have an active login"
    echo "      session (SSH, console, etc.) — it will NOT run in the background"
    echo "      on a headless homeserver between logins. To make it truly"
    echo "      unattended, run once (needs sudo on most distros):"
    echo
    echo "        sudo loginctl enable-linger $(whoami)"
    echo
fi
