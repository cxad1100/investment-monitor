#!/bin/bash
# Live local dashboard — the primary way to view the reports. Serves both pages
# (portfolio monitor + Pairs Trading Lab). A normal refresh is instant (builds
# from the on-disk data buffer); the "↻ Update to now" button forces a live
# re-fetch. A failed fetch keeps the last-good values and flags staleness.
# Opens http://localhost:8000. Ctrl-C stops.
#
cd "$(dirname "$0")"
# Resolve the venv: local first, else the main repo's (this may be a git worktree,
# where .venv lives in the main checkout — find it via git), else system python3.
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  MAIN="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  MAIN="${MAIN%/.git}"
  [ -x "$MAIN/.venv/bin/python" ] && PY="$MAIN/.venv/bin/python"
fi
[ -x "$PY" ] || PY=python3
exec "$PY" serve.py
