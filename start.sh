#!/bin/bash
# Live local dashboard. Every browser refresh (or the "Update to now" button)
# re-fetches prices and rebuilds. If a fetch fails it falls back to the last
# good snapshot (local/report.html). Opens http://localhost:8000. Ctrl-C stops.
#
# Static one-shot build instead (for GitHub Pages docs/): python build_report.py
cd "$(dirname "$0")"
.venv/bin/python serve.py
