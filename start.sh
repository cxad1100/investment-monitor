#!/bin/bash
# Live local dashboard — the primary way to view the reports. Serves both pages
# (portfolio monitor + Pairs Trading Lab). A normal refresh is instant (builds
# from the on-disk data buffer); the "↻ Update to now" button forces a live
# re-fetch. A failed fetch keeps the last-good values and flags staleness.
# Opens http://localhost:8000. Ctrl-C stops.
#
# The static build is only for the public GitHub Pages snapshot (docs/):
#   .venv/bin/python build_report.py   &&   .venv/bin/python build_pairs_report.py
cd "$(dirname "$0")"
.venv/bin/python serve.py
