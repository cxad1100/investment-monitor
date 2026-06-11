#!/bin/bash
# Live local dashboard — every browser refresh re-fetches prices and rebuilds.
# (Static build: ./start.sh ; GitHub Pages: docs/index.html)
cd "$(dirname "$0")"
.venv/bin/python serve.py
