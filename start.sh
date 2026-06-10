#!/bin/bash
# Build the static report and open the private version in the browser.
# No server — output is plain HTML:
#   local/report.html  private, full € amounts (gitignored)
#   docs/index.html    public, percentages only (deployed via GitHub Pages)
cd "$(dirname "$0")"
.venv/bin/python build_report.py --open
