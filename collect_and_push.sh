#!/bin/bash
set -euo pipefail

REPO="/Users/cxmc/code/claudecode/a"
LOG="$REPO/data/collect.log"
DATE=$(date +%Y-%m-%d)

cd "$REPO"
echo "[$DATE $(date +%H:%M)] Starting collection" >> "$LOG"

if .venv/bin/python collect_all.py >> "$LOG" 2>&1; then
    echo "[$DATE $(date +%H:%M)] Collection done" >> "$LOG"
else
    echo "[$DATE $(date +%H:%M)] Collection FAILED" >> "$LOG"
    exit 1
fi

git add data/signals.json
if git diff --cached --quiet; then
    echo "[$DATE $(date +%H:%M)] No changes to push" >> "$LOG"
    exit 0
fi

git commit -m "Weekly signals — $DATE"
git push origin master >> "$LOG" 2>&1
echo "[$DATE $(date +%H:%M)] Pushed signals.json" >> "$LOG"
