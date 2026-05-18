#!/bin/bash
# Monitor — start dashboard
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run app.py
