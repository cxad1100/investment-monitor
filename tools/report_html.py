"""Shared HTML building blocks for the static reports."""

import plotly.graph_objects as go

from tools import theme


def fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def pct(x, signed=True, nd=1) -> str:
    if x is None:
        return "—"
    s = f"{x:+.{nd}f}%" if signed else f"{x:.{nd}f}%"
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls} mono">{s}</span>'


def card(label: str, value: str) -> str:
    return f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div></div>'


def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{theme.REPORT_CSS}</style>
</head><body><main>{body}</main></body></html>"""
