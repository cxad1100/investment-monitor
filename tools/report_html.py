"""Shared HTML building blocks for the static reports."""

import plotly.graph_objects as go
import plotly.offline

from tools import theme

# Match the CDN plotly.js to the installed python-plotly: a v2 CDN rendering v3
# figure JSON breaks the legend (entrywidth ignored, labels overlap/trim).
_PLOTLY_JS = plotly.offline.get_plotlyjs_version()


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
<script src="https://cdn.plot.ly/plotly-{_PLOTLY_JS}.min.js"></script>
<style>{theme.REPORT_CSS}</style>
</head><body><main>{body}</main>
<script>
// Plotly measures legend text width at first paint; if the webfont paints a beat
// later the legend clip box is left one-char wide (truncated labels). Redraw once
// fonts are ready (and after full load) so widths are remeasured.
(function(){{
  function redraw(){{
    if(!window.Plotly) return;
    document.querySelectorAll('.js-plotly-plot').forEach(function(g){{
      try{{ Plotly.redraw(g); }}catch(e){{}}
    }});
  }}
  if(document.fonts&&document.fonts.ready) document.fonts.ready.then(redraw);
  window.addEventListener('load',function(){{ setTimeout(redraw,150); }});
}})();
</script>
</body></html>"""
