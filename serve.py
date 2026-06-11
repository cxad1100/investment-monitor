"""Local live server for the dashboard.

Unlike the static HTML build, every page load re-fetches live prices and rebuilds
the report — so refreshing the browser (or the "Update to now" button) shows data
as of right now. Local only; no deps beyond the stdlib + the project.

  .venv/bin/python serve.py        # then open http://localhost:8000

The static build (build_report.py / GitHub Pages) is unchanged.
"""

import http.server
import socketserver
import webbrowser
from datetime import datetime

import build_report as B

PORT = 8000

# Spinner shown while the server rebuilds; the report loads in a hidden iframe so
# Plotly's inline scripts execute correctly, then we reveal it.
LOADER = """<!doctype html><html><head><meta charset="utf-8"><title>Updating…</title>
<style>
body{background:#1e1e1e;color:#d4d4d4;margin:0;height:100vh;display:flex;flex-direction:column;
 align-items:center;justify-content:center;font-family:-apple-system,"Segoe UI",sans-serif}
.sp{width:40px;height:40px;border:3px solid #2d2d2d;border-top-color:#569cd6;border-radius:50%;
 animation:s .8s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
p{color:#808080;font-size:14px;margin-top:16px}
iframe{position:fixed;inset:0;width:100%;height:100%;border:0;display:none;background:#1e1e1e}
</style></head><body>
<div class="sp" id="spin"></div><p id="msg">Fetching live prices &amp; rebuilding…</p>
<iframe id="f" src="/report"></iframe>
<script>
var f=document.getElementById('f');
f.onload=function(){document.getElementById('spin').style.display='none';
 document.getElementById('msg').style.display='none';f.style.display='block';};
f.onerror=function(){document.getElementById('msg').textContent='Build failed — see terminal.';};
</script></body></html>"""

# Fixed bar injected into the served report: timestamp + reload (re-triggers a build).
def _bar() -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    return f"""<div style="position:fixed;top:0;right:0;z-index:99999;background:#252526;
border:1px solid #2d2d2d;border-radius:0 0 0 6px;padding:6px 12px;display:flex;gap:12px;
align-items:center;font-family:'SF Mono',ui-monospace,Menlo,monospace;font-size:12px;color:#808080">
<span>data as of {ts}</span>
<a href="/" target="_top" style="color:#569cd6;text-decoration:none;font-weight:600">↻ Update to now</a>
</div>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/report"):
            print(f"[{datetime.now():%H:%M:%S}] rebuilding (live prices)…")
            try:
                d = B.gather()
                html = B.build(d, public=False)
                html = html.replace("<main>", "<main>" + _bar(), 1)
            except Exception as e:
                html = f"<body style='background:#1e1e1e;color:#d16969;font-family:monospace;padding:40px'>build failed: {e}</body>"
            self._send(html)
        else:
            self._send(LOADER)

    def log_message(self, *args):
        pass  # quiet; we print our own line on rebuild


def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"Live dashboard at {url}  (every refresh = fresh data; Ctrl-C to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
