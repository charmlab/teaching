#!/usr/bin/env python3
"""Local preview server that never lets the browser cache (so edits always show).
Usage:  python3 serve.py   →  http://localhost:8787"""
import http.server, functools, os

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

port = int(os.environ.get("PORT", 8787))
handler = functools.partial(NoCache, directory=os.path.dirname(os.path.abspath(__file__)))
print(f"Serving on http://localhost:{port}  (no caching)")
http.server.ThreadingHTTPServer(("", port), handler).serve_forever()
