"""Tiny web server for the frontend (standard library only).

Run:  python server.py        then open  http://localhost:8000
"""
import json, sys, pathlib
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import outlook

yo = outlook.YieldOutlook()
OPTIONS = yo.options()
OPTIONS["diseases"] = list(outlook.DISEASE_SEVERITY)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path            # ignore ?query so shareable links work
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/options":
            self._send(200, OPTIONS)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self._send(404, {"error": "not found"})
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            loss = 0.0
            if q.get("disease") in outlook.DISEASE_SEVERITY:
                loss = outlook.DISEASE_SEVERITY[q["disease"]] * float(q.get("share", 0)) / 100
            r = yo.predict(crop=q["crop"], state=q["state"], district=q["district"], season=q["season"],
                           year=int(q["year"]) if q.get("year") else None, scenario=q.get("scenario", "normal"),
                           yield_loss=loss)
            r["assumed_loss"] = loss
            self._send(200, r)
        except (ValueError, KeyError) as e:
            self._send(400, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("Crop Yield Outlook running at http://localhost:8000  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
