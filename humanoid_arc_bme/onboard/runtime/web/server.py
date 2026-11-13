"""Operator web UI server: standard library only (http.server + JSON polling, no websockets, no CDN).

Security model:
- Binds 127.0.0.1. The DEFAULT and recommended access is an SSH tunnel from the laptop
  (`ssh -L 8080:127.0.0.1:8080 <user>@<jetson>`). `--lan` (0.0.0.0) is refused by the CLI unless
  `--lan-plain-http-ack` is also given (UI-05): plain HTTP, so anyone on the network path can read the token and drive.
- A random token (printed once at startup) is required for every /api call (header `Authorization: Bearer <t>`; GET of
  camera images and log downloads may pass ?token=). The page reads it from the URL fragment (#token=...), which the
  browser never sends to the server.
- Exception: POST /api/stop works WITHOUT the token. Anyone who can reach the page can stop the robot; nobody without
  the token can move it.
- While armed, only the client id that armed may send heartbeats (others can only STOP).
- UI-07: at most MAX_CONCURRENT requests are handled at once and each client IP at most RATE_PER_S requests per second
  (failed-auth requests AUTH_FAIL_PER_S); beyond that 503 / 429. STOP is exempt from both. status/plots never take the
  control lock. Camera PNGs are encoded once per frame.
- A heartbeat while the control loop is down answers 503 (the UI shows NO CONTROL LOOP).
"""

from __future__ import annotations

import collections
import hmac
import json
import math
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
CTYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
MAX_BODY = 16 * 1024
MAX_CONCURRENT = 8  # request threads handled at once (STOP exempt)
RATE_PER_S = 40  # requests per second per client IP (one UI tab needs ~20: heartbeat 10, status 4, plots 2, camera 2...)
AUTH_FAIL_PER_S = 5  # failed-auth requests per second per client IP
CSP = "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"


def png_from_frame(fr, max_width: int = 320) -> bytes:
    """CameraFrame (rgb8 / bgr8 / mono8) -> PNG bytes, downscaled by striding to <= max_width. Standard library only."""
    ch = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(fr.encoding)
    if ch is None:
        raise ValueError(f"no preview for encoding {fr.encoding}")
    s = max(1, math.ceil(fr.width / max_width))
    w, h = len(range(0, fr.width, s)), len(range(0, fr.height, s))
    rows = []
    data = fr.data
    for y in range(0, fr.height, s):
        row = data[y * fr.width * ch:(y + 1) * fr.width * ch]
        if s == 1:
            px = row
        else:
            px = b"".join(row[x * ch:(x + 1) * ch] for x in range(0, fr.width, s))
        if fr.encoding == "bgr8":
            px = bytes(b for i in range(0, len(px), 3) for b in (px[i + 2], px[i + 1], px[i]))
        rows.append(b"\x00" + px)

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2 if ch == 3 else 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows), 3)) + chunk(b"IEND", b"")


class WebApp:
    def __init__(self, runtime, token: str, host: str = "127.0.0.1", port: int = 8080, clock=time.monotonic):
        if not token or len(token) < 16:
            raise ValueError("token must be at least 16 characters")
        self.rt, self.token, self.host, self.port, self.clock = runtime, token, host, port, clock
        self._auth_fail_log = 0.0
        self._rate: dict[str, collections.deque] = {}
        self._fail: dict[str, collections.deque] = {}
        self._rate_lock = threading.Lock()
        self._busy = threading.BoundedSemaphore(MAX_CONCURRENT)
        self._png: dict[str, tuple] = {}  # camera name -> (frame object id, stamp, png bytes)
        self.httpd: ThreadingHTTPServer | None = None
        self._thread = None

    # ---------------------------------------------------------------- dispatch (also called directly by tests)
    def authorized(self, headers, query) -> bool:
        h = headers.get("Authorization", "")
        tok = h[7:] if h.startswith("Bearer ") else headers.get("X-Token", "")
        if not tok:
            tok = (query.get("token") or [""])[0]
        return bool(tok) and hmac.compare_digest(tok.encode(), self.token.encode())

    def _over(self, table: dict, ip: str, limit: int) -> bool:
        """Sliding 1 s window per IP; True when this request exceeds `limit`."""
        now = self.clock()
        with self._rate_lock:
            q = table.setdefault(ip, collections.deque())
            while q and now - q[0] > 1.0:
                q.popleft()
            if len(q) >= limit:
                return True
            q.append(now)
            if len(table) > 256:  # bounded memory under a scan
                for k in [k for k, v in table.items() if not v][:128]:
                    del table[k]
            return False

    def handle(self, method: str, path: str, query: dict, headers, body: bytes, client_ip: str = "?") -> tuple[int, str, bytes]:
        """Returns (status, content type, body)."""
        if method == "POST" and path == "/api/stop":  # deliberately without the token, before any limit
            confirmed = self.rt.stop("ui" if self.authorized(headers, {}) else f"ui_noauth:{client_ip}")
            st = self.rt.sup.state
            return self._json({"ok": True, "confirmed": bool(confirmed) and st != "armed", "state": st,
                               "stop_latched": self.rt.sup.stop_latched})
        if self._over(self._rate, client_ip, RATE_PER_S):
            return 429, "application/json", b'{"error": "rate limit"}'
        if method == "GET" and path in STATIC_FILES:
            f = STATIC / STATIC_FILES[path]
            return 200, CTYPES[f.suffix], f.read_bytes()
        if not path.startswith("/api/"):
            return 404, "text/plain", b"not found"
        if path == "/api/ping":
            return self._json({"ok": True})
        if not self.authorized(headers, query if method == "GET" else {}):
            if self._over(self._fail, client_ip, AUTH_FAIL_PER_S):
                return 429, "application/json", b'{"error": "rate limit"}'
            now = self.clock()
            if now - self._auth_fail_log > 5.0:
                self._auth_fail_log = now
                self.rt._log_event("auth_failed", {"ip": client_ip, "path": path})
            return 401, "application/json", b'{"error": "token required"}'
        try:
            payload = json.loads(body or b"{}") if method == "POST" else {}
            if not isinstance(payload, dict):
                raise ValueError("JSON object expected")
            return self._route(method, path, query, payload, headers.get("X-Client", "anon"))
        except PermissionError as e:
            return 409, "application/json", json.dumps({"error": str(e)}).encode()
        except RuntimeError as e:  # ControlLoopDown: the control loop is not running
            return 503, "application/json", json.dumps({"error": str(e), "control_loop": "down"}).encode()
        except (ValueError, TypeError, KeyError) as e:
            return 400, "application/json", json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()

    def _json(self, obj, code=200):
        return code, "application/json", json.dumps(obj, default=str).encode()

    def _route(self, method, path, query, p, client):
        rt = self.rt
        if method == "GET":
            if path == "/api/status":
                return self._json(rt.status())
            if path == "/api/plots":
                return self._json(rt.plots(float((query.get("since") or ["0"])[0])))
            if path.startswith("/api/camera/") and path.endswith(".png"):
                name = unquote(path[len("/api/camera/"):-4])
                fr = rt.frames.get(name)
                if fr is None:
                    return 404, "text/plain", b"no frame"
                c = self._png.get(name)
                if c is None or c[0] is not fr:
                    c = (fr, png_from_frame(fr))
                    self._png[name] = c
                return 200, "image/png", c[1]
            if path == "/api/logs":
                return self._json(rt.runlog.files())
            if path.startswith("/api/logs/"):
                name = unquote(path[len("/api/logs/"):])
                if name not in {f["name"] for f in rt.runlog.files()}:  # whitelist: no path traversal
                    return 404, "text/plain", b"no such log"
                return 200, "application/octet-stream", (rt.runlog.dir / name).read_bytes()
            return 404, "text/plain", b"not found"
        if path == "/api/heartbeat":
            return self._json(rt.heartbeat(p, client=client))
        if path == "/api/arm/request":
            return self._json(rt.request_arm())
        if path == "/api/arm/confirm":
            ok, why = rt.confirm_arm(str(p.get("text", "")), client=client)
            return self._json({"ok": ok, "reason": why}, 200 if ok else 409)
        if path == "/api/disarm":
            return self._json({"ok": rt.disarm("operator")})
        if path == "/api/reset_stop":
            rt.reset_stop()
            return self._json({"ok": True})
        if path == "/api/damp":
            return self._json({"ok": True, "result": rt.damp()})
        if path == "/api/controller":
            ok, why = rt.select_controller(str(p.get("name", "")))
            return self._json({"ok": ok, "reason": why}, 200 if ok else 409)
        if path == "/api/release_vendor":
            ok, why = rt.release_vendor(str(p.get("text", "")))
            return self._json({"ok": ok, "reason": why}, 200 if ok else 409)
        if path == "/api/waypoints":
            if rt.nav is None:
                return self._json({"ok": False, "reason": "no navigation module loaded"}, 409)
            wps = [tuple(float(c) for c in w) for w in p.get("waypoints", [])]
            if not wps:
                raise ValueError("no waypoints")
            with rt.lock:
                info = rt.nav.set_waypoints(wps)
            rt._log_event("waypoints_set", {"waypoints": wps, "info": info})
            return self._json({"ok": True, "info": info})
        return 404, "text/plain", b"not found"

    # ---------------------------------------------------------------- http.server glue
    def make_server(self) -> ThreadingHTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "g1_runtime"
            sys_version = ""

            def _do(self, method):
                u = urlparse(self.path)
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return self._send(400, "text/plain", b"bad length")
                if n > MAX_BODY or n < 0:
                    return self._send(413, "text/plain", b"too large")
                is_stop = method == "POST" and u.path == "/api/stop"
                got = app._busy.acquire(blocking=False)
                if not got and not is_stop:
                    return self._send(503, "application/json", b'{"error": "busy"}')
                try:
                    body = self.rfile.read(n) if n else b""
                    try:
                        code, ctype, data = app.handle(method, u.path, parse_qs(u.query), self.headers, body, self.client_address[0])
                    except Exception as e:  # noqa: BLE001 - never kill the server thread
                        code, ctype, data = 500, "application/json", json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
                    self._send(code, ctype, data)
                finally:
                    if got:
                        app._busy.release()

            def _send(self, code, ctype, data):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", CSP)
                self.send_header("Referrer-Policy", "no-referrer")
                if ctype == "application/octet-stream":
                    self.send_header("Content-Disposition", "attachment")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                self._do("GET")

            def do_POST(self):  # noqa: N802
                self._do("POST")

            def log_message(self, fmt, *args):  # heartbeats at 10 Hz would flood the console
                pass

        Handler.timeout = 5.0  # a stalled client cannot hold a request thread for long
        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        return self.httpd

    def start(self) -> None:
        if self.httpd is None:
            self.make_server()
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="g1-web", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
