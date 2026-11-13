"""Operator command gateway: a small FastAPI app that carries typed operator sentences to the robot.

Security model (the same as runtime/web/server.py, stated here so the two stay comparable):
- `serve` binds loopback or a Tailscale address (100.64.0.0/10) only. Any other host needs `allow_any_host=True`.
- Every route except POST /api/stop needs `Authorization: Bearer <token>`, compared in constant time.
- POST /api/stop works WITHOUT the token and is exempt from rate limits: anyone who can reach it may stop the robot,
  nobody without the token can move it.
- Per-IP limits, sliding one-second windows: `rate_per_s` requests for everything except stop, and at most
  FAILED_AUTH_PER_S failed authentications. Wrong-token guesses are refused before they are compared, so guessing runs
  at the failed-auth rate, not at the request rate.
- A parsed Stop sentence runs at once. Any other command is only stored as the single pending command and runs after
  POST /api/confirm with its id, before it expires.
- Fail closed: an unexpected exception in a route answers 500 with a generic message and the robot is not commanded.

Stdlib plus fastapi, pydantic, uvicorn.
"""

from __future__ import annotations

import dataclasses
import hmac
import ipaddress
import logging
import secrets
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from humanoid_arc_bme.onboard.command.grammar import MAX_CHARS
from humanoid_arc_bme.onboard.command.types import Command, Reject, Status, Stop, intent_of
from humanoid_arc_bme.onboard.stack import RobotStack, SensorFrame
from humanoid_arc_bme.onboard.task.executor import FAILED, TaskStatus

log = logging.getLogger(__name__)

FAILED_AUTH_PER_S = 5
ID_MAX_CHARS = 64
TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")
SECURITY_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
_BEARER = "Bearer "


class CommandService(Protocol):
    """What the gateway needs from the robot. Implementations serialise access to the robot themselves."""

    def parse(self, text: str) -> Command: ...

    def submit(self, cmd: Command) -> TaskStatus: ...

    def stop(self, why: str) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def scene(self) -> list[dict[str, Any]]: ...

    def ready(self) -> tuple[bool, str]: ...


class StackCommandService:
    """CommandService over a RobotStack.

    `frame_provider()` returns the newest SensorFrame, or None when there is none yet. `lock` is shared with the
    control loop, which holds it while it calls `stack.step`; every method here holds it briefly. `frame_provider`
    must not take `lock` itself.
    """

    def __init__(self, stack: RobotStack, frame_provider: Callable[[], SensorFrame | None], lock: threading.Lock) -> None:
        self._stack = stack
        self._frame_provider = frame_provider
        self._lock = lock

    def parse(self, text: str) -> Command:
        with self._lock:
            return self._stack.parse(text)

    def submit(self, cmd: Command) -> TaskStatus:
        with self._lock:
            frame = self._frame_provider()
            if frame is None:
                return TaskStatus(state=FAILED, message="no sensor data: command not applied", command=intent_of(cmd))
            return self._stack.command(cmd, frame)

    def stop(self, why: str) -> None:
        with self._lock:
            self._stack.stop(why)

    def status(self) -> dict[str, Any]:
        with self._lock:
            st = self._stack.executor.status
            # RobotStack keeps the runner's safety level private and has no accessor for it.
            safety, reasons = self._stack.safety, self._stack.reasons
            return {"state": st.state, "phase": st.phase, "message": st.message, "command": st.command,
                    "safety": safety, "reasons": list(reasons)}

    def scene(self) -> list[dict[str, Any]]:
        with self._lock:
            frame = self._frame_provider()
            if frame is None:
                return []
            objects = self._stack.scene(frame.stamp)
            return [{"label": o.label, "color": o.color, "range_m": o.range_m, "confidence": o.confidence,
                     "position_body": list(o.position_body)} for o in objects]

    def ready(self) -> tuple[bool, str]:
        with self._lock:
            if self._frame_provider() is None:
                return False, "no sensor data yet"
            return True, ""


class _TextBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(max_length=MAX_CHARS)


class _IdBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictStr = Field(min_length=1, max_length=ID_MAX_CHARS)


@dataclasses.dataclass(frozen=True)
class _Pending:
    id: str
    text: str
    command: Command
    expires_at: float


class _RateWindow:
    """Events per key in the last second. Thread-safe; `take` records an event only while under the limit."""

    def __init__(self, limit: int, clock: Callable[[], float], window_s: float = 1.0) -> None:
        self._limit = limit
        self._clock = clock
        self._window_s = window_s
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> deque[float]:
        """The key's events inside the window, pruned; empty keys are dropped. Call with the lock held."""
        now = self._clock()
        events = self._events.get(key, deque())
        while events and now - events[0] >= self._window_s:
            events.popleft()
        if events:
            self._events[key] = events
        else:
            self._events.pop(key, None)
        return events

    def full(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key)) >= self._limit

    def take(self, key: str) -> bool:
        """Record one event and return True, or return False (nothing recorded) when the window is full."""
        with self._lock:
            events = self._recent(key)
            if len(events) >= self._limit:
                return False
            self._events.setdefault(key, events).append(self._clock())
            return True


class _Gateway:
    """The single pending command and its expiry. All access is under one lock; routes run in a thread pool."""

    def __init__(self, service: CommandService, ttl_s: float, clock: Callable[[], float]) -> None:
        self._service = service
        self._ttl_s = ttl_s
        self._clock = clock
        self._pending: _Pending | None = None
        self._lock = threading.Lock()

    def _live(self) -> _Pending | None:
        """The pending command, or None; an expired one is dropped. Call with the lock held."""
        if self._pending is not None and self._clock() >= self._pending.expires_at:
            self._pending = None
        return self._pending

    def store(self, text: str, cmd: Command) -> _Pending:
        with self._lock:
            pending = _Pending(uuid.uuid4().hex, text, cmd, self._clock() + self._ttl_s)
            self._pending = pending
            return pending

    def view(self) -> dict[str, Any] | None:
        with self._lock:
            pending = self._live()
        if pending is None:
            return None
        return {"id": pending.id, "text": pending.text, "command": _describe(pending.command)}

    def take(self, pending_id: str) -> Command:
        """Remove the pending command and return it, or raise HTTPException: 404 no match, 410 expired."""
        with self._lock:
            pending = self._pending
            if pending is None or pending.id != pending_id:
                raise HTTPException(404, "no pending command with that id")
            self._pending = None   # consumed, or expired: either way it can never run
            if self._clock() >= pending.expires_at:
                raise HTTPException(410, "the pending command has expired; send it again")
            return pending.command

    def clear(self) -> None:
        with self._lock:
            self._pending = None


def _describe(cmd: Command) -> dict[str, Any]:
    """A JSON-ready view of a command: its intent name and its fields."""
    return {"intent": intent_of(cmd), **dataclasses.asdict(cmd)}


def _status_view(st: TaskStatus) -> dict[str, str]:
    return {"state": st.state, "phase": st.phase, "message": st.message, "command": st.command}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _token_matches(header: str | None, token: str) -> bool:
    if header is None or not header.startswith(_BEARER):
        return False
    return hmac.compare_digest(header[len(_BEARER):].encode(), token.encode())


def _require_ready(service: CommandService) -> None:
    ready, reason = service.ready()
    if not ready:
        raise HTTPException(503, reason or "robot not ready")


def create_app(service: CommandService, token: str, *, confirm_ttl_s: float = 20.0,
               clock: Callable[[], float] = time.monotonic, rate_per_s: int = 20) -> FastAPI:
    """The gateway app. Raises ValueError for an empty token or a non-positive limit (fail closed)."""
    if not token:
        raise ValueError("token must be a non-empty string")
    if confirm_ttl_s <= 0:
        raise ValueError(f"confirm_ttl_s must be positive, got {confirm_ttl_s}")
    if rate_per_s < 1:
        raise ValueError(f"rate_per_s must be at least 1, got {rate_per_s}")

    app = FastAPI(title="humanoid_arc command gateway", docs_url=None, redoc_url=None, openapi_url=None)
    requests_per_ip = _RateWindow(rate_per_s, clock)
    failed_per_ip = _RateWindow(FAILED_AUTH_PER_S, clock)
    gateway = _Gateway(service, confirm_ttl_s, clock)

    def require_token(request: Request) -> None:
        ip = _client_ip(request)
        if not requests_per_ip.take(ip):
            raise HTTPException(429, "rate limit exceeded")
        if failed_per_ip.full(ip):
            raise HTTPException(429, "too many failed authentication attempts")
        if _token_matches(request.headers.get("authorization"), token):
            return
        if not failed_per_ip.take(ip):
            raise HTTPException(429, "too many failed authentication attempts")
        raise HTTPException(401, "invalid or missing token", headers={"WWW-Authenticate": "Bearer"})

    auth = [Depends(require_token)]

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Any]) -> Any:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        log.error("gateway error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500, headers=SECURITY_HEADERS)

    @app.post("/api/stop")
    def stop() -> dict[str, bool]:
        try:
            service.stop("operator stop (gateway)")
        except Exception:  # noqa: BLE001 - the stop is answered either way; the failure is logged
            log.exception("stop failed in the robot service")
        return {"ok": True}

    @app.get("/api/status", dependencies=auth)
    def get_status() -> dict[str, Any]:
        return {**service.status(), "pending": gateway.view()}

    @app.get("/api/scene", dependencies=auth)
    def get_scene() -> dict[str, list[dict[str, Any]]]:
        return {"objects": service.scene()}

    @app.post("/api/command", dependencies=auth)
    def post_command(body: _TextBody) -> dict[str, Any]:
        _require_ready(service)
        cmd = service.parse(body.text)
        if isinstance(cmd, Reject):
            return {"accepted": False, "reason": cmd.reason}
        if isinstance(cmd, Stop):
            service.stop("operator stop sentence")
            return {"accepted": True, "executed": True, "command": _describe(cmd)}
        if isinstance(cmd, Status):
            return {"accepted": True, "executed": True, "command": _describe(cmd), "answer": service.status()}
        pending = gateway.store(body.text, cmd)
        return {"accepted": True, "needs_confirm": True, "id": pending.id, "command": _describe(cmd),
                "expires_in_s": confirm_ttl_s}

    @app.post("/api/confirm", dependencies=auth)
    def post_confirm(body: _IdBody) -> dict[str, Any]:
        _require_ready(service)
        cmd = gateway.take(body.id)
        return {"executed": True, "status": _status_view(service.submit(cmd))}

    @app.post("/api/cancel", dependencies=auth)
    def post_cancel() -> dict[str, bool]:
        gateway.clear()
        return {"ok": True}

    return app


def check_bind_host(host: str, allow_any_host: bool = False) -> None:
    """Raise ValueError unless `host` is loopback or inside the Tailscale range, or `allow_any_host` is set."""
    if allow_any_host or host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"host {host!r} is not an IP address; bind 127.0.0.1 or a Tailscale address "
                         "(100.64.0.0/10), or pass allow_any_host=True") from None
    if address.is_loopback or address in TAILSCALE_CGNAT:
        return
    raise ValueError(f"host {host} is neither loopback nor a Tailscale address (100.64.0.0/10); "
                     "pass allow_any_host=True to bind it anyway")


def serve(service: CommandService, token: str | None = None, host: str = "127.0.0.1", port: int = 8090,
          allow_any_host: bool = False) -> None:
    """Run the gateway with uvicorn until interrupted. A generated token is printed once, at startup."""
    check_bind_host(host, allow_any_host)
    if not 1 <= port <= 65535:
        raise ValueError(f"port must be 1..65535, got {port}")
    generated = token is None
    if token is None:
        token = secrets.token_urlsafe(24)
    app = create_app(service, token)
    if generated:
        print(f"command gateway token (shown once): {token}", flush=True)
    print(f"command gateway bound to http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", proxy_headers=False)
