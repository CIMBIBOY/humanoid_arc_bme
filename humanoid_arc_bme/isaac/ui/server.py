"""HTTP side of the live view: one page, a small JSON / JPEG API over `LiveSim`, and the stack's own command gateway under /gw.

Everything except the page itself and STOP needs the bearer token. Bind it to loopback or a Tailscale address only
(`command_gateway.check_bind_host` is applied by `serve`).
"""


# No `from __future__ import annotations` here: FastAPI resolves the handlers' annotations (Request, the request models) at run time,
# and with string annotations the nested handlers lose them and every request fails with 422.
import hmac
import threading
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"


def create_ui_app(live, token: str):
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from pydantic import BaseModel

    from humanoid_arc_bme.onboard.runtime.command_gateway import StackCommandService, create_app

    if not token:
        raise ValueError("a token is required")
    app = FastAPI(title="G1 simulation view", docs_url=None, redoc_url=None, openapi_url=None)
    ses = live.ses

    def auth(request: Request) -> None:
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode("utf-8", "replace"), f"Bearer {token}".encode()):
            raise HTTPException(status_code=401, detail="token")

    def client_of(request: Request) -> str:
        return (request.headers.get("x-client") or "ui")[:40]

    class Heartbeat(BaseModel):
        held: bool = False
        mode: str = "ros"
        vx: float = 0.0
        vy: float = 0.0
        wz: float = 0.0

    class Goal(BaseModel):
        x: float
        y: float

    class Mode(BaseModel):
        mode: str

    class Fault(BaseModel):
        name: str
        on: bool

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    # the shared operator UI (same files the robot server serves): /ui/ talks to this server's heartbeat, cloud, plan and fault routes
    ui_dir = Path(__file__).resolve().parents[2] / "onboard" / "runtime" / "web" / "static"
    ui_files = {"": ("index.html", "text/html; charset=utf-8"), "index.html": ("index.html", "text/html; charset=utf-8"),
                "app.js": ("app.js", "application/javascript; charset=utf-8"), "style.css": ("style.css", "text/css; charset=utf-8")}

    @app.get("/ui/{name:path}")
    def shared_ui(name: str) -> Response:
        if name not in ui_files:
            raise HTTPException(status_code=404, detail="not found")
        f, ctype = ui_files[name]
        return Response(content=(ui_dir / f).read_bytes(), media_type=ctype, headers={"Cache-Control": "no-store"})

    @app.get("/api/plan", dependencies=[Depends(auth)])
    def plan() -> dict:
        return live.plan

    @app.post("/api/heartbeat", dependencies=[Depends(auth)])
    def heartbeat(hb: Heartbeat, request: Request) -> dict:
        """The operator's hold-to-run heartbeat, as the robot's UI sends it. No heartbeat, no motion: the runtime's deadman."""
        st = dict(live.snap["state"])
        try:
            st["heartbeat"] = ses.rt.heartbeat(hb.model_dump(), client=client_of(request))
        except Exception as exc:  # noqa: BLE001 - refused heartbeats are shown, not raised
            st["heartbeat_error"] = f"{type(exc).__name__}: {exc}"[:160]
        return st

    @app.get("/api/img/{name}.jpg", dependencies=[Depends(auth)])
    def image(name: str) -> Response:
        got = live.image(name)
        if got is None:
            raise HTTPException(status_code=404, detail="no such image yet")
        return Response(content=got[1], media_type="image/jpeg", headers={"Cache-Control": "no-store", "X-Frame": str(got[0])})

    @app.get("/api/cloud", dependencies=[Depends(auth)])
    def cloud() -> JSONResponse:
        return JSONResponse(live.snap["cloud"] or {"frame": -1, "scan": [], "map": [], "trail": []})

    @app.get("/api/map", dependencies=[Depends(auth)])
    def lidar_map() -> dict:
        return {"cells": live.map_cells()}

    @app.post("/api/stop")
    def stop() -> dict:                                              # no token: STOP must always work
        return live.call("stop", timeout=1.0)

    @app.post("/api/arm", dependencies=[Depends(auth)])
    def arm(request: Request) -> dict:
        return live.call("arm", client=client_of(request))

    @app.post("/api/disarm", dependencies=[Depends(auth)])
    def disarm() -> dict:
        return live.call("disarm")

    @app.post("/api/reset", dependencies=[Depends(auth)])
    def reset() -> dict:
        return live.call("reset", timeout=30.0)

    @app.post("/api/goal", dependencies=[Depends(auth)])
    def goal(g: Goal) -> dict:
        return live.call("goal", x=g.x, y=g.y)

    @app.post("/api/parser", dependencies=[Depends(auth)])
    def parser(m: Mode) -> dict:
        return live.call("parser", mode=m.mode)

    @app.post("/api/fault", dependencies=[Depends(auth)])
    def fault(f: Fault) -> dict:
        return live.call("fault", name=f.name, on=f.on)

    # the stack's own gateway (text -> parse -> confirm -> execute), exactly as on the robot
    app.mount("/gw", create_app(StackCommandService(ses.stack, lambda: ses.loop.latest, ses.loop.lock), token, clock=lambda: live.sim.t))
    return app


def serve(live, token: str, host: str, port: int) -> threading.Thread:
    """Run the app with uvicorn in a daemon thread. `host` must be loopback or a Tailscale address."""
    import uvicorn

    from humanoid_arc_bme.onboard.runtime.command_gateway import check_bind_host

    check_bind_host(host)
    server = uvicorn.Server(uvicorn.Config(create_ui_app(live, token), host=host, port=port, log_level="warning", access_log=False))
    th = threading.Thread(target=server.run, name="ui-server", daemon=True)
    th.start()
    return th
