"""python -m humanoid_arc_bme.onboard.runtime {observe|serve|preflight} ...

observe    read and publish only (Foxglove / JSONL), never commands:
           observe --io fake|unitree [--iface enp2s0] --camera none|fake|realsense --jsonl out.jsonl --foxglove 8765 --mcap out.mcap --seconds 10
serve      the operator runtime + web UI. Starts DISARMED; nothing moves until an operator arms in the UI:
           serve --io fake|unitree [--iface eth0] [--models DIR] [--shadow DIR] [--port 8080] [--allow-lowcmd]
                 [--limits config/g1_limits_firstwalk.yaml] [--lan --lan-plain-http-ack]
           The UI binds 127.0.0.1: use an SSH tunnel from the laptop (default, recommended). --lan (0.0.0.0, plain
           HTTP: the token can be sniffed and the robot driven) is refused unless --lan-plain-http-ack is also given.
preflight  read-only checks (versions, network, disk, clock, config, models; --listen S subscribes to rt/lowstate):
           preflight [--iface eth0] [--listen 3] [--models DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .camera import FakeCamera, RealSenseCamera
from .io.fake import FakeRobotIO
from .loop import Runtime
from .telemetry import FoxgloveSink, JsonlSink, MemorySink, MultiSink

ROOT = Path(__file__).resolve().parents[3]


def default_config_dir() -> Path:
    """G1_RUNTIME_CONFIG, else <repo>/config (checkout or --symlink-install), else share/g1_runtime_ros/config of a colcon
    install (found through AMENT_PREFIX_PATH, which `source install/setup.bash` sets)."""
    env = os.environ.get("G1_RUNTIME_CONFIG")
    if env:
        return Path(env)
    if (ROOT / "config" / "g1_limits.yaml").is_file():
        return ROOT / "config"
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        d = Path(prefix) / "share" / "g1_runtime_ros" / "config"
        if prefix and (d / "g1_limits.yaml").is_file():
            return d
    return ROOT / "config"


def build(args, clock=time.monotonic, sleep=time.sleep) -> Runtime:
    if args.io == "fake":
        io = FakeRobotIO(clock)
    else:
        from .io.unitree import UnitreeG1IO

        io = UnitreeG1IO(args.iface, clock=clock)
    cams = {"none": [], "fake": [FakeCamera(clock)], "realsense": None}[args.camera]
    if cams is None:
        cams = [RealSenseCamera(clock)]
    sinks = []
    if args.jsonl:
        sinks.append(JsonlSink(args.jsonl))
    if args.foxglove is not None or args.mcap:
        sinks.append(FoxgloveSink(port=args.foxglove, mcap_path=args.mcap))
    sink = MultiSink(*sinks) if sinks else MemorySink()
    return Runtime(io, sink, cams, hz=args.hz, clock=clock, sleep=sleep)


def lan_addresses() -> list[str]:
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=3).stdout
        return [ln.split()[3].split("/")[0] for ln in out.splitlines() if " inet " in ln and not ln.split()[3].startswith("127.")]
    except Exception:  # noqa: BLE001
        return []


def redact_argv(argv) -> list[str]:
    """UI-08: never log the token."""
    out, skip = [], False
    for a in argv:
        if skip:
            out.append("<redacted>")
            skip = False
        elif a == "--token":
            out.append(a)
            skip = True
        elif a.startswith("--token="):
            out.append("--token=<redacted>")
        else:
            out.append(a)
    return out


class LanRefused(SystemExit):
    pass


def check_lan(args) -> None:
    """UI-05: plain HTTP on 0.0.0.0 only with an explicit acknowledgement."""
    if getattr(args, "lan", False) and not getattr(args, "lan_plain_http_ack", False):
        raise LanRefused("--lan serves plain HTTP on 0.0.0.0: anyone on the network path can read the token and drive the "
                         "robot. Use the default 127.0.0.1 with `ssh -L 8080:127.0.0.1:8080 <user>@<jetson>`. If you must, "
                         "add --lan-plain-http-ack (wired robot network only, never Wi-Fi).")


def build_operator(args, clock=time.monotonic):
    """Everything `serve` needs, without starting threads. Returns (runtime, web app, token)."""
    import yaml

    from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, load_limits

    from .controllers import FakeController, FakeLocoClient, PolicyController, ShadowPolicy, VendorLocoController, list_policies, make_loco_client
    from .operator import load_operator_limits
    from .operator_runtime import OperatorRuntime, RuntimeConfig
    from .path_follow import load_nav
    from .runlog import RunLog, new_run_dir
    from .web.server import WebApp

    check_lan(args)
    cfg_dir = Path(args.config_dir) if args.config_dir else default_config_dir()
    rcfg = yaml.safe_load((cfg_dir / "robot_runtime.yaml").read_text())
    lim_arg = getattr(args, "limits", None)
    limits_path = cfg_dir / "g1_limits.yaml"
    if lim_arg:  # a path, or a file name inside the config directory
        limits_path = Path(lim_arg).expanduser()
        if not limits_path.is_file() and (cfg_dir / limits_path.name).is_file():
            limits_path = cfg_dir / limits_path.name
    limits = load_limits(limits_path)
    op = load_operator_limits(limits_path, limits.base_vel_max)
    safety = SafetyLayer(limits, Watchdog(0.1))
    lc = rcfg.get("lowcmd", {})
    if args.io == "fake":
        io = FakeRobotIO(clock)
    else:
        from .io.unitree import UnitreeG1IO

        io = UnitreeG1IO(args.iface, clock=clock, enable_lowcmd=args.allow_lowcmd, hold_kp=lc.get("hold_kp"), hold_kd=lc.get("hold_kd"),
                         damping_kd=float(lc.get("damping_kd", 8.0)))
    v = rcfg.get("vendor", {})
    vkw = dict(send_hz=float(v.get("send_hz", 10.0)), cmd_duration_s=float(v.get("cmd_duration_s", 0.4)),
               max_consecutive_errors=int(v.get("max_consecutive_errors", 3)), clock=clock)
    controllers: dict = {}
    if args.io == "fake":
        controllers["fake"] = FakeController()
        controllers["vendor"] = lambda: VendorLocoController(FakeLocoClient(), **vkw)  # UI dry run: a fake LocoClient
    else:
        controllers["vendor"] = lambda: VendorLocoController(make_loco_client(float(v.get("rpc_timeout_s", 0.25))), **vkw)
    if getattr(args, "groot", False):
        from .groot_controller import FakeTransport, GrootPlannerController, ZmqPubTransport

        endpoint = (rcfg.get("groot") or {}).get("endpoint", "tcp://127.0.0.1:5556")
        controllers["groot"] = lambda: GrootPlannerController(FakeTransport() if args.io == "fake" else ZmqPubTransport(endpoint),
                                                              enabled=True, limits=limits)
    height = float(rcfg.get("policy_height", 0.78))
    for d in (list_policies(args.models) if args.models else []):
        controllers[f"policy:{d.name}"] = (lambda d=d: PolicyController(d, io, safety, height=height))
    shadow = ShadowPolicy(args.shadow, height=height) if args.shadow else None
    cams = {"none": [], "fake": [FakeCamera(clock)], "realsense": None}[args.camera]
    if cams is None:
        cams = [RealSenseCamera(clock)]
    nav, nav_why = load_nav(rcfg.get("nav_module"), rcfg, clock)
    run_dir = new_run_dir(args.log_root or rcfg.get("log_root", "~/g1_runs"))
    runlog = RunLog(run_dir, state_log_hz=float(rcfg.get("state_log_hz", 10.0)))
    sink = FoxgloveSink(port=args.foxglove) if args.foxglove is not None else None
    rt = OperatorRuntime(io, controllers, op, safety, runlog, sink=sink, cameras=cams, cfg=RuntimeConfig.from_dict(rcfg), shadow=shadow,
                         nav=nav, default_controller="vendor", clock=clock)
    rt._log_event("startup", {"argv": redact_argv(sys.argv), "io": args.io, "lan": args.lan, "allow_lowcmd": args.allow_lowcmd,
                              "nav": nav_why, "config_dir": str(cfg_dir), "limits_file": str(limits_path),
                              "limits_sha256": hashlib.sha256(limits_path.read_bytes()).hexdigest(),
                              "limits_validated": limits.validated_on_robot})
    rt.limits_path = limits_path
    rt.autonomy = None
    if getattr(args, "autonomy", None):
        from .autonomy_main import build_autonomy, load_config

        acfg = load_config(args.autonomy)
        rt.autonomy = build_autonomy(rt, acfg, controller=None)
        rt.autonomy_cfg = acfg
    token = args.token or os.environ.get("G1_RUNTIME_TOKEN") or secrets.token_urlsafe(18)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    app = WebApp(rt, token, host=host, port=args.port if args.port is not None else int(rcfg.get("web_port", 8080)))
    return rt, app, token


def _start_gateway(rt, args) -> str:
    """Start the autonomy loop and the FastAPI command gateway (its own thread). Returns the gateway token."""
    from .command_gateway import StackCommandService, serve as serve_gateway

    loop = rt.autonomy
    gw = rt.autonomy_cfg.get("gateway", {})
    service = StackCommandService(loop.stack, lambda: loop.latest, loop.lock)
    token = secrets.token_urlsafe(24)
    loop.start()
    threading.Thread(target=serve_gateway, kwargs=dict(service=service, token=token, host=gw.get("host", "127.0.0.1"),
                                                       port=int(gw.get("port", 8090))), name="command-gateway", daemon=True).start()
    print(f"  autonomy ON. Arm in the UI, hold the HOLD input while a task runs; commands: command gateway on "
          f"{gw.get('host', '127.0.0.1')}:{gw.get('port', 8090)} (token {token})", flush=True)
    return token


def serve(args) -> int:
    try:
        check_lan(args)
    except LanRefused as e:
        print(f"refused: {e}", file=sys.stderr, flush=True)
        return 2
    rt, app, token = build_operator(args)
    app.make_server()
    hosts = lan_addresses() if args.lan else ["127.0.0.1"]
    print("humanoid_arc_bme.onboard.runtime operator UI (DISARMED; nothing moves until you arm in the UI)", flush=True)
    for h in hosts:
        print(f"  open: http://{h}:{app.port}/#token={token}", flush=True)
    if not args.lan:
        print(f"  bound to 127.0.0.1 only: from the laptop use  ssh -L {app.port}:127.0.0.1:{app.port} <user>@<robot-ip>", flush=True)
    else:
        print("  WARNING: --lan, plain HTTP on 0.0.0.0: the token can be sniffed; wired robot network only", flush=True)
    print(f"  limits: {rt.limits_path} (validated_on_robot={rt.safety.limits.validated_on_robot})", flush=True)
    print(f"  run log: {rt.runlog.dir}", flush=True)
    done = threading.Event()

    def _sig(*_a):
        done.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    rt.start()
    app.start()
    if rt.autonomy is not None:
        _start_gateway(rt, args)
    t_start = time.monotonic()
    try:
        while not done.wait(0.5):
            if args.seconds and time.monotonic() - t_start > args.seconds:
                break
    finally:
        if rt.autonomy is not None:
            rt.autonomy.close()
        rt.close()  # disarms ("shutdown") -> controller safe state, closes logs
        app.close()
    print(f"stopped; logs in {rt.runlog.dir}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="humanoid_arc_bme.onboard.runtime", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["observe", "serve", "preflight"])
    ap.add_argument("--io", choices=["fake", "unitree"], default="fake")
    ap.add_argument("--iface", default=None, help="network interface to the robot (unitree io), e.g. eth0")
    ap.add_argument("--camera", choices=["none", "fake", "realsense"], default="none")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="observe: duration (default 10); serve: 0 = until Ctrl-C")
    ap.add_argument("--jsonl")
    ap.add_argument("--foxglove", type=int, default=None, metavar="PORT", help="serve the Foxglove WebSocket on 127.0.0.1:PORT")
    ap.add_argument("--mcap")
    ap.add_argument("--models", default=None, help="serve/preflight: directory of policy dirs (policy.onnx + layout.json)")
    ap.add_argument("--shadow", default=None, help="serve: run this policy dir in shadow mode (computes, logs, never sends)")
    ap.add_argument("--lan", action="store_true", help="serve: bind 0.0.0.0 (default 127.0.0.1; needs --lan-plain-http-ack)")
    ap.add_argument("--lan-plain-http-ack", action="store_true",
                    help="serve: acknowledge that --lan is plain HTTP (token sniffable, anyone on the path can drive)")
    ap.add_argument("--limits", default=None, help="serve: S1 + operator limits file (default <config>/g1_limits.yaml; "
                                                   "first walk: g1_limits_firstwalk.yaml)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--token", default=None, help="serve: fixed token (default: random, printed)")
    ap.add_argument("--groot", action="store_true", help="serve: register the GR00T planner controller (the deploy runs with --input-type zmq_manager)")
    ap.add_argument("--autonomy", default=None, metavar="YAML", help="serve: run the autonomy stack from this config (config/autonomy.yaml); needs --groot for arm tasks")
    ap.add_argument("--allow-lowcmd", action="store_true", help="serve: create the gated rt/lowcmd path for the policy controller (Hold)")
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--log-root", default=None)
    ap.add_argument("--listen", type=float, default=0.0, help="preflight: subscribe to rt/lowstate for this many seconds")
    args = ap.parse_args(argv)
    if args.mode == "serve":
        return serve(args)
    if args.mode == "preflight":
        from . import preflight

        text, code = preflight.report(preflight.run_all(args.iface, args.listen, args.models, args.config_dir or default_config_dir(),
                                                        args.log_root or "~/g1_runs"))
        print(text)
        return code
    rt = build(args)
    try:
        print(json.dumps(rt.run(args.seconds or 10.0), indent=1))
    finally:
        rt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
