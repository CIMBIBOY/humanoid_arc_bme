"""The whole pipeline, live: the simulated G1 with the full onboard stack in a scene, and a web page to watch and operate it.

    python.sh scripts/isaac/sim_ui.py --scene <scene dir> --host <127.0.0.1 or a Tailscale address> --port 8765 \
        [--yolo yolo11n-seg.onnx] [--token <token>] [--seconds 3600]

Real time, headless. The page shows the lidar map and the pose, both RGB-D cameras, the detector's masks, and takes operator input
(arm, hold-to-run, velocity, map goal, stop) and sentences (the stack's command gateway, with the Jev fake selectable). Prints the
address with the token. Stand-ins are named on the page: locomotion policy, pose source (no DLIO), Jev (fake).
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=0, help="0 = a free port chosen by the kernel (other services on this machine use many fixed ones)")
ap.add_argument("--token", default=None)
ap.add_argument("--yolo", default=None, help="YOLO-seg ONNX file; without it the simulator's segmentation stands in")
ap.add_argument("--guard", default="fitted", choices=("stock", "fitted"))
ap.add_argument("--seconds", type=float, default=3600.0)
ap.add_argument("--run-dir", default=None)
args, _ = ap.parse_known_args()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for _p in filter(None, os.environ.get("HUMANOID_ARC_DDS_PATHS", "").split(os.pathsep)):
    sys.path.insert(0, _p)
from humanoid_arc_bme.isaac.app import make_app  # noqa: E402

app = make_app("sim_ui", lidar=True)


def main() -> None:
    import socket
    from humanoid_arc_bme.isaac.scenarios import Bench
    from humanoid_arc_bme.isaac.ui.live import LiveSim, RecordingSemantics
    from humanoid_arc_bme.isaac.ui.server import serve
    from humanoid_arc_bme.onboard.command.jev import FakeJev

    head = None
    if args.yolo:
        from humanoid_arc_bme.onboard.perception.yolo_seg import YoloSegConfig, YoloSegDetector
        cfg = YoloSegConfig(conf=0.30)
        det = RecordingSemantics(YoloSegDetector(args.yolo, cfg), f"YOLO ({Path(args.yolo).stem}, stock COCO weights)")
        head = YoloSegDetector(args.yolo, cfg)
    else:
        det = RecordingSemantics(None, "simulator segmentation (stand-in, no YOLO weights)")
    run_dir = args.run_dir or f"/tmp/sim_ui_{os.getpid()}"
    bench = Bench(args.scene, run_dir, guard=args.guard, semantics=det if args.yolo else None, realtime=True)
    if not args.yolo:                                                # wrap the stand-in the bench built, so the page can show its masks
        det.inner = bench.ses.stack.semantics
        bench.ses.stack.semantics = det
        bench.ses.stack.runner.semantics = det
    live = LiveSim(bench, det, head_detector=head, jev_factory=lambda mode: FakeJev(mode=mode))
    token = args.token or secrets.token_urlsafe(12)
    bench.ses.run(2.0)                                               # sensors deliver, the robot settles (DISARMED)
    live.after_step()
    port = args.port
    if not port:
        import socket
        with socket.socket() as sk:
            sk.bind((args.host, 0))
            port = sk.getsockname()[1]
    serve(live, token, args.host, port)
    time.sleep(1.0)
    with socket.socket() as probe:                                   # the server thread logs a bind error and dies: check it listens
        if probe.connect_ex((args.host, port)) != 0:
            raise RuntimeError(f"the UI server is not listening on {args.host}:{port}")
    print(f"SIM_UI_READY http://{args.host}:{port}/#token={token}", flush=True)
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        bench.sim.step()
        live.after_step()
    print("SIM_UI_DONE", flush=True)


FAILED = False
try:
    main()
except Exception:
    print("SIM_UI_TRACEBACK", traceback.format_exc(), flush=True)
    FAILED = True
finally:
    app.close()
if FAILED:
    sys.exit(1)
