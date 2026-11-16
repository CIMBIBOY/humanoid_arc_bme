"""seed's scene -> G1 with the full stack -> scenarios S1-S7 -> JSON report. Headless Isaac Sim.

    python.sh scripts/isaac/run_scenarios.py --scene <scene dir> --out report.json [--only s1,s2] [--seed 0]

The scene is any directory with a `scene.json` manifest and its USD file. The report has one entry per scenario (pass / fail / not
applicable, metrics, failure reason), the timing of the run and the list of stand-ins in use. Checkpointed: the report is rewritten
after every scenario, so a run that is killed keeps what it finished.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="s1,s2,s3,s4,s5,s6,s7")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--record", default=None, help="directory for one .npz per sensor frame (images, raw lidar, IMU, truth): large, use a data drive")
ap.add_argument("--guard", default="fitted", choices=("stock", "fitted"), help="pipeline runner obstacle geometry (see scenarios.GUARD_PROFILES)")
args, _ = ap.parse_known_args()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_arc_bme.isaac.app import make_app  # noqa: E402

app = make_app("run_scenarios", lidar=True)

STAND_INS = {"locomotion": "NVIDIA AGILE lower-body policy behind the SONIC planner interface (isaac/standin.py), lockstep in-process link",
             "detector": "simulator segmentation behind the SemanticsModel interface (no YOLO weights on this machine)",
             "localizer": "simulator ground truth + 5 mm / 3 mrad white noise behind the Localizer interface (DLIO not run)",
             "jev": "FakeJev (placeholder API schema)", "lidar": "40-line approximation of the Mid-360 rosette, 2 slices per frame"}


def main() -> dict:
    from humanoid_arc_bme.isaac.scenarios import BOUNDS, Bench

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bench = Bench(args.scene, out.with_suffix("").as_posix() + "_runlog", seed=args.seed, guard=args.guard, record_dir=args.record)
    rep = {"scene": bench.man["scene_id"], "seed": bench.man.get("seed"), "profile": bench.man.get("profile"), "bounds": BOUNDS, "stand_ins": STAND_INS, "guard_profile": args.guard,
           "home": [round(v, 3) for v in bench.home], "scenarios": {}}
    for k in args.only.split(","):
        rep["scenarios"].update(bench.run_all((k,)))
        rep["timing"] = bench.timing()
        out.write_text(json.dumps(rep, indent=1, default=str))
        r = rep["scenarios"][k]
        print("SCENARIO", k, "na" if r.get("na") else "PASS" if r["pass"] else "FAIL", r.get("reason", ""), f"({r['sim_s']} s sim, {r['wall_s']} s wall)", flush=True)
    rep["complete"] = True
    out.write_text(json.dumps(rep, indent=1, default=str))
    return rep


FAILED = False
try:
    r = main()
    print("SCENARIOS_DONE", json.dumps({k: ("na" if v.get("na") else v["pass"]) for k, v in r["scenarios"].items()}), json.dumps(r["timing"]), flush=True)
except Exception:
    print("SCENARIOS_TRACEBACK", traceback.format_exc(), flush=True)
    FAILED = True
finally:
    app.close()
if FAILED:
    sys.exit(1)
