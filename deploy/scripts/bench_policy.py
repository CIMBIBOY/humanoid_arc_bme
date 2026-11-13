#!/usr/bin/env python3
"""Per-tick latency of the leg policy chain on this machine, for every inference backend found here.

    python3 deploy/scripts/bench_policy.py <policy dir or pack> [--engine policy.fp32.engine] [--calls 10000]
        [--backends ort-cpu,trt] [--threads 1,2] [--paced] [--cpu 3] [--rt-prio 80] [--out bench.json]

One tick = what the runtime does per 20 ms control period with this policy (real humanoid_arc_bme.onboard.runtime / S1 code, no copies):
  convert  humanoid_arc_bme.onboard.runtime.io.unitree.state_from_lowstate  (rt/lowstate-like object -> RobotState)
  obs      LowerBodyObsBuilder.build                  (history, projected gravity, scaling -> float32 [obs_dim])
  infer    backend.run                                (ONNX/TensorRT, batch 1)
  decode   LowerBodyObsBuilder.decode                 (raw-action clamp + default + scale * a)
  s1       SafetyLayer.filter_joints                  (position + rate clamp, watchdog; config/g1_limits.yaml)
  total    all of the above, timed as one span
Inputs are replayed from closedloop.npz when it is next to the model (sim states, 4 envs x 300 steps, cycled),
otherwise synthetic. Warm-up calls are run and discarded first. Times are perf_counter_ns, reported in microseconds.

--paced sleeps to a 50 Hz grid between ticks (as the runtime does): caches cool and DVFS clocks drop between calls,
which is the realistic case on a Jetson; back-to-back (default) is the optimistic case. Run both on the robot.
--cpu pins the process to one core; --rt-prio tries SCHED_FIFO (needs root or CAP_SYS_NICE; the JSON says if it
was applied). The JSON also records host facts (Jetson model, nvpmodel/L4T release when readable, versions).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backends  # noqa: E402
from packdata import DEFAULT_LIMITS, load_policy_dir, replay_inputs, synthetic_inputs  # noqa: E402

from humanoid_arc_bme.onboard.runtime.io.unitree import state_from_lowstate  # noqa: E402
from humanoid_arc_bme.onboard.runtime.obs_builders import LowerBodyObsBuilder  # noqa: E402
from humanoid_arc_bme.onboard.api.contract import JointTargets  # noqa: E402
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, load_limits  # noqa: E402

STAGES = ("convert", "obs", "infer", "decode", "s1", "total")
DT = 0.02


def _read(p, n=300):
    try:
        return Path(p).read_text(errors="replace")[:n].strip().replace("\x00", "")
    except OSError:
        return None


def host_facts() -> dict:
    f = {"platform": platform.platform(), "cpu_count": os.cpu_count(), **backends.versions()}
    cpu = _read("/proc/cpuinfo", 20000) or ""
    for line in cpu.splitlines():
        if line.lower().startswith(("model name", "cpu part")):
            f["cpu"] = line.split(":", 1)[1].strip()
            break
    f["jetson_model"] = _read("/proc/device-tree/model")
    f["l4t_release"] = _read("/etc/nv_tegra_release", 200)
    f["nvpmodel_status"] = _read("/var/lib/nvpmodel/status")  # best effort; `sudo nvpmodel -q` is authoritative
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    f["cpu0_governor"] = gov
    return f


def stats_us(ns: np.ndarray) -> dict:
    us = ns.astype(np.float64) / 1e3
    p = np.percentile(us, [50, 95, 99, 99.9])
    return {"p50": round(p[0], 2), "p95": round(p[1], 2), "p99": round(p[2], 2), "p99.9": round(p[3], 2),
            "max": round(float(us.max()), 2), "mean": round(float(us.mean()), 2)}


def apply_sched(cpu: int | None, rt_prio: int | None) -> dict:
    out = {"cpu_affinity": None, "sched_fifo": None}
    if cpu is not None:
        try:
            os.sched_setaffinity(0, {cpu})
            out["cpu_affinity"] = cpu
        except (OSError, AttributeError) as e:
            out["cpu_affinity"] = f"failed: {e}"
    if rt_prio:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(rt_prio))
            out["sched_fifo"] = rt_prio
        except (OSError, AttributeError) as e:
            out["sched_fifo"] = f"failed: {e}"
    return out


def bench_one(runner, raw_layout, lay, inputs, limits, calls: int, warmup: int, paced: bool) -> dict:
    builder = LowerBodyObsBuilder(lay)
    names = tuple(lay.action_joint_names)
    s1 = SafetyLayer(limits, Watchdog(0.1))
    t_ns = {k: np.empty(calls, np.int64) for k in STAGES}
    s1_events = 0
    pc = time.perf_counter_ns
    next_t = time.monotonic()
    n_in = len(inputs)
    for i in range(-warmup, calls):
        inp = inputs[i % n_in]
        if inp["done"]:
            builder.reset()
        if paced:
            next_t += DT
            d = next_t - time.monotonic()
            if d > 0:
                time.sleep(d)
        now = (i + warmup) * DT  # simulated stamps: S1 sees a steady 50 Hz stream, its watchdog never fires
        t0 = pc()
        st, _ = state_from_lowstate(inp["lowstate"], now)
        t1 = pc()
        obs = builder.build(st, inp["last_action"], inp["command"])
        t2 = pc()
        raw = runner.run(obs[None])[0]
        t3 = pc()
        _, targets = builder.decode(raw)
        cmd = JointTargets(names, tuple(float(x) for x in targets))
        t4 = pc()
        s1.command_received(now)
        _, ev = s1.filter_joints(now, DT, st, cmd)
        t5 = pc()
        if i >= 0:
            for k, a, b in (("convert", t0, t1), ("obs", t1, t2), ("infer", t2, t3), ("decode", t3, t4), ("s1", t4, t5),
                            ("total", t0, t5)):
                t_ns[k][i] = b - a
            s1_events += bool(ev)
    return {"stages_us": {k: stats_us(v) for k, v in t_ns.items()}, "s1_ticks_altered": s1_events,
            "worst_tick_index": int(np.argmax(t_ns["total"]))}


def determinism(runner, inputs, lay, reps=200) -> dict:
    """Same observation many times: bitwise identical outputs? (run-to-run, same process)."""
    builder = LowerBodyObsBuilder(lay)
    st, _ = state_from_lowstate(inputs[0]["lowstate"], 0.0)
    x = builder.build(st, inputs[0]["last_action"], inputs[0]["command"])[None]
    ref = runner.run(x).copy()
    diffs = [float(np.abs(runner.run(x) - ref).max()) for _ in range(reps)]
    return {"reps": reps, "bitwise_identical": all(d == 0.0 for d in diffs), "max_abs_diff": max(diffs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir")
    ap.add_argument("--engine", help="TensorRT engine from build_engine.sh (enables the trt backend)")
    ap.add_argument("--backends", help="comma list; default: every backend detected here")
    ap.add_argument("--threads", default="1", help="ORT intra-op thread counts to try, e.g. 1,2,4")
    ap.add_argument("--fp16", action="store_true", help="ort-trt: FP16 engine")
    ap.add_argument("--calls", type=int, default=10000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--paced", action="store_true", help="sleep to a 50 Hz grid between ticks")
    ap.add_argument("--cpu", type=int, help="pin to this core")
    ap.add_argument("--rt-prio", type=int, help="SCHED_FIFO priority (needs privileges)")
    ap.add_argument("--no-gc", action="store_true", help="gc.disable() during the timed loop")
    ap.add_argument("--limits", default=str(DEFAULT_LIMITS))
    ap.add_argument("--budget-us", type=float, default=2000.0, help="pass if total p99.9 and max are below this")
    ap.add_argument("--out", default="bench_policy.json")
    a = ap.parse_args(argv)

    raw_layout, lay, onnx_path = load_policy_dir(a.model_dir)
    npz = Path(a.model_dir) / "closedloop.npz"
    inputs = replay_inputs(npz, raw_layout["all_joint_names"]) if npz.exists() else synthetic_inputs(lay)
    limits = load_limits(a.limits)
    sched = apply_sched(a.cpu, a.rt_prio)
    found = backends.detect(a.engine)
    wanted = a.backends.split(",") if a.backends else found
    report = {"model_dir": str(a.model_dir), "onnx_sha256": backends.sha256(onnx_path),
              "inputs": "closedloop.npz replay" if npz.exists() else "synthetic", "calls": a.calls,
              "warmup": a.warmup, "paced_50hz": a.paced, "gc_disabled": a.no_gc, "sched": sched,
              "budget_us": a.budget_us, "host": host_facts(), "detected": found, "results": [], "skipped": {}}
    for name in wanted:
        for th in ([int(t) for t in a.threads.split(",")] if name.startswith("ort") else [1]):
            label = f"{name}/t{th}" if name.startswith("ort") else name
            try:
                t_init = time.perf_counter()
                r = backends.make_runner(name, onnx_path, engine=a.engine, threads=th, fp16=a.fp16)
                init_s = time.perf_counter() - t_init
            except Exception as e:  # noqa: BLE001 - a missing backend is a skip, recorded with its reason
                report["skipped"][label] = f"{type(e).__name__}: {e}"
                print(f"[bench] skip {label}: {e}")
                continue
            t_first = time.perf_counter()
            r.run(np.zeros([1, lay_obs(lay)], np.float32))
            first_s = time.perf_counter() - t_first
            if a.no_gc:
                gc.collect()
                gc.disable()
            try:
                res = bench_one(r, raw_layout, lay, inputs, limits, a.calls, a.warmup, a.paced)
            finally:
                gc.enable()
            res.update({"backend": label, "info": r.info, "session_init_s": round(init_s, 4),
                        "first_call_s": round(first_s, 4), "determinism": determinism(r, inputs, lay)})
            tot = res["stages_us"]["total"]
            res["within_budget"] = bool(tot["p99.9"] < a.budget_us and tot["max"] < a.budget_us)
            report["results"].append(res)
            s = res["stages_us"]
            print(f"[bench] {label:12s} total p50 {tot['p50']:8.1f} p99 {tot['p99']:8.1f} p99.9 {tot['p99.9']:8.1f} "
                  f"max {tot['max']:8.1f} us | infer p50 {s['infer']['p50']:.1f} obs {s['obs']['p50']:.1f} "
                  f"s1 {s['s1']['p50']:.1f} convert {s['convert']['p50']:.1f} | budget {'OK' if res['within_budget'] else 'MISS'}")
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[bench] wrote {a.out}")
    return report


def lay_obs(lay) -> int:
    return LowerBodyObsBuilder(lay).obs_dim


if __name__ == "__main__":
    main()
