"""LiveSim: what the web page shows and what it may ask for, between the simulator's main thread and the HTTP threads.

The simulator steps in the main thread (Kit requires it). After every outer step `LiveSim.after_step()` runs there: it executes the
requests queued by the HTTP side (arm, stop, reset, map goal, parser mode, fault tests) and publishes an immutable snapshot (state
at 20 Hz, images and the lidar cloud at the sensor rate). HTTP threads only read the newest snapshot and queue requests; the two
exceptions are the ones the robot's own software is built for: the operator heartbeat (`OperatorRuntime.heartbeat`, hold-to-run)
and the command gateway (`StackCommandService` under the autonomy loop's lock), which are called from HTTP threads there too.
"""

from __future__ import annotations

import io
import math
import queue
import threading
import time

import numpy as np

# height bands of the lidar view: (upper bound in metres above the floor, name). Few colours on purpose: floor, low, body height, overhead.
BANDS = ((0.12, "floor"), (0.5, "low"), (1.3, "body"), (99.0, "overhead"))
CEILING_CUT = 2.0            # metres: returns above this are not drawn (the ceiling would cover every wall in the top view)
VOXEL = 0.10                 # metres: accumulated map resolution
MAP_MAX = 60000              # voxels kept
# depth colour scale (near -> far), shared with the page's legend
DEPTH_STOPS = np.array([[255, 245, 200], [253, 174, 97], [215, 48, 39], [118, 42, 131], [20, 20, 60]], dtype=np.float32)
DEPTH_RANGE = {"d435": (0.3, 3.0), "head_rgbd": (0.3, 6.0)}
MASK_COLOURS = ((230, 97, 1), (94, 60, 153), (27, 158, 119), (231, 41, 138), (102, 166, 30), (117, 112, 179), (230, 171, 2), (166, 118, 29))


def colour_depth(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Depth (m) -> RGB uint8 with DEPTH_STOPS; invalid pixels dark grey, beyond `hi` the far colour."""
    d = np.asarray(depth, np.float32)
    ok = np.isfinite(d) & (d > 0)
    t = np.clip((np.where(ok, d, lo) - lo) / (hi - lo), 0.0, 1.0) * (len(DEPTH_STOPS) - 1)
    i = np.minimum(t.astype(np.int32), len(DEPTH_STOPS) - 2)
    f = (t - i)[..., None]
    rgb = DEPTH_STOPS[i] * (1.0 - f) + DEPTH_STOPS[i + 1] * f
    rgb[~ok] = (45, 45, 48)
    return rgb.astype(np.uint8)


def overlay_detections(rgb: np.ndarray, dets: list) -> np.ndarray:
    """RGB with each detection's mask tinted, outlined by its box and labelled. `dets`: [(label, confidence, mask | None, box | None)]."""
    from PIL import Image, ImageDraw

    out = rgb.astype(np.float32)
    for k, (_, _, mask, _) in enumerate(dets):
        if mask is not None and mask.shape == rgb.shape[:2]:
            c = np.array(MASK_COLOURS[k % len(MASK_COLOURS)], np.float32)
            out[mask] = 0.45 * out[mask] + 0.55 * c
    img = Image.fromarray(out.astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for k, (label, conf, _, box) in enumerate(dets):
        if box is None:
            continue
        c = MASK_COLOURS[k % len(MASK_COLOURS)]
        x0, y0, x1, y1 = box
        dr.rectangle([x0, y0, x1 - 1, y1 - 1], outline=c, width=2)
        text = f"{label} {conf:.0%}"
        tw = dr.textlength(text)
        ty = max(0, y0 - 12)
        dr.rectangle([x0, ty, x0 + tw + 6, ty + 12], fill=c)
        dr.text((x0 + 3, ty), text, fill=(255, 255, 255))
    return np.asarray(img)


def jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class RecordingSemantics:
    """The stack's detector, unchanged, with its last input and output kept for display."""

    def __init__(self, inner, name: str):
        self.inner, self.name = inner, name
        self.last = None                    # (rgb, [(label, conf, mask, box)], wall ms)

    def detect(self, rgb, depth):
        a = time.perf_counter()
        dets = self.inner.detect(rgb, depth)
        self.last = (rgb, [(d.label, d.confidence, d.mask, d.box) for d in dets], (time.perf_counter() - a) * 1e3)
        return dets


class LiveSim:
    def __init__(self, bench, detector: RecordingSemantics, head_detector=None, jev_factory=None):
        self.b, self.det, self.head_det = bench, detector, head_detector
        self.ses, self.sim = bench.ses, bench.sim
        self.ses.external_heartbeat = True
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.snap: dict = {"frame": -1, "state": {"ready": False}, "images": {}, "cloud": None}
        self._jpeg: dict = {}
        self._frame = 0
        self._last_stamp = -1.0
        self._map: dict = {}
        self._map_flat = None
        self._trail: list = []
        self._goal = None
        self.parser_mode = "grammar"
        self._jev = None
        self.jev_factory = jev_factory
        self.faults: dict = {}
        self._head_last = None
        self._wall0, self._sim0 = time.perf_counter(), self.sim.t
        self._rtf = 1.0
        self.plan = self._plan()
        if head_detector is not None:
            threading.Thread(target=self._head_loop, name="head-detector", daemon=True).start()

    # ------------------------------------------------------------------ HTTP side
    def call(self, request: str, /, timeout: float = 3.0, **kw) -> dict:
        """Queue a request for the main thread and wait for its answer (`request` is positional-only: a request's own parameter may be `name`)."""
        done, box = threading.Event(), {}
        self._q.put((request, kw, done, box))
        if not done.wait(timeout):
            box["cancel"] = True                                # a request the page was told failed must not run later (STOP excepted)
            return {"ok": False, "error": "the simulator did not answer in time (paused or busy)"}
        return box.get("r", {"ok": False, "error": "no result"})

    def map_cells(self) -> list:
        """Accumulated lidar map, cm integers [x, y, band, ...] (what the robot has seen with its estimated pose)."""
        return self._map_flat or []

    def image(self, name: str) -> tuple[int, bytes] | None:
        """(frame number, JPEG) of an image of the newest snapshot; encoded once per frame, in the caller's thread."""
        with self._lock:
            snap = self.snap
            hit = self._jpeg.get(name)
            if hit is not None and hit[0] == snap["frame"]:
                return hit
        src = snap["images"].get(name)
        if src is None:
            return None
        data = jpeg(src() if callable(src) else src)
        with self._lock:
            self._jpeg[name] = (snap["frame"], data)
        return snap["frame"], data

    # ------------------------------------------------------------------ simulator side (main thread)
    def after_step(self) -> None:
        while True:
            try:
                name, kw, done, box = self._q.get_nowait()
            except queue.Empty:
                break
            if box.get("cancel") and name != "stop":
                continue
            try:
                box["r"] = getattr(self, "_do_" + name)(**kw)
            except Exception as exc:  # noqa: BLE001 - a failed request is answered, the simulator goes on
                box["r"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
            done.set()
        self._publish()

    def _pose(self):
        e = self.ses.localizer.estimate(self.sim.t)
        if e is None:
            return None
        return {"x": float(e.pose.position[0]), "y": float(e.pose.position[1]), "yaw": float(e.pose.yaw()), "healthy": bool(e.healthy), "source": e.source}

    def _state(self) -> dict:
        ses, rt, b = self.ses, self.ses.rt, self.b
        st, sup = ses.stack.executor.status, rt.sup
        tr = b.truth()
        now_w = time.perf_counter()
        if now_w - self._wall0 >= 1.0:
            self._rtf = (self.sim.t - self._sim0) / (now_w - self._wall0)
            self._wall0, self._sim0 = now_w, self.sim.t
        scene = [{"label": o.label, "color": o.color, "range_m": round(o.range_m, 2), "confidence": round(o.confidence, 2),
                  "x": round(o.position_body[0], 2), "y": round(o.position_body[1], 2), "z": round(o.position_body[2], 2)} for o in ses.stack.scene(self.sim.t)]
        parser = ses.stack.parser
        nav = rt.nav.status() if rt.nav is not None else {}
        return {"ready": True, "t": round(self.sim.t, 2), "rtf": round(self._rtf, 2), "frame": self._frame,
                "armed": bool(sup.armed), "held": bool(sup.held), "supervisor": sup.state, "stop_latched": bool(sup.stop_latched), "mode": rt.mode, "owner": rt.owner,
                "fault": rt.fault, "cmd": [round(float(v), 3) for v in rt.last_cmd], "target": [round(float(v), 3) for v in sup.target],
                "limits": [float(v) for v in rt.safety.limits.base_vel_max],
                "safety": {"level": ses.stack.safety, "reasons": list(ses.stack.reasons)},
                "task": {"state": st.state, "phase": st.phase, "message": st.message, "command": st.command},
                "pose": self._pose(), "truth": {"x": tr["pos"][0], "y": tr["pos"][1], "yaw": tr["yaw"], "fallen": tr["fallen"], "tilt_deg": round(tr["tilt_deg"], 1)},
                "scene": scene, "goal": self._goal, "nav": {k: nav.get(k) for k in ("state", "active", "reason")},
                "parser": {"mode": self.parser_mode, "source": getattr(parser, "last_source", "local"), "error": getattr(parser, "last_error", ""),
                           "jev_calls": len(self._jev.calls) if self._jev is not None else 0},
                "detector": {"name": self.det.name, "ms": None if self.det.last is None else round(self.det.last[2], 1)},
                "teleop_guard": (rt.obstacle_info or {}).get("warning"), "faults": dict(self.faults), "motors_passive": self.b.robot.passive,
                "deploy": {"running": self.b.deploy.running, "mode": self.b.deploy.status()["mode"]}, "autonomy_error": ses.loop.last_error,
                "guard_profile": b.guard}

    def _publish(self) -> None:
        rig, state = self.b.rig, self._state()
        d = rig.depth("d435")
        images, cloud = self.snap["images"], self.snap["cloud"]
        new_frame = d is not None and d.stamp != self._last_stamp
        if new_frame:
            self._last_stamp = d.stamp
            self._frame += 1
            images = {}
            for cam, key in (("d435", "d435"), ("head_rgbd", "head")):
                rgb, dep = rig.rgb(cam), rig.depth(cam)
                if rgb is not None:
                    images[f"{key}_rgb"] = rgb
                if dep is not None:
                    lo, hi = DEPTH_RANGE[cam]
                    images[f"{key}_depth"] = (lambda dd=dep.depth, lo=lo, hi=hi: colour_depth(dd, lo, hi))
            if self.det.last is not None:
                rgb, dets, _ = self.det.last
                images["d435_sem"] = (lambda r=rgb, ds=dets: overlay_detections(r, ds))
                state["detections"] = {"d435": [{"label": l, "confidence": round(float(c), 2)} for l, c, _, _ in dets]}
            if self._head_last is not None:
                rgb, dets = self._head_last
                images["head_sem"] = (lambda r=rgb, ds=dets: overlay_detections(r, ds))
                state.setdefault("detections", {})["head"] = [{"label": l, "confidence": round(float(c), 2)} for l, c, _, _ in dets]
            cloud = self._cloud(state["pose"])
        elif "detections" in self.snap["state"]:
            state["detections"] = self.snap["state"]["detections"]
        state["frame"] = self._frame
        with self._lock:
            self.snap = {"frame": self._frame, "state": state, "images": images, "cloud": cloud,
                         "head_input": (rig.rgb("head_rgbd"), rig.depth("head_rgbd")) if new_frame else self.snap.get("head_input")}

    def _cloud(self, pose) -> dict | None:
        """The newest lidar frame placed in the map with the ESTIMATED pose (what the robot believes), banded by height, plus the
        accumulated map. Integers in centimetres: [x, y, band, x, y, band, ...]."""
        lf = self.b.rig.lidar_frame()
        if lf is None or pose is None:
            return self.snap["cloud"]
        p = lf.points
        if len(p) > 3500:
            p = p[:: len(p) // 3500 + 1]
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        wx, wy, z = pose["x"] + c * p[:, 0] - s * p[:, 1], pose["y"] + s * p[:, 0] + c * p[:, 1], p[:, 2]
        keep = (z > -0.3) & (z < CEILING_CUT) & (np.hypot(p[:, 0], p[:, 1]) < 12.0)       # the ceiling is not an obstacle for a 1.3 m robot
        wx, wy, z = wx[keep], wy[keep], z[keep]
        band = np.searchsorted(np.array([b[0] for b in BANDS]), z)
        scan = np.stack([np.round(wx * 100), np.round(wy * 100), band], axis=1).astype(np.int32)
        if pose["healthy"]:
            vox = np.stack([np.floor(wx / VOXEL), np.floor(wy / VOXEL), band], axis=1).astype(np.int32)
            for v in map(tuple, np.unique(vox[band > 0], axis=0)):      # the floor is not mapped: it would hide everything else
                self._map[v] = self._map.get(v, 0) + 1
            if len(self._map) > MAP_MAX:
                self._map = dict(sorted(self._map.items(), key=lambda kv: -kv[1])[: MAP_MAX // 2])
            self._trail.append((round(pose["x"], 2), round(pose["y"], 2)))
            self._trail = self._trail[-600:]
        if self._frame % 10 == 0 or self._map_flat is None:           # the map changes slowly: rebuilt twice a second at most
            top = sorted(((n, k) for k, n in self._map.items() if n >= 2), reverse=True)[:12000]
            self._map_flat = np.array([(int((k[0] + 0.5) * VOXEL * 100), int((k[1] + 0.5) * VOXEL * 100), k[2]) for _, k in top], dtype=np.int32).ravel().tolist()
        return {"frame": self._frame, "scan": scan.ravel().tolist(), "trail": [v for xy in self._trail[::3] for v in xy], "points_in_frame": int(len(lf.points)),
                "voxels": len(self._map)}

    def _plan(self) -> dict:
        """The scene's floor plan from the manifest (ground truth, drawn faintly on request): obstacle rectangles and rooms."""
        from humanoid_arc_bme.isaac import scene as SC

        man = self.b.man
        return {"scene": man["scene_id"], "rects": [[round(v, 3) for v in r] for r in SC.obstacles_2d(man)],
                "rooms": [{"id": r["id"], "type": r.get("type"), "rect": r["rect"]} for r in man["rooms"]],
                "doors": [{"id": o["id"], "x": o["pose"][0], "y": o["pose"][1], "state": o.get("door", {}).get("state")} for o in man["objects"] if o["class"] == "door"],
                "home": list(self.b.home), "bands": [{"name": n, "below_m": u} for u, n in BANDS],
                "depth": {"stops": DEPTH_STOPS.astype(int).tolist(), "range": {"d435": DEPTH_RANGE["d435"], "head": DEPTH_RANGE["head_rgbd"]}}}

    def _head_loop(self) -> None:
        """The head camera through the same detector class, for display only (the stack perceives with the down camera)."""
        seen = None
        while True:
            time.sleep(0.4)
            item = self.snap.get("head_input")
            if not item or item[0] is None or item[1] is None or item[0] is seen:
                continue
            seen = item[0]
            try:
                dets = self.head_det.detect(item[0], item[1])
                self._head_last = (item[0], [(d.label, d.confidence, d.mask, d.box) for d in dets])
            except Exception:  # noqa: BLE001 - display only
                pass

    # ------------------------------------------------------------------ requests (main thread)
    def _do_arm(self, client: str = "local") -> dict:
        rt = self.ses.rt
        if rt.fault or rt.sup.stop_latched:
            rt.reset_stop()
        code = rt.request_arm()["code"]
        ok, why = rt.confirm_arm(f"ARM {code}", client=client)
        return {"ok": bool(ok), "reason": why}

    def _do_disarm(self) -> dict:
        self.ses.stack.stop("operator disarm")
        return {"ok": bool(self.ses.rt.disarm("operator"))}

    def _do_stop(self) -> dict:
        self._goal = None
        self.ses.stack.stop("operator stop")
        return {"ok": True, "confirmed": bool(self.ses.rt.stop("ui"))}

    def _do_reset(self) -> dict:
        self._goal, self.faults = None, {}
        r = self.b.reset()
        self.ses.rt.disarm("reset")                              # a reset never leaves the robot armed: the operator arms
        self._apply_parser()
        return {"ok": True, **r, "armed": False}

    def _do_goal(self, x: float, y: float) -> dict:
        rt, pose = self.ses.rt, self._pose()
        if rt.nav is None or pose is None:
            return {"ok": False, "error": "no navigation module or no pose"}
        if not (math.isfinite(x) and math.isfinite(y)) or math.hypot(x - pose["x"], y - pose["y"]) > 6.0:
            return {"ok": False, "error": "goal must be within 6 m of the robot"}
        c, s = math.cos(-pose["yaw"]), math.sin(-pose["yaw"])
        dx, dy = x - pose["x"], y - pose["y"]
        with rt.lock:
            rt.nav.set_waypoints([(c * dx - s * dy, s * dx + c * dy)])       # the runtime's waypoints are relative to the robot at the start
        self._goal = {"x": x, "y": y}
        return {"ok": True, "goal": self._goal}

    def _do_parser(self, mode: str) -> dict:
        if mode not in ("grammar", "jev", "jev_outage"):
            return {"ok": False, "error": "mode: grammar | jev | jev_outage"}
        self.parser_mode = mode
        self._apply_parser()
        return {"ok": True, "mode": mode}

    def _apply_parser(self) -> None:
        from humanoid_arc_bme.onboard.command.composite import CompositeParser

        if self.parser_mode == "grammar" or self.jev_factory is None:
            self._jev, self.ses.stack.parser = None, self.b.grammar
        else:
            self._jev = self.jev_factory("timeout" if self.parser_mode == "jev_outage" else "ok")
            self.ses.stack.parser = CompositeParser(self.b.grammar, classifier=self._jev)

    def _do_fault(self, name: str, on: bool) -> dict:
        from humanoid_arc_bme.onboard.runtime.groot_controller import build_command_message

        b = self.b
        table = {"depth_dropout": (b.rig.faults, "d435_dropout"), "stale_frames": (b.rig.faults, "d435_stale"), "localizer_lost": (self.ses.localizer.faults, "lost"),
                 "lowstate_lost": (self.ses.io.faults, "state_lost")}
        if name == "controller_exit":
            if on:
                b.deploy.on_message(build_command_message(False, True, True), self.sim.t)
            self.faults[name] = bool(on)
            return {"ok": True, "note": "the controller process is gone; use Reset robot to start it again"}
        if name not in table:
            return {"ok": False, "error": "unknown fault"}
        target, key = table[name]
        if on:
            target[key] = True
        else:
            target.pop(key, None)
            target.pop("_frozen", None)
        self.faults[name] = bool(on)
        return {"ok": True}
