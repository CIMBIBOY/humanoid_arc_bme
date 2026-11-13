#!/usr/bin/env python3
"""READ-ONLY camera probe: which cameras exist, their modes, intrinsics and real frame rate, plus one saved frame.

What it does: (1) RealSense through pyrealsense2: device info (model, serial, firmware, USB type), sensor stream
profiles, then a short stream (default 640x480 @ 30 depth + colour) to measure the frame rate from the camera's own
timestamps, read intrinsics, depth scale and depth->colour extrinsics, save one colour JPEG and one depth frame
(.npy uint16 + 16-bit PNG); optionally a 2 s IMU stream (D435i-type motion module). (2) Other V4L2 capture nodes
through OpenCV: open, grab `--frames`, report size / fourcc / measured fps, save a JPEG.

What it never does: write camera options or firmware, reset hardware, or move anything. If a camera is busy (the
vendor image service may hold it) that is reported, and nothing is stopped to free it.

Usage (on the Jetson):
    python3 camera_probe.py --i-understand-read-only [--out DIR] [--frames 30] [--no-realsense] [--no-v4l2]
                            [--no-imu] [--rs-width 640 --rs-height 480 --rs-fps 30] [--v4l2-max 8]
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import rate_stats, require_ack, utc_stamp, write_json  # noqa: E402


# ------------------------------------------------------------------ pure helpers (tested)
def list_v4l2_nodes(sysfs_root: str = "/sys/class/video4linux", dev_root: str = "/dev") -> list[dict]:
    out = []
    if not os.path.isdir(sysfs_root):
        return out
    for ent in sorted(os.listdir(sysfs_root), key=lambda s: (len(s), s)):
        if not ent.startswith("video"):
            continue
        base = os.path.join(sysfs_root, ent)

        def rd(name):
            try:
                with open(os.path.join(base, name)) as f:
                    return f.read().strip()
            except OSError:
                return None
        dev = os.path.join(base, "device")
        out.append({"node": os.path.join(dev_root, ent), "name": rd("name"), "index": rd("index"),
                    "device_path": os.path.realpath(dev) if os.path.exists(dev) else None})
    return out


def is_realsense_name(name: str | None) -> bool:
    return bool(name) and "realsense" in name.lower()


def intrinsics_to_dict(intr) -> dict:
    return {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "model": str(getattr(intr, "model", "")), "coeffs": [float(c) for c in getattr(intr, "coeffs", [])]}


def fps_from_hw_ms(stamps_ms: list[float]) -> float | None:
    """Frame rate from camera hardware/global timestamps in milliseconds."""
    if len(stamps_ms) < 2 or stamps_ms[-1] <= stamps_ms[0]:
        return None
    return round((len(stamps_ms) - 1) / ((stamps_ms[-1] - stamps_ms[0]) / 1e3), 3)


def write_ppm(path: str, rgb_rows: list[list[tuple[int, int, int]]]) -> str:
    """Last-resort image writer (no OpenCV, no PIL): binary PPM from rows of (r, g, b)."""
    h, w = len(rgb_rows), len(rgb_rows[0]) if rgb_rows else 0
    with open(path, "wb") as f:
        f.write(f"P6 {w} {h} 255\n".encode())
        for row in rgb_rows:
            f.write(bytes(v for px in row for v in px))
    return path


def save_bgr(path_noext: str, bgr) -> str | None:
    """Save an HxWx3 BGR array as JPEG (OpenCV, else PIL), else PPM. Returns the written path."""
    try:
        import cv2
        p = path_noext + ".jpg"
        if cv2.imwrite(p, bgr):
            return p
    except Exception:
        pass
    try:
        from PIL import Image
        p = path_noext + ".jpg"
        Image.fromarray(bgr[:, :, ::-1]).save(p, quality=90)
        return p
    except Exception:
        pass
    try:
        rows = [[(int(px[2]), int(px[1]), int(px[0])) for px in row] for row in bgr]
        return write_ppm(path_noext + ".ppm", rows)
    except Exception:
        return None


def depth_stats(depth_u16, scale_m: float) -> dict:
    import numpy as np
    d = np.asarray(depth_u16)
    valid = d > 0
    vals = d[valid].astype("float64") * scale_m
    h, w = d.shape[:2]
    return {"shape": [int(h), int(w)], "valid_fraction": round(float(valid.mean()), 4),
            "median_m": round(float(np.median(vals)), 4) if vals.size else None,
            "min_m": round(float(vals.min()), 4) if vals.size else None,
            "max_m": round(float(vals.max()), 4) if vals.size else None,
            "center_m": round(float(d[h // 2, w // 2]) * scale_m, 4)}


# ------------------------------------------------------------------ RealSense
def _info(rs, dev, key):
    ci = getattr(rs.camera_info, key)
    try:
        return dev.get_info(ci) if dev.supports(ci) else None
    except Exception:
        return None


def realsense_profiles(rs, dev, limit: int = 60) -> list[dict]:
    sensors = []
    for s in dev.query_sensors():
        name = _info(rs, s, "name")
        modes = set()
        for p in s.get_stream_profiles():
            try:
                if p.is_video_stream_profile():
                    vp = p.as_video_stream_profile()
                    modes.add((str(p.stream_type()), str(p.format()), vp.width(), vp.height(), p.fps()))
                else:
                    modes.add((str(p.stream_type()), str(p.format()), 0, 0, p.fps()))
            except Exception:
                continue
        ml = sorted(modes)
        sensors.append({"name": name, "mode_count": len(ml),
                        "modes": [f"{st} {fm} {w}x{h}@{fps}" if w else f"{st} {fm} @{fps}" for st, fm, w, h, fps in ml[:limit]]})
    return sensors


def realsense_stream(rs, serial: str, out_dir: str, frames: int, w: int, h: int, fps: int) -> dict:
    import numpy as np
    res: dict = {"requested": f"{w}x{h}@{fps}"}
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    try:
        prof = pipe.start(cfg)
    except Exception as e:
        res["requested_mode_error"] = str(e)[:200]
        cfg = rs.config()
        cfg.enable_device(serial)
        prof = pipe.start(cfg)  # device default streams
        res["fallback"] = "device default streams"
    try:
        dev = prof.get_device()
        try:
            res["depth_scale_m"] = float(dev.first_depth_sensor().get_depth_scale())
        except Exception:
            res["depth_scale_m"] = None
        streams = {}
        for sp in prof.get_streams():
            key = str(sp.stream_type())
            d = {"format": str(sp.format()), "fps": sp.fps()}
            try:
                d["intrinsics"] = intrinsics_to_dict(sp.as_video_stream_profile().get_intrinsics())
            except Exception:
                pass
            streams[key] = (sp, d)
        res["streams"] = {k: d for k, (_, d) in streams.items()}
        ds, cs = streams.get("stream.depth"), streams.get("stream.color")
        if ds and cs:
            ex = ds[0].get_extrinsics_to(cs[0])
            res["extrinsics_depth_to_color"] = {"rotation": list(ex.rotation), "translation_m": list(ex.translation)}
        arrivals, hw = {"color": [], "depth": []}, {"color": [], "depth": []}
        domains, last = {}, {}
        for _ in range(5):  # warm-up: auto exposure settles
            pipe.wait_for_frames(5000)
        for _ in range(frames):
            fs = pipe.wait_for_frames(5000)
            t = time.monotonic()
            for kind, fr in (("color", fs.get_color_frame()), ("depth", fs.get_depth_frame())):
                if not fr:
                    continue
                arrivals[kind].append(t)
                hw[kind].append(float(fr.get_timestamp()))
                domains[kind] = str(fr.get_frame_timestamp_domain())
                last[kind] = fr
        for kind in ("color", "depth"):
            if arrivals[kind]:
                res[f"{kind}_arrival"] = rate_stats(arrivals[kind])
                res[f"{kind}_hw_fps"] = fps_from_hw_ms(hw[kind])
                res[f"{kind}_timestamp_domain"] = domains.get(kind)
        tag = os.path.join(out_dir, f"realsense_{serial}")
        if "color" in last:
            res["color_saved"] = save_bgr(tag + "_color", np.asanyarray(last["color"].get_data()))
        if "depth" in last:
            depth = np.asanyarray(last["depth"].get_data()).copy()
            np.save(tag + "_depth.npy", depth)
            res["depth_saved"] = tag + "_depth.npy"
            try:
                import cv2
                cv2.imwrite(tag + "_depth16.png", depth)
                res["depth_png"] = tag + "_depth16.png"
            except Exception:
                pass
            if res.get("depth_scale_m"):
                res["depth_stats"] = depth_stats(depth, res["depth_scale_m"])
    finally:
        pipe.stop()
    return res


def realsense_imu(rs, serial: str, seconds: float = 2.0) -> dict:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.accel)
    cfg.enable_stream(rs.stream.gyro)
    times = {"accel": [], "gyro": []}
    sample = {}

    def cb(fr):
        try:
            k = "accel" if "accel" in str(fr.get_profile().stream_type()) else "gyro"
            times[k].append(time.monotonic())
            d = fr.as_motion_frame().get_motion_data()
            sample[k] = [d.x, d.y, d.z]
        except Exception:
            pass
    pipe.start(cfg, cb)
    try:
        time.sleep(seconds)
    finally:
        pipe.stop()
    return {k: {"rate": rate_stats(v, seconds), "last_xyz": sample.get(k)} for k, v in times.items()}


def probe_realsense(out_dir: str, frames: int, w: int, h: int, fps: int, imu: bool, rs=None) -> dict:
    if rs is None:
        try:
            import pyrealsense2 as rs
        except Exception as e:
            return {"available": False, "error": f"pyrealsense2 not importable: {e}"}
    out = {"available": True, "version": getattr(rs, "__version__", None), "devices": []}
    ctx = rs.context()
    for dev in ctx.query_devices():
        d = {k: _info(rs, dev, k) for k in ("name", "serial_number", "firmware_version", "product_line",
                                            "usb_type_descriptor", "physical_port", "product_id")}
        d["sensors"] = realsense_profiles(rs, dev)
        has_motion = any("Motion" in (s["name"] or "") or any("accel" in m for m in s["modes"]) for s in d["sensors"])
        d["has_motion_module"] = has_motion
        try:
            d["stream"] = realsense_stream(rs, d["serial_number"], out_dir, frames, w, h, fps)
        except Exception as e:
            d["stream_error"] = f"{type(e).__name__}: {e}"[:300]
        if imu and has_motion:
            try:
                d["imu"] = realsense_imu(rs, d["serial_number"])
            except Exception as e:
                d["imu_error"] = f"{type(e).__name__}: {e}"[:300]
        out["devices"].append(d)
    return out


# ------------------------------------------------------------------ V4L2 via OpenCV
def probe_v4l2(nodes: list[dict], out_dir: str, frames: int, skip_realsense: bool, max_nodes: int, cv2=None) -> list[dict]:
    if cv2 is None:
        try:
            import cv2
        except Exception as e:
            return [{"error": f"cv2 not importable: {e}"}]
    res = []
    for n in nodes:
        if skip_realsense and is_realsense_name(n.get("name")):
            res.append({**n, "skipped": "RealSense node (probed through pyrealsense2)"})
            continue
        if len([r for r in res if "opened" in r]) >= max_nodes:
            res.append({**n, "skipped": "max nodes reached"})
            continue
        r = dict(n)
        cap = cv2.VideoCapture(n["node"], getattr(cv2, "CAP_V4L2", 200))
        try:
            r["opened"] = bool(cap.isOpened())
            if not r["opened"]:
                r["note"] = "could not open (metadata node, busy, or no permission)"
                res.append(r)
                continue
            times, frame = [], None
            for _ in range(frames):
                ok, fr = cap.read()
                if ok:
                    times.append(time.monotonic())
                    frame = fr
            fourcc = int(cap.get(getattr(cv2, "CAP_PROP_FOURCC", 6)) or 0)
            r.update({
                "width": int(cap.get(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)) or 0),
                "height": int(cap.get(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)) or 0),
                "fps_reported": float(cap.get(getattr(cv2, "CAP_PROP_FPS", 5)) or 0),
                "fourcc": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)) if fourcc else None,
                "frames_read": len(times), "arrival": rate_stats(times),
            })
            if frame is not None:
                r["shape"] = list(getattr(frame, "shape", []))
                r["saved"] = save_bgr(os.path.join(out_dir, os.path.basename(n["node"])), frame)
        finally:
            cap.release()
        res.append(r)
    return res


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    require_ack(argv, "camera_probe.py")
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-understand-read-only", action="store_true")
    ap.add_argument("--out", default="./inventory_reports")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--no-realsense", action="store_true")
    ap.add_argument("--no-v4l2", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--rs-width", type=int, default=640)
    ap.add_argument("--rs-height", type=int, default=480)
    ap.add_argument("--rs-fps", type=int, default=30)
    ap.add_argument("--v4l2-max", type=int, default=8)
    ap.add_argument("--sysfs-root", default="/sys/class/video4linux", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    out_dir = os.path.join(a.out, f"camera_probe_{utc_stamp()}")
    os.makedirs(out_dir, exist_ok=True)
    report = {"probe": "camera_probe", "version": "1.0", "utc": utc_stamp(), "args": vars(a)}
    nodes = list_v4l2_nodes(a.sysfs_root)
    report["v4l2_nodes"] = nodes
    if not a.no_realsense:
        try:
            report["realsense"] = probe_realsense(out_dir, a.frames, a.rs_width, a.rs_height, a.rs_fps, not a.no_imu)
        except Exception as e:
            report["realsense"] = {"available": None, "error": f"{type(e).__name__}: {e}"[:300]}
    rs_found = bool(report.get("realsense", {}).get("devices"))
    if not a.no_v4l2:
        report["v4l2"] = probe_v4l2(nodes, out_dir, a.frames, skip_realsense=rs_found, max_nodes=a.v4l2_max)
    path = write_json(os.path.join(out_dir, "camera_probe.json"), report)
    rsd = report.get("realsense", {}).get("devices", [])
    print(f"[camera_probe] read-only; RealSense devices: {[(d.get('name'), d.get('serial_number')) for d in rsd] or 'none'}")
    for r in report.get("v4l2", []):
        if r.get("opened"):
            print(f"[camera_probe] {r['node']} {r.get('name')} {r.get('width')}x{r.get('height')} {r.get('fourcc')} measured {r['arrival'].get('mean_hz')} Hz")
    print(f"[camera_probe] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
