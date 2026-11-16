"""Load a scene into the open Isaac stage from its manifest. Needs Kit.

A scene is `scene.json` (schema in the producer's repository; this package reads only the fields used below) next to the USD file
it names. Nothing here knows how the scene was made: a generated room and a reconstructed one load the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

# manifest role / class -> the object word the stack's grammar and detector use. GUESS where the two vocabularies do not meet.
LABEL_OF = {"mug": "cup", "bottle": "bottle", "can": "can", "bowl_plate": "bowl", "book": "book", "box": "box", "cookware": "pot",
            "table": "table", "desk": "table", "cabinet": "cabinet", "chair": "chair", "sofa": "sofa", "door": "door", "appliance": "appliance"}


def load_scene(scene_dir, root: str = "/World/Scene", labels: bool = True) -> dict:
    """Reference the scene under `root` (its own PhysicsScene is switched off: the World's is used) and return the manifest.
    With `labels`, every manifest object gets a semantic label `<word>#<object id>` so a segmentation render tells objects apart."""
    import omni.usd

    scene_dir = Path(scene_dir).resolve()
    man = json.loads((scene_dir / "scene.json").read_text())
    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim(root).GetReferences().AddReference(str(scene_dir / man["usd"]))
    dup = stage.GetPrimAtPath(f"{root}/PhysicsScene")
    if dup:
        dup.SetActive(False)
    if labels:
        from pxr import Sdf

        for o in man["objects"]:
            word = LABEL_OF.get(o.get("role") or o["class"])
            prim = stage.GetPrimAtPath(root + o["prim_path"][len("/World"):])
            if word and prim:                                           # Isaac 5.0 labels, authored by schema name (its helper fails here)
                prim.AddAppliedSchema("SemanticsLabelsAPI:class")
                prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray).Set([f"{word}#{o['id']}"])
    return man


def spawn_pose(man: dict, spawn_id: str | None = None) -> tuple[float, float, float]:
    """(x, y, yaw) of a spawn pose of the manifest (the first one by default)."""
    sp = next((s for s in man["spawn_poses"] if spawn_id in (None, s["id"])), None)
    if sp is None:
        raise ValueError("the scene has no such spawn pose")
    return tuple(float(v) for v in sp["floor_pose"])


# ------------------------------------------------------------------ plan-view geometry from the manifest (scoring and test set-up)
def obstacles_2d(man: dict) -> list[tuple]:
    """Floor-blocking obstacles as oriented rectangles (cx, cy, half_x, half_y, yaw): wall pieces (door openings left out, windows
    kept) and every object that stands on the floor. Clutter on furniture and doors (they swing) are not included."""
    import math

    out = []
    for w in man["structure"]["walls"]:
        (ax, ay), (bx, by) = w["a"], w["b"]
        length, yaw = math.dist(w["a"], w["b"]), math.atan2(by - ay, bx - ax)
        cuts = sorted((o["offset"] - o["width"] / 2, o["offset"] + o["width"] / 2) for o in man["structure"]["openings"] if o["wall"] == w["id"] and o["kind"] == "door")
        s0 = 0.0
        for c0, c1 in cuts + [(length, length)]:
            if c0 - s0 > 1e-3:
                m = (s0 + c0) / 2
                out.append((ax + math.cos(yaw) * m, ay + math.sin(yaw) * m, (c0 - s0) / 2, w["thickness"] / 2, yaw))
            s0 = c1
    for o in man["objects"]:
        if o.get("support") or o["class"] in ("door", "window"):
            continue
        (x0, y0, _), (x1, y1, _) = o["aabb"]
        qw, qz = o["pose"][3], o["pose"][6]
        yaw = 2.0 * math.atan2(qz, qw)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        out.append((o["pose"][0] + math.cos(yaw) * cx - math.sin(yaw) * cy, o["pose"][1] + math.sin(yaw) * cx + math.cos(yaw) * cy,
                    (x1 - x0) / 2, (y1 - y0) / 2, yaw))                # aabb_rest is already scaled by the physicalizer
    return out


def clearance(obstacles: list[tuple], x: float, y: float) -> float:
    """Distance from a point to the nearest obstacle rectangle (0 inside one)."""
    import math

    best = math.inf
    for cx, cy, hx, hy, yaw in obstacles:
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * (x - cx) - s * (y - cy), s * (x - cx) + c * (y - cy)
        best = min(best, math.hypot(max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0)))
    return best


def free_run(obstacles: list[tuple], x: float, y: float, yaw: float, radius: float, max_m: float = 6.0, step: float = 0.05) -> float:
    """How far a disc of `radius` can travel from (x, y) along `yaw` before it touches an obstacle."""
    import math

    d = 0.0
    while d < max_m and clearance(obstacles, x + math.cos(yaw) * d, y + math.sin(yaw) * d) > radius:
        d += step
    return d
