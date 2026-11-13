#!/usr/bin/env python3
"""Make a policy pack (one flat directory, see verify_policy_pack.py) from an export directory, then verify it.

    python3 deploy/scripts/make_policy_pack.py outputs/W4/onnx/deployE packs/deployE-e2 [--with-npz] [--engine <e>]
        [--note "text for the model card"]

Copies policy.onnx + layout.json (+ closedloop.npz with --with-npz, + an engine and its .json with --engine), writes
model-card.md (provenance from layout.json, git HEAD, limits file hash, parity if a recording is included) and
SHA256SUMS over every file. Refuses to write into a non-empty directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backends  # noqa: E402
from packdata import DEFAULT_LIMITS, ROOT  # noqa: E402
from verify_policy_pack import verify_pack  # noqa: E402


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def model_card(src: Path, lay: dict, onnx_sha: str, note: str, extra: list[str]) -> str:
    dirty = "dirty" if _git("status", "--porcelain") else "clean"
    lim_sha = backends.sha256(DEFAULT_LIMITS)
    lines = [
        f"# Policy pack: {src.name}",
        "",
        f"onnx_sha256: `{onnx_sha}`",
        "",
        "| field | value |",
        "|---|---|",
        f"| task | `{lay.get('task')}` |",
        f"| checkpoint | `{lay.get('checkpoint')}` |",
        f"| source export dir | `{src}` |",
        f"| packed | {time.strftime('%Y-%m-%d %H:%M %z')}, git `{_git('rev-parse', '--short', 'HEAD')}` ({dirty}) |",
        f"| input | `obs` float32 [1, {lay.get('obs_dim')}]: terms {lay.get('term_names')}, history {lay.get('hist')} (oldest first) |",
        f"| output | `actions` float32 [1, {len(lay['action_joint_names'])}] raw leg actions |",
        f"| decode | target = action_offset + {lay.get('action_scale')} * clamp(a, raw_lo, raw_hi) |",
        f"| clamp source | S1 position limits, `config/g1_limits.yaml` sha256 `{lim_sha[:16]}...` at pack time |",
        f"| base_lin_vel observed | {lay.get('lin_vel')} |",
        f"| control_dt | {lay.get('control_dt')} s (50 Hz) |",
        f"| trained latency DR | 0..{lay.get('max_delay')} control steps |",
        f"| PD gains the policy was trained with (kp / kd) | {lay.get('action_kp')} / {lay.get('action_kd')} |",
        "",
        "## Intended use and limits",
        "- Lower-body walking on the Unitree G1 (29-DoF body + Dex3), blind, velocity command (vx, vy, wz, height).",
        "- Sim-only evidence so far (docs/plan/logs/W4.md, HANDOFF_sim.md); S1 limits are placeholders "
        "(`validated_on_robot: false`) until B3.3. Not stair-safe going up (W5).",
        "- The robot's LowCmd kp/kd must equal the gains above (B4).",
    ]
    if note:
        lines += ["", "## Note", note]
    if extra:
        lines += ["", "## Checks at pack time", *extra]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--with-npz", action="store_true")
    ap.add_argument("--engine")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"[pack] {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    for n in ("policy.onnx", "layout.json"):
        shutil.copy2(src / n, out / n)
    extra = []
    if a.with_npz:
        shutil.copy2(src / "closedloop.npz", out / "closedloop.npz")
        import check_parity

        rep = check_parity.main([str(out), "--backends", "ort-cpu", "--out", "/dev/null"])
        r = rep["results"][0]
        extra.append(f"- ORT CPU {rep['versions'].get('onnxruntime')} vs torch on {rep['samples']} recorded steps: max |raw| "
                     f"{r['raw_vs_torch_max']:.2e}; builder obs vs sim obs {rep['builder_max_abs_obs_diff']:.2e}")
    if a.engine:
        e = Path(a.engine)
        shutil.copy2(e, out / e.name)
        shutil.copy2(str(e) + ".json", out / (e.name + ".json"))
    lay = json.loads((out / "layout.json").read_text())
    onnx_sha = backends.sha256(out / "policy.onnx")
    (out / "model-card.md").write_text(model_card(src, lay, onnx_sha, a.note, extra))
    files = sorted(p.name for p in out.iterdir() if p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{backends.sha256(out / n)}  {n}\n" for n in files))
    errs, _ = verify_pack(out)
    if errs:
        print("[pack] written but REFUSED by the verifier:\n  - " + "\n  - ".join(errs))
        return 1
    print(f"[pack] {out}: {len(files)} files + SHA256SUMS, verified OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
