"""Build the tarball that goes onto the Jetson: humanoid_arc_bme.onboard.runtime (+ web UI), the stdlib-only contract, config, the Jetson tooling
(humanoid_arc_bme.onboard.deploy), the ROS 2 wrapper (ros2/, its symlinks kept as symlinks), deploy/scripts and the fake-IO tests.
No Isaac, no torch, no training code. (An alternative to `git clone` when the robot has no access to the repository.)

    python -m humanoid_arc_bme.onboard.runtime.bundle out.tar.gz          # writes out.tar.gz and out.tar.gz.manifest.json (sha256 per file, git HEAD, dirty flag)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INCLUDE = ["humanoid_arc_bme/__init__.py", "humanoid_arc_bme/onboard", "config", "deploy", "tests/__init__.py", "tests/conftest.py", "tests/onboard"]


def _walk(p: Path) -> list[Path]:
    """Files and symlinks under p; symlinked directories are NOT followed (deploy/ros2/g1_runtime_ros links to the core)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(p, followlinks=False):
        d = Path(dirpath)
        dirnames[:] = [n for n in dirnames if n != "__pycache__"]
        out += [d / n for n in dirnames if (d / n).is_symlink()]
        out += [d / n for n in filenames if not n.endswith(".pyc")]
    return sorted(out)


def files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += _walk(p)
    return out


def git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build(out: str) -> dict:
    fs = files()
    manifest = {"git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "files": {str(f.relative_to(ROOT)): ("symlink:" + os.readlink(f)) if f.is_symlink() else hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in fs}}
    with tarfile.open(out, "w:gz") as tar:
        for f in fs:
            tar.add(f, arcname=str(f.relative_to(ROOT)), recursive=False)
    Path(out + ".manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    m = build(sys.argv[1])
    print(f"{len(m['files'])} files, git {m['git_head'][:10]} dirty={m['git_dirty']}")
