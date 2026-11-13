"""Checkpoint manifest: what was trained, from what, and the sha256 of every file (A5.5, B3.5)."""

import hashlib
import json
import os
import subprocess
import time

MANIFEST_NAME = "manifest.json"
SUMS_NAME = "SHA256SUMS"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _files(directory):
    out = []
    for root, _, names in os.walk(directory):
        for n in sorted(names):
            if n in (MANIFEST_NAME, SUMS_NAME):
                continue
            full = os.path.join(root, n)
            out.append(os.path.relpath(full, directory))
    return sorted(out)


def git_commit(repo="."):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def build(directory, *, seed, config_path=None, dataset_path=None, notes="", repo="."):
    """Write manifest.json and SHA256SUMS into `directory`; return the manifest dict.

    seed is required (no default): a checkpoint with no seed cannot be reproduced.
    """
    files = _files(directory)
    if not files:
        raise ValueError(f"{directory}: no files to describe")
    m = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(repo),
        "seed": seed,
        "config_sha256": sha256_file(config_path) if config_path else None,
        "dataset_sha256": sha256_file(dataset_path) if dataset_path else None,
        "notes": notes,
        "files": {rel: sha256_file(os.path.join(directory, rel)) for rel in files},
    }
    with open(os.path.join(directory, MANIFEST_NAME), "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
    with open(os.path.join(directory, SUMS_NAME), "w") as f:
        for rel, digest in sorted(m["files"].items()):
            f.write(f"{digest}  {rel}\n")
    return m


def verify(directory):
    """Return a list of problems (empty = every listed file present with the recorded hash, no extras)."""
    path = os.path.join(directory, MANIFEST_NAME)
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        return [f"cannot read {path}: {e}"]
    problems = []
    for rel, digest in m.get("files", {}).items():
        full = os.path.join(directory, rel)
        if not os.path.isfile(full):
            problems.append(f"missing: {rel}")
        elif sha256_file(full) != digest:
            problems.append(f"hash differs: {rel}")
    extra = set(_files(directory)) - set(m.get("files", {}))
    problems += [f"not in manifest: {rel}" for rel in sorted(extra)]
    if m.get("seed") is None:
        problems.append("manifest has no seed")
    return problems
