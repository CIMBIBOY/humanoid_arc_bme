#!/usr/bin/env python3
"""Assemble a model pack (flat directory: files, SHA256SUMS, model-card.md) from an export directory, then verify it.

    python3 scripts/export/make_model_pack.py <export_dir> <pack_dir> [--note "text"] [--limits config/g1_limits.yaml]
        [--repo <git work tree, for the HEAD line of the card>]

<export_dir> is an export_policy_onnx.py output (policy.onnx, closedloop.npz, export.json, optional layout.json) or an
export_yolo_onnx.py output (model.onnx, model.json). Nothing is written unless the export passes check_onnx (policy) or
validate_yolo_seg_onnx (YOLO). A policy pack also records the sha256 of the safety limits file (--limits). <pack_dir>
must not exist or must be empty. The finished pack is verified by re-hashing every file against SHA256SUMS.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "scripts"))

import backends  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_onnx import check_policy_pack_dir  # noqa: E402
from export_yolo_onnx import LICENCE_NOTE, validate_yolo_seg_onnx  # noqa: E402

LIBRARY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIMITS = LIBRARY_ROOT / "config" / "g1_limits.yaml"
POLICY_FILES = ("policy.onnx", "closedloop.npz", "export.json")
POLICY_OPTIONAL = ("layout.json",)
YOLO_FILES = ("model.onnx", "model.json")
CARD = "model-card.md"
SUMS = "SHA256SUMS"
_SUM_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")
_CARD_HASH = re.compile(r"onnx_sha256:\s*`?([0-9a-f]{64})")


def _kind(export_dir: Path) -> str:
    if all((export_dir / n).is_file() for n in POLICY_FILES):
        return "policy"
    if all((export_dir / n).is_file() for n in YOLO_FILES):
        return "yolo"
    raise ValueError(f"{export_dir} is neither a policy export ({', '.join(POLICY_FILES)}) nor a YOLO export "
                     f"({', '.join(YOLO_FILES)})")


def _git_head(repo: Path | None) -> str:
    """'<short sha> (clean|dirty)', or a reason it is unknown. No repo given means no git query at all."""
    if repo is None:
        return "not recorded"
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return f"{head} ({'dirty' if dirty else 'clean'})"


def _policy_card(name: str, export_dir: Path, report: dict[str, Any], info: dict[str, Any], limits: Path,
                 git: str, note: str) -> str:
    sig = report["signature"]
    norm = info.get("normalizer")
    lines = [
        f"# Policy model pack: {name}",
        "",
        f"onnx_sha256: `{report['onnx_sha256']}`",
        "",
        "| field | value |",
        "|---|---|",
        f"| input | `{sig['input']['name']}` {sig['input']['type']} {sig['input']['shape']}: raw observations"
        + (f", normaliser `{norm['source']}` baked in (eps {norm['eps']})" if norm else ", no normaliser") + " |",
        f"| output | `{sig['output']['name']}` {sig['output']['type']} {sig['output']['shape']}: actions |",
        f"| architecture | hidden {info['hidden']}, activation {info['activation']}, opset {info['opset']} |",
        f"| source checkpoint | `{info.get('checkpoint')}` sha256 `{info.get('checkpoint_sha256')}` |",
        f"| exported with | torch {info['torch']}, onnx {info['onnx']} |",
        f"| limits file | `{limits.name}` sha256 `{backends.sha256(limits)}` |",
        f"| git HEAD | {git} |",
        f"| source export | `{export_dir}` |",
        f"| packed | {time.strftime('%Y-%m-%d %H:%M %z')} |",
        "",
        "## Checks at pack time (onnxruntime, CPU)",
        f"- parity: max |onnx - torch| {report['max_abs_diff']:.2e} over {report['samples']} closed-loop samples "
        f"(atol {report['atol']:.0e}): {'PASS' if report['passed'] else 'FAIL'}",
        f"- latency, batch 1: median {report['median_ms']:.3f} ms, p99 {report['p99_ms']:.3f} ms "
        f"over {report['runs']} runs",
        "",
        "## Limits",
        "- Verified on CPU only here: no TensorRT engine built, not run on the robot.",
        "- Acceptance on hardware is a separate step.",
    ]
    if note:
        lines += ["", "## Note", note]
    return "\n".join(lines) + "\n"


def _yolo_card(name: str, export_dir: Path, facts: dict[str, Any], info: dict[str, Any], git: str, note: str) -> str:
    lines = [
        f"# YOLO segmentation model pack: {name}",
        "",
        f"onnx_sha256: `{info['sha256']}`",
        "",
        "| field | value |",
        "|---|---|",
        f"| input | `{facts['input']['name']}` {facts['input']['type']} {facts['input']['shape']} "
        f"(RGB, 0-1 scaled by the caller) |",
        "| outputs | " + ", ".join(f"`{k}` {v}" for k, v in facts["outputs"].items()) + " |",
        f"| classes | nc {info['nc']}: {', '.join(info['names'])} |",
        f"| source weights | `{info['source_weights']}` sha256 `{info['source_weights_sha256']}` |",
        f"| exported with | ultralytics {info['ultralytics']}, opset {info['opset']}, half {info['half']}, "
        f"simplify {info['simplify']} |",
        f"| git HEAD | {git} |",
        f"| source export | `{export_dir}` |",
        f"| packed | {time.strftime('%Y-%m-%d %H:%M %z')} |",
        "",
        "## Checks at pack time (onnxruntime, CPU)",
        f"- structure: imgsz {facts['imgsz']}, {facts['anchors']} anchors, nc {facts['nc']}: PASS",
        "- parity against the torch model: not run (needs ultralytics)",
        "",
        "## Licence",
        f"- {LICENCE_NOTE}",
    ]
    if note:
        lines += ["", "## Note", note]
    return "\n".join(lines) + "\n"


def make_model_pack(export_dir: str | Path, pack_dir: str | Path, *, note: str = "",
                    limits: str | Path = DEFAULT_LIMITS, repo: str | Path | None = None) -> dict[str, Any]:
    """Write the pack, verify it by re-hashing, and return {kind, pack, files, onnx_sha256}.

    Raises FileExistsError if pack_dir is not empty, ValueError if the export fails its check (nothing is written),
    FileNotFoundError if a policy pack has no limits file, and RuntimeError if the written pack fails verification.
    `repo=None` records 'not recorded' for git HEAD and makes no git call.
    """
    src, dst = Path(export_dir), Path(pack_dir)
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise FileExistsError(f"{dst} exists and is not an empty directory; refusing to write a pack into it")
    kind = _kind(src)
    git = _git_head(None if repo is None else Path(repo))

    if kind == "policy":
        report = check_policy_pack_dir(src)
        if not report["passed"]:
            raise ValueError(f"{src} fails check_onnx: " + "; ".join(report["failures"]))
        limits_path = Path(limits)
        if not limits_path.is_file():
            raise FileNotFoundError(f"limits file not found: {limits_path}")
        info = json.loads((src / "export.json").read_text())
        copies = list(POLICY_FILES) + [n for n in POLICY_OPTIONAL if (src / n).is_file()]
        card = _policy_card(dst.name, src, report, info, limits_path, git, note)
    else:
        facts = validate_yolo_seg_onnx(src / "model.onnx")
        info = json.loads((src / "model.json").read_text())
        if info.get("sha256") != backends.sha256(src / "model.onnx"):
            raise ValueError(f"{src}/model.json sha256 differs from model.onnx (file changed after export)")
        if len(info.get("names", [])) != facts["nc"]:
            raise ValueError(f"model.json lists {len(info.get('names', []))} names for nc={facts['nc']}")
        copies = list(YOLO_FILES)
        card = _yolo_card(dst.name, src, facts, info, git, note)

    dst.mkdir(parents=True, exist_ok=True)
    for n in copies:
        (dst / n).write_bytes((src / n).read_bytes())
    (dst / CARD).write_text(card)
    lines = [f"{backends.sha256(dst / n)}  {n}\n" for n in sorted(p.name for p in dst.iterdir())]
    (dst / SUMS).write_text("".join(lines))

    errors = verify_pack_dir(dst)
    if errors:
        raise RuntimeError(f"{dst} was written but fails verification: " + "; ".join(errors))
    return {"kind": kind, "pack": str(dst), "files": sorted(copies + [CARD]),
            "onnx_sha256": info.get("onnx_sha256") or info.get("sha256")}


def verify_pack_dir(pack_dir: str | Path) -> list[str]:
    """Every problem with a model pack, by re-hashing its files against SHA256SUMS; an empty list means it is intact."""
    pack = Path(pack_dir)
    sums_path = pack / SUMS
    if not sums_path.is_file():
        return [f"no {SUMS} in {pack}"]
    errors: list[str] = []
    listed: dict[str, str] = {}
    for i, line in enumerate(sums_path.read_text().splitlines(), 1):
        m = _SUM_LINE.fullmatch(line)
        if not m:
            errors.append(f"{SUMS} line {i} is malformed: {line!r}")
            continue
        name = m.group(2)
        if "/" in name or "\\" in name or name == SUMS or name in listed:
            errors.append(f"{SUMS} line {i} names a path that is not a unique flat file: {name!r}")
            continue
        listed[name] = m.group(1)
    for p in sorted(pack.iterdir()):
        if p.is_dir():
            errors.append(f"subdirectory {p.name}/ is not allowed in a pack")
        elif p.name != SUMS and p.name not in listed:
            errors.append(f"{p.name} is not listed in {SUMS}")
    for name, digest in sorted(listed.items()):
        f = pack / name
        if not f.is_file():
            errors.append(f"{name} is listed in {SUMS} but missing")
        elif backends.sha256(f) != digest:
            errors.append(f"{name}: sha256 mismatch")
    if {"policy.onnx", "export.json"} <= listed.keys():
        required, onnx_name = POLICY_FILES + (CARD,), "policy.onnx"
    elif "model.json" in listed:
        required, onnx_name = YOLO_FILES + (CARD,), "model.onnx"
    else:
        return errors + ["neither a policy pack nor a YOLO pack"]
    errors += [f"required file {n} is missing from {SUMS}" for n in required if n not in listed]
    if errors:
        return errors
    try:
        meta = json.loads((pack / ("export.json" if onnx_name == "policy.onnx" else "model.json")).read_text())
    except ValueError as e:
        return [f"metadata file is not JSON: {e}"]
    if (meta.get("onnx_sha256") or meta.get("sha256")) != listed[onnx_name]:
        errors.append(f"metadata does not record the sha256 of {onnx_name}")
    m = _CARD_HASH.search((pack / CARD).read_text())
    if not m or m.group(1) != listed[onnx_name]:
        errors.append(f"{CARD} does not state `onnx_sha256: <hash of {onnx_name}>`")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export_dir")
    ap.add_argument("pack_dir")
    ap.add_argument("--note", default="", help="free text for the model card")
    ap.add_argument("--limits", default=str(DEFAULT_LIMITS), help="safety limits file hashed into a policy card")
    ap.add_argument("--repo", default=str(LIBRARY_ROOT), help="git work tree for the HEAD line ('' to skip)")
    a = ap.parse_args(argv)
    try:
        res = make_model_pack(a.export_dir, a.pack_dir, note=a.note, limits=a.limits, repo=a.repo or None)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"[pack] {res['pack']}: {res['kind']} pack, {len(res['files'])} files + {SUMS} + {CARD}, verified by "
          f"re-hashing (onnx sha256 {str(res['onnx_sha256'])[:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
