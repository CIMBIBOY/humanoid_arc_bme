"""python -m humanoid_arc_bme.onboard.deploy {preflight|manifest|verify|deploy} ..."""

import argparse
import sys

from . import deploy, manifest, preflight


def main(argv=None):
    p = argparse.ArgumentParser(prog="humanoid_arc_bme.onboard.deploy", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight", help="read-only checks of this computer against the plan's stack")
    m = sub.add_parser("manifest", help="write manifest.json and SHA256SUMS for a checkpoint directory")
    m.add_argument("directory"); m.add_argument("--seed", type=int, required=True)
    m.add_argument("--config"); m.add_argument("--dataset"); m.add_argument("--notes", default="")
    v = sub.add_parser("verify", help="check a checkpoint directory against its manifest")
    v.add_argument("directory")
    d = sub.add_parser("deploy", help="copy a verified checkpoint to the robot (dry run unless --execute)")
    d.add_argument("directory"); d.add_argument("--host", required=True); d.add_argument("--dest", required=True)
    d.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)

    if a.cmd == "preflight":
        checks = preflight.run_all()
        print(preflight.format_report(checks))
        return 1 if any(c.status == "fail" for c in checks) else 0
    if a.cmd == "manifest":
        mm = manifest.build(a.directory, seed=a.seed, config_path=a.config, dataset_path=a.dataset, notes=a.notes)
        print(f"{len(mm['files'])} files, commit {mm['git_commit']}")
        return 0
    if a.cmd == "verify":
        probs = manifest.verify(a.directory)
        print("\n".join(probs) if probs else "OK")
        return 1 if probs else 0
    try:
        deploy.deliver(a.directory, a.host, a.dest, execute=a.execute)
    except deploy.DeployError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
