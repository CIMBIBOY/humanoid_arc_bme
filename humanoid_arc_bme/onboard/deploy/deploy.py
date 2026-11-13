"""Verified delivery of a checkpoint directory to the robot over scp (B3.5).

Dry run by default: prints the exact commands. With execute=True it
  1. verifies the local manifest,
  2. copies to <dest>.incoming on the robot,
  3. checks every hash on the robot with sha256sum -c,
  4. only then renames <dest>.incoming to <dest>.
It never starts a policy, a service or a ROS node.
"""

import os
import re

from . import manifest
from .runner import run

_SAFE_HOST = re.compile(r"^[A-Za-z0-9_.@-]+$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")


class DeployError(RuntimeError):
    pass


def _check_args(host, dest):
    if not _SAFE_HOST.match(host):
        raise DeployError(f"unsafe host: {host!r}")
    if not _SAFE_PATH.match(dest) or ".." in dest.split("/") or dest in ("", "/", "~"):
        raise DeployError(f"unsafe destination: {dest!r}")


def plan(local_dir, host, dest):
    """Return the ordered list of argv lists that delivery would run."""
    _check_args(host, dest)
    incoming = dest.rstrip("/") + ".incoming"
    local = local_dir.rstrip("/") + "/."
    return [
        ["ssh", host, "rm", "-rf", incoming],
        ["scp", "-r", "-p", local, f"{host}:{incoming}"],
        ["ssh", host, f"cd {incoming} && sha256sum -c {manifest.SUMS_NAME}"],
        ["ssh", host, f"rm -rf {dest}.previous && if [ -e {dest} ]; then mv {dest} {dest}.previous; fi && mv {incoming} {dest}"],
    ]


def deliver(local_dir, host, dest, execute=False, runner=run, out=print):
    problems = manifest.verify(local_dir)
    if problems:
        raise DeployError("local manifest check failed: " + "; ".join(problems))
    cmds = plan(local_dir, host, dest)
    for c in cmds:
        out(("RUN  " if execute else "PLAN ") + " ".join(c))
    if not execute:
        return True
    for c in cmds:
        r = runner(c, timeout=600)
        if r.code != 0:
            raise DeployError(f"failed ({r.code}): {' '.join(c)}\n{r.err.strip() or r.out.strip()}")
    out(f"delivered {os.path.basename(local_dir.rstrip('/'))} to {host}:{dest}; hashes verified on the robot")
    return True
