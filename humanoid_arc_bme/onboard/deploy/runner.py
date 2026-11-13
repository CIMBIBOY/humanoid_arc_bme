"""One place where commands are run, so every other module can be tested with a fake."""

import subprocess
from dataclasses import dataclass


@dataclass
class Result:
    code: int
    out: str
    err: str = ""


def run(cmd, timeout=30):
    """Run argv (a list, never a shell string). Missing program or timeout gives code 127 / 124."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return Result(p.returncode, p.stdout, p.stderr)
    except FileNotFoundError:
        return Result(127, "", f"not found: {cmd[0]}")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"timeout after {timeout}s: {' '.join(cmd)}")
