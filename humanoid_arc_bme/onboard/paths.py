"""Where the shipped configuration lives. One definition, so no caller guesses a repository layout.

`CONFIG_DIR` is `<bme repo>/config` (the S1 limits, runtime and timing yaml). A deployment that keeps its configuration
elsewhere sets `HUMANOID_ARC_CONFIG`; the runtime CLI also takes `--config`.
"""

from __future__ import annotations

import os
from pathlib import Path

BME_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("HUMANOID_ARC_CONFIG", BME_ROOT / "config"))
DEFAULT_LIMITS = CONFIG_DIR / "g1_limits.yaml"
FIRSTWALK_LIMITS = CONFIG_DIR / "g1_limits_firstwalk.yaml"
