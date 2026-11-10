"""Run one of IsaacLab's own scripts with our gym ids registered.

IsaacLab's dataset and training tools (Mimic annotate/generate, robomimic,
rsl_rl train) look tasks up by gym id, and only know the ids in
isaaclab_tasks. This registers the core's tasks, plus any modules listed in
$HUMANOID_ARC_TASK_MODULES (comma-separated, e.g. a downstream repo's envs),
and then runs the tool unchanged. The path is relative to IsaacLab's
scripts/ directory.

    scripts/launch.sh scripts/isaaclab_tool.py imitation_learning/robomimic/train.py \
        --task HumanoidArc-G1-PickPlace-Demo-v0 --algo bc_obs_noise_h50 --dataset datasets/g1_pick_place_generated.hdf5
"""

import builtins
import functools
import importlib
import os
import runpy
import sys
from pathlib import Path

# Kit block-buffers piped stdout and the hard exit below skips the flush, so
# without this a tool's prints never reach a log file.
builtins.print = functools.partial(builtins.print, flush=True)

import isaaclab  # noqa: E402

TASK_MODULES = ["humanoid_arc_bme.tasks.pick_place"]
TASK_MODULES += [m for m in os.environ.get("HUMANOID_ARC_TASK_MODULES", "").split(",") if m]
for module in TASK_MODULES:
    importlib.import_module(module)  # registers the module's gym ids

if len(sys.argv) < 2:
    sys.exit(__doc__)
ISAACLAB_SCRIPTS = Path(isaaclab.__file__).resolve().parents[3] / "scripts"
tool = ISAACLAB_SCRIPTS / sys.argv[1]
sys.argv = [str(tool)] + sys.argv[2:]
sys.path.insert(0, str(tool.parent))  # tools import their sibling modules (e.g. cli_args)
code = 0
try:
    runpy.run_path(str(tool), run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else 1
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
# simulation_app.close() can hang on this install; exit hard instead.
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
