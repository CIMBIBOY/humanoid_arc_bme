"""robomimic 0.5 train.py with one patch: our Isaac Lab HDF5 `env_args` has no `env_kwargs`
key, which 0.5's get_env_metadata_from_dataset requires. Rollouts are disabled (we evaluate in
Isaac Lab with scripts/bc/eval_bc.py --backend server), so a stub is enough. The dataset file is not touched.

    $ROBOMIMIC05_PYTHON scripts/bc_chunked/train.py --config outputs/bc_chunked/X.json
"""
import json
import sys

import robomimic.utils.file_utils as FileUtils

_orig = FileUtils.get_env_metadata_from_dataset


def _patched(dataset_path, set_env_specific_obs_processors=True):
    import h5py, os
    with h5py.File(os.path.expanduser(dataset_path), "r") as f:
        meta = json.loads(f["data"].attrs["env_args"])
    meta.setdefault("env_kwargs", {})
    return meta


FileUtils.get_env_metadata_from_dataset = _patched
import runpy  # noqa: E402

if __name__ == "__main__":
    runpy.run_module("robomimic.scripts.train", run_name="__main__")
