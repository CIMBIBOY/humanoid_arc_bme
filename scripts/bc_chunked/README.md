# Chunked BC policy (robomimic 0.5 Diffusion Policy)

Trains an action-chunking policy on the pick-place Mimic dataset and evaluates
it in Isaac. Result so far: negative (0/24 clean with the first config), see
[log.md](log.md).

## Separate environment

IsaacLab 2.3.2 pins robomimic 0.4.0, which has no Diffusion Policy, so
training and inference run in their own venv. Isaac's python is not touched.

```bash
uv venv ~/envs/robomimic05 --python 3.10
uv pip install --python ~/envs/robomimic05/bin/python git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0
export ROBOMIMIC05_PYTHON=~/envs/robomimic05/bin/python
```

(PyPI only has robomimic 0.3.0.)

## Files

| file | runs in | what |
|---|---|---|
| `make_config.py` | robomimic 0.5 venv | writes a Diffusion Policy config with the same observation keys, observation noise and batch size as the BC-RNN `bc_obs_noise_h50` baseline |
| `train.py` | robomimic 0.5 venv | robomimic 0.5 `train.py` with one patch: our IsaacLab HDF5 `env_args` has no `env_kwargs` key |
| `server.py` | robomimic 0.5 venv | policy server; started by `scripts/bc/eval_bc.py --backend server`, not by hand |

## Run

```bash
$ROBOMIMIC05_PYTHON scripts/bc_chunked/make_config.py outputs/bc_chunked/dp.json --epochs 800
$ROBOMIMIC05_PYTHON scripts/bc_chunked/train.py --config outputs/bc_chunked/dp.json
scripts/launch.sh scripts/bc/eval_bc.py --headless --num_envs 24 --horizon 1000 --seed 303 \
    --backend server --checkpoint outputs/bc_chunked/runs/<run>/models/model_epoch_600.pth [--action_noise 0.005]
```

The server stacks the policy's observation history itself (robomimic 0.5's
diffusion policy does not), repeating the first frame at episode start as in
training (`pad_frame_stack=True`).
