# Chunked policy (robomimic 0.5 Diffusion Policy) for open problem B

Started 2026-10-06. Open problem B ([plan.md](../../plan.md#problem-b-bc-fragile-to-action-noise)): BC-RNN `bc_obs_noise_h50` is 24/24 clean but
12/24 at action noise 0.005 and 0/24 at 0.01 (FINDINGS M3; no eval log
files exist in the repo, so they are unverified).

## Hypothesis
Predicting an action chunk (receding horizon) lets the policy ride out per-step perturbations
and removes the "copy the current wrist pose" shortcut that per-step BC suffers from, so it is
more tolerant of action noise than BC-RNN, trained on the same data.

## Setup (fixed before running)
- Dataset: existing `datasets/g1_pick_place_generated.hdf5` (v1, 1,000 demos), unmodified.
  Any conversion goes to `outputs/bc_chunked/` as a new file.
- Separate venv `~/envs/robomimic05` (outside Isaac / robomimic 0.4.0 install).
- Eval: same protocol as BC-RNN, `scripts/bc/eval_bc.py` semantics (24 rollouts, horizon 1000,
  seed 303, wrist-position action noise std 0.005 / 0.01 / clean).
- Training budget: at most 45 min wall time.

## Prediction (numeric)
- Clean >= 20/24.
- Action noise 0.005 >= 18/24 (baseline 12/24).
- Action noise 0.01: no prediction (baseline 0/24).

## Kill criterion
If 0.005 action noise gives < 12/24, the chunked policy does not help; report and stop.

## Steps and results
(filled in below as each step is run)

### B1: separate env (PASS, 2026-10-06)
`uv venv ~/envs/robomimic05 --python 3.10`; `uv pip install git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0`
(PyPI only has robomimic 0.3.0). Got robomimic 0.5.0, torch 2.14.1+cu130 (sees the RTX 5090),
diffusers; `config_factory("diffusion_policy")` loads. Isaac Sim / Isaac Lab / robomimic 0.4.0 untouched.

### B2: smoke test on the existing dataset (PASS)
Found, in order: (a) the dataset's `env_args` has no `env_kwargs`, which 0.5's `train.py` requires ->
`scripts/bc_chunked/train.py` patches that lookup (dataset untouched, rollouts disabled); (b) diffusion
policy needs `frame_stack=2` and `seq_length=16` (shape error otherwise) -> set in
`scripts/bc_chunked/make_config.py`. Same 6 obs keys (41 dims), same +-1.5cm uniform obs noise and batch
size 100 as BC-RNN; actions min-max normalised (32 dims; 4 lower-body dims are constant, handled).
Result: 4 epochs (100 steps each), loss 1.043 -> 0.996 -> 0.808 -> 0.421, checkpoints saved and load
(keys model/config/shape_metadata/normalisation stats), 2.4 s per epoch, 8.2 GB GPU. No dataset conversion needed.
Design choice before B3: DDIM with 10 inference steps (instead of 100-step DDPM) for tractable eval.

### B3: train (45 min cap) and evaluate (FAIL, negative result)
Training: `outputs/bc_chunked/dp_800.json` (800-epoch cosine schedule, 100 steps/epoch, batch 100, DDIM 10 inference
steps, obs horizon 2, action horizon 8, prediction horizon 16, 66.4M params). Stopped by me at 44 min wall, epoch 603
(about 4.4 s/epoch because 1 GB checkpoints are written each epoch and I ran two short eval smoke tests meanwhile).
Loss 1.04 -> 0.011. Evaluated `model_epoch_600.pth` (LR schedule not finished). The cap was a wall-clock limit, not a
convergence limit, so under-training is not excluded.

Eval wiring: originally a separate adapted copy of the BC eval script, now merged into `scripts/bc/eval_bc.py --backend server` (same env, seed 303, horizon 1000,
success term, action-noise definition on the 6 wrist-position dims); the policy runs in `scripts/bc_chunked/server.py`
inside the 0.5 env, because Isaac's python has robomimic 0.4.0. robomimic 0.5's diffusion policy does not keep an
observation history itself, so the server stacks 2 frames and repeats the first frame at reset (as in training).

| condition (24 rollouts, seed 303) | BC-RNN baseline (docs only) | chunked (DDIM, epoch 600) | prediction |
|---|---|---|---|
| clean | 24/24 | **0/24** | >= 20/24 |
| action noise 0.005 | 12/24 | **0/24** | >= 18/24 (kill if < 12/24) |
| action noise 0.01 | 0/24 | 0/24 | none |

Outcomes: 22/24 "never lifted" in clean and 0.005, 24/24 at 0.01; several boxes knocked off the table.
Sanity check (a one-off script, since deleted: demo_0, recorded observations fed in open loop with the same
2-frame history): mean absolute wrist-target error 1.2 mm, hands 1.6 mm. So the network fits the data and the wiring
of observation order, normalisation and chunk indexing is plausible; the failure is closed-loop, not a fit problem.

Verdict: prediction not met, kill criterion triggered (0/24 < 12/24 at 0.005). The chunked policy as configured does not
help. It is worse than BC-RNN even with no noise. This does not show that action chunking cannot work; it shows this
configuration fails. Untested candidate causes (each would be its own step, not done here): the 2-frame history
lets the policy extrapolate its own recent motion (the "copycat" shortcut that BC-RNN needed a 1.5 cm observation noise
and a 50-step context to escape; the noise is in this run too but was not tuned for chunks), open-loop execution of
8 absolute wrist targets, a not-yet-annealed LR (epoch 600 of 800), and a possible subtle eval wiring difference
that the offline check cannot see. Stopped here as the task specified.
