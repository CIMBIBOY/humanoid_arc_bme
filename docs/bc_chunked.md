# Negative result: an action-chunking policy did not beat BC-RNN

2026-10-06. Preregistered (hypothesis, prediction and kill criterion were
fixed before running), run once, and stopped at the kill criterion.

## Question

Our best behaviour-cloning policy for standing pick-and-place (robomimic
BC-RNN with ±1.5 cm observation noise and a 50-step context) is reported at
24/24 clean but 12/24 under 0.005 action noise per step and 0/24 at 0.01
(those BC-RNN numbers have no kept eval logs, see ../results.md). Compounding
error is the textbook cause, and action chunking (Diffusion Policy, arXiv
2303.04137) is the textbook remedy.

**Hypothesis:** predicting action chunks (receding horizon) rides out
per-step perturbations and removes the "copy the current wrist pose"
shortcut that per-step BC suffers from.

**Prediction:** clean ≥ 20/24; action noise 0.005 ≥ 18/24.
**Kill criterion:** under 12/24 at 0.005 (no better than BC-RNN).

## Setup

- Data: the same 1,000 Mimic-generated demos BC-RNN was trained on, unmodified.
- Policy: robomimic 0.5 Diffusion Policy (IsaacLab 2.3.2 pins robomimic
  0.4.0, which lacks it, so it ran in its own venv). Same 6 observation keys
  (41 dims), same ±1.5 cm observation noise and batch size 100 as BC-RNN;
  observation horizon 2, action horizon 8, prediction horizon 16; DDIM with 10
  inference steps; 66.4M parameters.
- Training: capped at 45 min wall time; stopped at epoch 603 of 800, loss
  1.04 to 0.011. Evaluated the epoch-600 checkpoint.
- Eval: same env, success term, horizon (1,000 steps), seed (303) and
  action-noise definition (Gaussian on the 6 wrist-position dims) as BC-RNN.
  The policy ran in a separate process; the eval harness stacks the 2-frame
  observation history and repeats the first frame at reset, as in training.

## Result

| condition (24 rollouts) | BC-RNN (reported) | chunked, epoch 600 | prediction |
|---|---|---|---|
| clean | 24/24 | **0/24** | ≥ 20/24 |
| action noise 0.005 | 12/24 | **0/24** | ≥ 18/24 |
| action noise 0.01 | 0/24 | 0/24 | none |

22 of 24 rollouts never lifted the box (clean and 0.005); at 0.01, all 24.
An open-loop check (recorded observations of one demo fed in with the same
history) gave 1.2 mm mean wrist-target error and 1.6 mm on the hands, so the
network fits the data and the observation order, normalisation and chunk
indexing look right. The failure is closed-loop.

**Verdict:** kill criterion met; this configuration does not help, and is
worse than BC-RNN even without noise. It does not show that action chunking
cannot work here.

## Untested candidate causes

Each would be its own experiment:

1. The 2-frame history lets the policy extrapolate its own recent motion,
   the same "copycat" shortcut BC-RNN needed observation noise and a long
   context to escape. The noise was present but not tuned for chunks.
2. Open-loop execution of 8 absolute wrist targets per chunk.
3. A learning rate not yet annealed (epoch 600 of 800; the 45 min cap was a
   wall-clock limit, not convergence).
4. A subtle eval wiring difference that the open-loop check cannot see.
