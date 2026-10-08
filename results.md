# Results

Measured on one machine: RTX 5090 (32 GB), driver 580.159.03, IsaacLab 2.3.2
on Isaac Sim 5.1, rsl-rl-lib 5.0.1, robomimic 0.4.0. Every row gives the
command that produced it and says whether a log of that run still exists.

**Evidence** column:

- **log**: the run's output is kept.
- **no log**: the number was written down at the time, but the run's output
  was not kept. Re-run the command before you quote it.
- **re-run**: measured again during the 2026-10-08 restructuring.

Commands assume `scripts/launch.sh` from this repo (see README). `core:`
means a script from humanoid_arc_core, run through the core's
`tools/launch.sh`.

## Pick-and-place on the standing G1 (core task, `agile_legs` base)

| what | result | command | evidence |
|---|---|---|---|
| scripted pick-place, 4 envs x 2 episodes | 8/8 placed, 7 mm | core: `tools/run_pick_place.py --headless --num_envs 4 --episodes 2` | re-run 2026-10-08, identical to the 2026-09-28 run line by line |
| 16 recorded demos replayed from actions alone | 16/16 succeed; final object max 2.6 mm (mean 0.6 mm) from the recording | core: `tools/replay_demos.py --headless --dataset_file datasets/g1_pick_place_scripted.hdf5` | re-run 2026-10-08 (2026-09-28 notes said max 2.0, mean 0.5) |
| place error vs place offset (single table, 4 envs) | sideways 0.25 m: 4-5 mm; forward 0.10: 7 mm; 0.15: 24 mm; 0.20: 60 mm, 0/4 placed | core: `tools/run_pick_place.py --place_offset X Y Z` | no log |
| Mimic: 16 scripted demos to 1,000 | 1,000 successes in 1,032 attempts (96.9%); place error mean 16 mm, max 24 mm | core: IsaacLab `annotate_demos.py --auto` + `generate_dataset.py` through `tools/isaaclab_tool.py` | dataset kept; no generation log |
| BC-RNN, IsaacLab default config | 0/8 | robomimic `--algo bc`, 500 epochs, `eval/eval_bc.py --num_envs 8` | no log |
| BC-RNN + ±1.5 cm observation noise | 12/24 | `--algo bc_obs_noise`, 300 epochs | no log |
| BC-RNN + noise + 50-step context | reported as **45/48** (21/24 and 24/24, two eval seeds) | `--algo bc_obs_noise_h50`, 300 epochs, `eval/eval_bc.py --num_envs 24 --horizon 1000 --seed 303` | **only one training run (`bc_noise_h50_300`) is kept and no eval log for either seed: the "two seeds" are unverified** |
| noise sweep on that policy | obs noise 0.005/0.01/0.02: 20, 20, 8 of 24; action noise 0.005/0.01/0.02: 12, 0, 0 of 24 | `eval/eval_bc.py --obs_noise S` / `--action_noise S` | **no eval logs**: unverified |
| chunked policy (robomimic 0.5 Diffusion Policy) | 0/24 clean, 0/24 at action noise 0.005 and 0.01 | [docs/bc_chunked.md](docs/bc_chunked.md) | negative result; run log kept in the core repo |

## Table tour (walk between four tables, pick-place at each)

| what | result | command | evidence |
|---|---|---|---|
| 1 lap | 4/4 placed, 0 drops; red 7 mm, green 31 mm, blue 29 mm, yellow 29 mm | `scripts/run_table_tour.py --headless --laps 1` | re-run 2026-10-08 from this repo: identical (4/4; 7, 31, 29, 29 mm) |
| the same on Isaac Sim 5.0 vs 5.1 | identical | same | no log |

## Locomotion

| what | result | command | evidence |
|---|---|---|---|
| IsaacLab pretrained G1 flat walking (converted checkpoint), 32 envs x 1,000 steps | 0 falls, 0.095 m/s tracking error | `scripts/locomotion/eval_velocity.py --headless --checkpoint assets/policies/g1_velocity_flat/checkpoint.pt` | no log |
| same, 1,024 envs | 0 falls, 0.098 m/s (Isaac Sim 5.0), 0.097 m/s (5.1); pelvis z 0.630 | `... --num_envs 1024` | no log |
| G1 flat walking trained from scratch: stock `Isaac-Velocity-Flat-G1-v0` and PPO config, 8,192 envs, 1,500 iterations, seed 42 | 27.8 min, 295M env-steps; reward 30.1 at iteration 1,500 | IsaacLab `rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 8192 --max_iterations 1500 --seed 42 --headless` | training log (tensorboard) kept |
| its eval, 1,024 envs x 1,000 steps | 0 falls, **0.059 m/s** tracking, pelvis z 0.660 | `eval_velocity.py --checkpoint .../model_1499.pt --num_envs 1024` | **no eval log, and `model_1499.pt` was deleted by mistake** (only `model_950.pt` remains). Re-training takes about 28 min |

The 0.059 vs 0.098 comparison is not a fair win for 8,192 envs: that run saw
twice the samples of the stock recipe, and it is one seed.

## Lower body under arm motion (`HumanoidArc-G1-LowerBody-v0`)

Goal: a lower body with a working height command that rocks less than AGILE
while the arms move. AGILE's baseline on the scripted pick-place (8 envs,
speed 1.5): pelvis z 0.721 m, pitch swing 11.3°, foot slip 0.2 cm, 0/8 falls
(`scripts/lower_body/baseline_lower_body.py`, no log). AGILE in our training
env: 37 falls in 258 episodes (`scripts/lower_body/check_lower_body_env.py`,
no log). AGILE's height input does nothing: a sweep from 0.0 to 5.0 gives the
same stance (no log).

| run | result | evidence |
|---|---|---|
| from scratch, 300 iterations | failed (pelvis sinks to 0.18-0.5 m) | no log |
| warm start from AGILE weights (`init_lower_body_from_agile.py`) | stands, walks, ignores the height command (5.1 cm mean error) | no log |
| warm start, height column re-drawn, height rewards doubled, 300 iterations at 4,096 envs | height error 0.4-1.4 cm from 0.62 to 0.78 m | no log, no checkpoint |
| `s4_main`: continued 600 iterations at 8,192 envs (about 95 min), stronger tilt/slip penalties | height error 0.3-1.1 cm (mean 0.6); pitch swing 8.1-9.1° (target under 3°: **not met**); 32 falls with disturbances (21 clean), 28 of them at the 0.62 m block right after spawn | `scripts/lower_body/eval_lower_body.py --checkpoint .../model_898.pt [--disturb]`: **log and checkpoint kept** |
| `s5_modes`: three arm modes | training stopped at iteration 901 of 1,098 | checkpoint `model_900.pt` kept; **not evaluated** |

Open: the pitch swing target is not met; standing at 0.78 m shuffles (140 cm
of foot slip per 250 steps), probably at the leg's reach limit; 0.8 m/s was
not tested.

## Parallel training infrastructure

Throughput, stock `Isaac-Velocity-Flat-G1-v0` PPO, idle GPU, 30 iterations
(first 5 dropped). `scripts/infra/benchmark_throughput.sh`; **CSV kept**.

| num_envs | env-steps/s | iteration (collect + learn) | GPU memory |
|---|---|---|---|
| 1,024 | 34,369 | 0.72 s | 3.4 GB |
| 2,048 | 63,651 | 0.77 s | 3.9 GB |
| 4,096 | 103,519 | 0.95 s | 5.0 GB |
| 8,192 | 178,685 | 1.10 s | 7.1 GB |
| 16,384 | 245,440 | 1.60 s | 11.3 GB |

Scaling is near-linear to 8,192 and saturates at 16,384, at about 0.5 MB per
env.

Env isolation (`scripts/infra/check_env_isolation.py`): every robot stacked
at the same spot (`env_spacing=0`), walked by the pretrained policy. No log
was kept.

| run | envs x steps | falls | tracking error | verdict |
|---|---|---|---|---|
| spacing 2.5 m, filtered | 64 x 500 | 0 | 0.090 m/s | isolated |
| co-located, filtered | 64 x 500 | 0 | 0.094 m/s | isolated |
| co-located, **unfiltered** (control) | 64 x 500 | 31,903 | 0.580 m/s | interacting |
| co-located, filtered | 8,192 x 300 | 0 | 0.092 m/s | isolated |

With `filter_collisions=True`, co-located robots don't interact; spacing is
cosmetic. The control shows the check can detect a leak.
