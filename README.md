# humanoid_arc_bme

Demos on a simulated Unitree G1 humanoid (29 DoF, three-finger hands) in
IsaacLab, following the current literature: a reinforcement-learned lower body
under arm motion, a hierarchical walk-and-manipulate tour, massively parallel
locomotion training, and behaviour cloning from scripted and Mimic-generated
demonstrations.

The manipulation stack (robot assets, the pick-and-place task, the scripted
pick-and-place skill, BC evaluation) lives once in **humanoid_arc_core**; this
repo imports it and adds what is specific to the legged G1.

## Demos

| demo | what it shows | entry point |
|---|---|---|
| Table tour | hierarchical control: a navigator commands the AGILE lower-body policy to walk a square of four tables; at each, the core's scripted skill picks a box and puts it down | `scripts/run_table_tour.py` |
| Lower body under arm motion | PPO lower body (8,192 envs) with a working height command, trained while the arms replay pick-place motion and carry payloads (FALCON/HOMIE style) | `humanoid_arc_bme/envs/g1_lower_body`, `scripts/lower_body/` |
| Walking from scratch | stock IsaacLab G1 flat walking at 8,192 envs in 28 min; evaluation against the published checkpoint | `scripts/locomotion/` |
| Parallel-training infrastructure | throughput vs env count, and a check that co-located parallel envs don't interact | `scripts/infra/` |
| Behaviour cloning | 16 scripted demos to 1,000 with Isaac Lab Mimic, robomimic BC-RNN, and a preregistered negative result for action chunking | core task + [docs/bc_chunked.md](docs/bc_chunked.md) |

## Results

Headline numbers. All of them, with commands and which ones lack kept logs,
are in **[results.md](results.md)**.

| result | number | evidence |
|---|---|---|
| scripted pick-place at one table, 4 envs x 2 episodes | 8/8 placed, 7 mm | re-run 2026-10-08 |
| replay of 16 recorded demos from actions alone | 16/16, ≤ 2.6 mm from the recording | re-run 2026-10-08 |
| table tour, 1 lap | 4/4 placed, 7-31 mm, 0 drops | re-run 2026-10-08 |
| Mimic generation, 16 to 1,000 demos | 96.9% success | dataset kept |
| BC-RNN (obs noise + 50-step context) | 45/48 reported | one run kept, two-seed claim unverified |
| action-chunking policy | 0/24 (negative result) | log kept |
| lower body, height tracking 0.62-0.78 m | 0.3-1.1 cm error | log kept |
| lower body, pitch swing under arm motion | 8.1-9.1° (target under 3°, not met; AGILE 11.3°) | log kept |
| G1 walking from scratch, 8,192 envs | 27.8 min; 0.059 m/s tracking vs 0.098 pretrained | training log kept; eval log and final checkpoint missing |
| PPO throughput, RTX 5090 | 178,685 env-steps/s at 8,192 envs | CSV kept |

## Install

Needs IsaacLab 2.3.2 on Isaac Sim 5.1 and a checkout of humanoid_arc_core.

```bash
git clone <humanoid_arc_core repository> ../humanoid_arc
export ISAACLAB_DIR=/path/to/IsaacLab           # default: ../IsaacLab
./setup.sh                                       # numpy<2 overlay + NVIDIA assets into assets/
```

`setup.sh` writes only inside this repo. It fetches the G1 USD, the AGILE
lower-body policy and IsaacLab's pretrained G1 walking checkpoint (converted to
the rsl-rl-lib 5 format) from NVIDIA's public content bucket; none of them are
committed. The `.deps/` overlay pins numpy 1.26 because the pinocchio build that
Pink IK needs segfaults under numpy 2.

`scripts/launch.sh` runs any script through IsaacLab with this repo, the core
(`$HUMANOID_ARC_CORE`; default: the parent repo when this is its `bme/` submodule, else `../humanoid_arc`) and the overlay on
`PYTHONPATH`, and points the core at `assets/` (`$HUMANOID_ARC_ASSETS`).

## Run

```bash
scripts/launch.sh scripts/run_table_tour.py --headless --laps 1
scripts/launch.sh scripts/run_table_tour.py --headless --laps 1 --video logs/tour.mp4 --speed 1.5

# lower body: check the env with AGILE in the legs, train, evaluate
scripts/launch.sh scripts/lower_body/check_lower_body_env.py --headless --num_envs 256 --steps 1000
scripts/launch.sh ../humanoid_arc/tools/isaaclab_tool.py reinforcement_learning/rsl_rl/train.py \
    --task HumanoidArc-G1-LowerBody-v0 --num_envs 8192 --headless
scripts/launch.sh scripts/lower_body/eval_lower_body.py --headless --checkpoint logs/rsl_rl/g1_lower_body/<run>/model_<N>.pt

# locomotion
scripts/launch.sh "$ISAACLAB_DIR/scripts/reinforcement_learning/rsl_rl/train.py" \
    --task Isaac-Velocity-Flat-G1-v0 --num_envs 8192 --max_iterations 1500 --seed 42 --headless
scripts/launch.sh scripts/locomotion/eval_velocity.py --headless --checkpoint assets/policies/g1_velocity_flat/checkpoint.pt

# infrastructure
scripts/infra/benchmark_throughput.sh 4096 8192
scripts/launch.sh scripts/infra/check_env_isolation.py --headless
```

The lower-body env replays recorded pick-place arm motion. Record it with the
core (`tools/record_demos.py --speed 1.5 --dataset_file datasets/g1_pick_place_scripted_v2.hdf5`)
and put it at `datasets/g1_pick_place_scripted_v2.hdf5` here, or point
`$HUMANOID_ARC_ARM_DEMOS` at it.

## Layout

```
humanoid_arc_bme/
  envs/g1_table_tour/   four tables on a generated circuit (extends the core pick-place task)
  envs/g1_lower_body/   lower-body PPO env: AGILE's observation/action layout, scripted arms, payloads, pushes
  policies/navigator.py walk-to-pose controller for the AGILE command [vx, vy, wz, hip_height]
  policies/table_tour.py orchestrates navigation and one pick-place skill per table
scripts/
  launch.sh  run_table_tour.py
  lower_body/  baseline, env check, warm start from AGILE, eval, side-by-side video
  locomotion/  eval_velocity.py, convert_rsl_rl_checkpoint.py
  infra/       benchmark_throughput.sh, check_env_isolation.py
docs/
  literature.md   the background: two-layer architecture, parallel RL, sim-to-real, human motion data
  audit.md        bugs found while generalising one table to four
  bc_chunked.md   preregistered negative result for action chunking
results.md        every number, its command, and whether its log was kept
```

The G1-on-legs pick-place variant is not a separate env: it is the core task
with its default `agile_legs` base.

## License

Not chosen yet. TODO for the repository owner: add a LICENSE before making
this repository public. Third-party assets (NVIDIA's G1 USD and policies) are
fetched by `setup.sh` under NVIDIA's terms and are not redistributed here.
