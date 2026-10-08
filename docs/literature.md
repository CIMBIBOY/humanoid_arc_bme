# Literature: humanoid locomotion and loco-manipulation on one GPU

The background behind the demos in this repo, merged from the notes we kept
while building them (2026-09 to 2026-10). Each claim is marked with how far it
was checked:

- **V**: read in the primary source (paper, repo, docs).
- **A**: abstract only.
- **R**: from memory or secondhand. Treat it as a hypothesis.

Numbers measured on our own machine are in [../results.md](../results.md). When
a number here disagrees with one there, the measured one wins.

## 1. The dominant architecture: two layers

1. **Low level: a whole-body or lower-body controller trained with massively
   parallel RL.** This is the only part trained with thousands of envs:
   HOMIE used 4,096 envs on an RTX 4090 (V), and the AGILE quick-start uses
   2,048 (V). Our walking run used 8,192 envs (results.md).
2. **Task level: a policy learned by imitation** from a few demos,
   multiplied in sim. HOMIE uses 50 episodes per task (V), Isaac Lab Mimic
   turns 5 demos into 1,000 (V), and SONIC + GR00T use 300-3,900
   trajectories (V).

Low-level variants, from what this repo uses to the current frontier:

| variant | examples | fits one RTX 5090? |
|---|---|---|
| RL lower body + IK arms (**this repo**) | NVIDIA GR00T "Decoupled WBC" (V), Isaac Lab `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` (V) | yes |
| RL body + teleoperated arms / dual-agent RL | HOMIE 2502.13013, FALCON 2505.06776, AMO 2505.03738 (V) | yes |
| universal motion tracker + VLA | TWIST 2505.02833, BeyondMimic 2508.08241, NVIDIA SONIC 2511.07820 (V) | fine-tune only: SONIC trained on 128 GPUs x 7 days (V) |

NVIDIA's own stack moved from Decoupled WBC (2025) to SONIC + GR00T N1.7 VLA
(2026); their G1 VLA workflow recommends 50-100 teleop demos (V). The AGILE
lower-body policy used here comes from NVIDIA's AGILE framework
(github.com/nvidia-isaac/WBC-AGILE, arXiv 2603.20147, V).

### Lower-body controllers that tolerate arm motion

Relevant to the lower-body demo (`humanoid_arc_bme/envs/g1_lower_body`).

| work | how arm motion enters training | code | real G1 | note |
|---|---|---|---|---|
| FALCON, 2505.06776 | two RL agents (legs, upper body), shared proprioception; curriculum of forces at the hands up to ±100 N | MIT, G1-29DoF configs, no checkpoints (V) | yes (A) | the recipe our lower-body env follows (arm trajectories + payloads) |
| HOMIE, 2502.13013 | legs trained under an upper-body pose curriculum | CC BY-NC-SA 4.0 (R) | yes, G1 + Dex3 (R) | same idea, non-commercial license |
| AMO, 2505.03738 | trajectory optimizer (Crocoddyl) makes whole-body references, distilled into RL | Apache-2.0 with checkpoints (R) | yes (R) | needs the optimizer too |
| SONIC, 2511.07820 | whole-body motion-tracking foundation model | code + G1 checkpoints (R) | yes (R) | replaces both lower body and IK |

We found no published G1 lower body that reads arm targets, rejects payload
forces and ships checkpoints (R), which is why the demo trains its own.

## 2. Massively parallel RL: how isolation and scale work

Reference: Rudin et al., *Learning to Walk in Minutes Using Massively
Parallel Deep RL*, arXiv 2109.11978 (V; RTX A6000, ANYmal, not a humanoid).

- **One PhysX simulation, thousands of instances.** Policy inference,
  physics, rewards and observations stay on the GPU; PCIe transfer is up to
  50x slower than the GPU compute (V).
- **Isolation:** per-environment collision groups are the guarantee, env
  spacing only a convenience (V).
- **PPO in this regime:** batch = robots x steps per robot; convergence
  breaks below about 25 steps per robot, so use few steps, many robots and
  large mini-batches. Timeouts must be told apart from failures for the
  critic (V).

How this maps onto IsaacLab (source checked, V):

| mechanism | IsaacLab | setting |
|---|---|---|
| instanced cloning | `replicate_physics=True` (default) | keep True |
| cross-env isolation | `filter_collisions=True` (default), cloner `enable_env_ids=True`: clones can be co-located with automatic filtering | keep True |
| shared colliders | `collision_group=-1` | ground and other global prims |

Caveat: manager-based envs need `replicate_physics=True` for the automatic
filtering (IsaacLab issue #1918). We checked isolation directly instead of
trusting it: see `scripts/infra/check_env_isolation.py` and results.md.

The usual throughput killer is host-side Python: keep rewards, observations,
resets and randomization as batched torch ops on the device.

Sizing: NVIDIA's G1 rough-terrain benchmark (RTX 4090, 4,096 envs) reports
94k / 88k / 82k env-steps/s (step / step + inference / full PPO loop) and
6.1 GB of VRAM (V). Isaac Lab's framework paper (arXiv 2511.04831) benchmarks
a real RTX 5090 on G1 locomotion and finds its lead is largest at small env
counts (V; numbers only in log-scale plots). Our own measurement (results.md)
saturates between 8,192 and 16,384 envs at about 0.5 MB per env, a third of
the 1.5 MB per env the 4090 benchmark implies.

## 3. Simulators and versions

- **PhysX (Isaac Lab default)** is what this repo uses.
- **Newton** (Warp-based, co-developed with Google DeepMind and Disney
  Research) ships in Isaac Lab 3.0 as an *experimental* backend with
  breaking API changes (V). Not adopted yet.
- **Isaac Lab 3.0** (tag `v3.0.0-EA`, 2026-09-16) needs Isaac Sim 6.1,
  Python 3.12, PyTorch 2.11 (V). Breaking changes include quaternion order
  WXYZ to XYZW (the silent-bug risk), `ProxyArray` data, and split
  `write_*_to_sim` APIs (V). It keeps the G1 locomanipulation task, Mimic,
  Pink IK and PhysX as the default (V). This repo targets Isaac Lab 2.3.2 on
  Isaac Sim 5.1.
- Published rsl_rl checkpoints from Isaac Lab (`--use_pretrained_checkpoint`)
  are in the pre-v5 format and fail under rsl-rl-lib 5.x with
  `KeyError: 'actor_state_dict'`. `scripts/locomotion/convert_rsl_rl_checkpoint.py`
  remaps the keys; weights, parameter order and optimizer state are unchanged.

## 4. If PPO stalls (pushes, whole-body tasks)

Try off-policy then, not before; PPO is still faster per wall-clock hour on
plain velocity tracking.

- **RSL-RL-SAC**, Sabatini, Li & Hutter, arXiv 2605.24975 (A): closes the gap
  with PPO in massively parallel training through policy initialization,
  timeout-aware critic targets and multi-step returns. Robots and settings
  not checked.
- **FastTD3 / FastSAC**, Seo et al., arXiv 2512.01996 (A): G1 and Booster T1
  walking in about 15 minutes on one RTX 4090 with strong randomization.
  The claim that PPO fails under strong pushes where these cope is not in
  the abstract (unchecked).

## 5. Sim-to-real practice from papers that deployed on a real G1

- **Domain randomization:** friction 0.1-2.0 (HOMIE) or 0.5-1.25 (FALCON);
  link mass 0.9-1.2x; torso payload about -5 to +10 kg; Kp/Kd ±10%; pushes
  0.5-1 m/s; joint position noise ±0.02 rad, velocity ±2 rad/s (V).
- **Control delay:** randomize 0-20 ms (FALCON, V). BeyondMimic saw failures
  from 5-10 ms of added latency and deploys in real-time C++ (V).
- **Model fidelity:** model joint armature (zero armature overswings);
  randomize joint offsets and torso center of mass (BeyondMimic, V); SONIC
  added per-motor Kp/Kd scaling (V).
- **Validation path:** train, then MuJoCo sim2sim, then real
  (unitree_rl_lab, AGILE; V).

## 6. Human motion data for more human-like walking

Only abstracts were reached for these (A) or they are from memory (R).
Licenses, sizes, formats and G1 retargeting support are all unverified.

| source | tag | fit | why |
|---|---|---|---|
| GMR, *Retargeting Matters*, arXiv 2510.02252 | A | high | retargets SMPL/BVH mocap to humanoid joint trajectories; G1 support (R) |
| BeyondMimic, arXiv 2508.08241 | A | high | Isaac Lab motion-tracking framework, G1 first-class (R) |
| LAFAN1 retargeted to G1 (Hugging Face) | R | high | ready G1 clips; LAFAN1 is CC BY-NC-ND (R) |
| AMASS | A | medium | largest mocap archive (SMPL); non-commercial terms (R) |
| PHUMA, arXiv 2510.26236 | A | medium | physically filtered locomotion data |
| TWIST, arXiv 2505.02833 | A | medium | mocap-suit teleop; useful for its tracking controller |
| UniTracker 2507.07356, OmniH2O, CMU mocap | A/R | low | method only, H1 not G1, or already inside AMASS |

Three ways walking data could enter training (our judgement):

1. **Motion tracking** (BeyondMimic): per-clip tracking policies. The output
   follows a reference phase, not a velocity command, so it would need
   distilling into a velocity-conditioned student.
2. **AMP style reward**: keep the velocity task reward and add a
   discriminator trained on G1-retargeted walking clips. Closest to the
   current setup.
3. **Reference statistics only**: small penalties on step width, arm swing
   or head height derived from the clips.

Suggested order: GMR + LAFAN1-G1 clips, then BeyondMimic on a few clips to
check the tracking loop in Isaac Lab, then an AMP reward on the velocity task.

## Sources

Rudin et al. 2109.11978 · Makoviychuk et al., Isaac Gym, NeurIPS 2021 D&B ·
Isaac Lab 2511.04831 · HOMIE 2502.13013 · FALCON 2505.06776 · AMO 2505.03738 ·
TWIST 2505.02833 · BeyondMimic 2508.08241 · SONIC 2511.07820 · AGILE 2603.20147 ·
RSL-RL-SAC 2605.24975 · FastSAC 2512.01996 · GMR 2510.02252 · PHUMA 2510.26236 ·
UniTracker 2507.07356 · github.com/nvidia-isaac/WBC-AGILE ·
github.com/NVlabs/GR00T-WholeBodyControl · github.com/unitreerobotics/unitree_rl_lab ·
Isaac Lab release notes and 3.0 migration guide.
