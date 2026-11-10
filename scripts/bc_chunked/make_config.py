"""Write a robomimic 0.5 diffusion-policy config for the g1_pick_place dataset.

Run with the separate env: $ROBOMIMIC05_PYTHON scripts/bc_chunked/make_config.py OUT.json --epochs N
Observation keys, +-1.5cm uniform observation noise and batch size match the BC-RNN
`bc_obs_noise_h50` baseline, so only the policy class (action chunks via diffusion) differs.
"""
import argparse, json
from pathlib import Path

import robomimic

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--name", default="bc_chunked")
p.add_argument("--data", default="datasets/g1_pick_place_generated.hdf5")
p.add_argument("--output_dir", default="outputs/bc_chunked/runs")
p.add_argument("--save_every", type=int, default=50)
a = p.parse_args()

tpl = Path(robomimic.__file__).parent / "exps/templates/diffusion_policy.json"
c = json.load(open(tpl))
c["experiment"]["name"] = a.name
c["experiment"]["rollout"]["enabled"] = False
c["experiment"]["render_video"] = False
c["experiment"]["save"]["every_n_epochs"] = a.save_every
c["experiment"]["save"]["on_best_rollout_success_rate"] = False
c["train"]["data"] = a.data
c["train"]["output_dir"] = a.output_dir
c["train"]["num_epochs"] = a.epochs
c["train"]["hdf5_load_next_obs"] = False
c["train"]["hdf5_normalize_obs"] = True
c["train"]["dataset_keys"] = ["actions"]
c["train"]["action_config"] = {"actions": {"normalization": "min_max"}}
c["train"]["num_data_workers"] = 2
c["train"]["seed"] = 101
# DDIM (10 inference steps) instead of 100-step DDPM so 24 rollouts x 1000 steps finish in minutes
c["algo"]["ddpm"]["enabled"] = False
c["algo"]["ddim"]["enabled"] = True
c["train"]["frame_stack"] = c["algo"]["horizon"]["observation_horizon"]
c["train"]["seq_length"] = c["algo"]["horizon"]["prediction_horizon"]
c["observation"]["modalities"]["obs"]["low_dim"] = [
    "left_eef_pos", "left_eef_quat", "right_eef_pos", "right_eef_quat", "hand_joint_state", "object"]
c["observation"]["encoder"] = {"low_dim": {"obs_randomizer_class": "GaussianNoiseRandomizer",
    "obs_randomizer_kwargs": {"noise_std": 0.03, "noise_mean": -0.015, "num_samples": 1}}}
json.dump(c, open(a.out, "w"), indent=2)
