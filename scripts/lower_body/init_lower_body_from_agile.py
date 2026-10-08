"""Write an rsl_rl checkpoint whose actor is the Agile locomotion policy, to warm-start lower-body PPO.

The lower-body env uses Agile's observation/action layout and the actor has its layer sizes and ELU
activations, so the weights copy over one to one. The critic and optimizer layout come from a donor
checkpoint (any run of the same task); the optimizer's moments are dropped and the action std reset.

    scripts/launch.sh scripts/lower_body/init_lower_body_from_agile.py \
        --donor logs/rsl_rl/g1_lower_body/<run>/model_299.pt --run_name agile_init
    scripts/launch.sh $CORE/tools/isaaclab_tool.py reinforcement_learning/rsl_rl/train.py --task ... \
        --resume --load_run agile_init --checkpoint model_0.pt
"""

import argparse
import time
from pathlib import Path

import torch

from humanoid_arc_core.robots.assets import AGILE_LOCOMOTION_POLICY

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--donor", required=True)
parser.add_argument("--agile", default=str(AGILE_LOCOMOTION_POLICY))
parser.add_argument("--run_name", default="agile_init")
parser.add_argument("--std", type=float, default=0.3)
parser.add_argument(
    "--height_col_std", type=float, default=0.0,
    help="If > 0, re-draw the first-layer weights of the height command (input 3) from N(0, std): Agile's are nearly zero, so gradients barely reach it.",
)
args = parser.parse_args()

ckpt = torch.load(args.donor, map_location="cpu", weights_only=False)
agile = torch.jit.load(args.agile, map_location="cpu").state_dict()
for k, v in agile.items():
    name = k.replace("actor.layers.", "mlp.")
    assert ckpt["actor_state_dict"][name].shape == v.shape, name
    ckpt["actor_state_dict"][name] = v.clone()
if args.height_col_std > 0:
    ckpt["actor_state_dict"]["mlp.0.weight"][:, 3] = torch.randn(256) * args.height_col_std
ckpt["actor_state_dict"]["distribution.std_param"].fill_(args.std)
ckpt["optimizer_state_dict"]["state"] = {}
ckpt["iter"] = 0
out = Path("logs/rsl_rl/g1_lower_body") / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
out.mkdir(parents=True)
torch.save(ckpt, out / "model_0.pt")
print(f"[INFO] wrote {out / 'model_0.pt'}")
