"""Convert an old-format rsl_rl checkpoint (single ActorCritic `model_state_dict`)
to the rsl-rl-lib >= 5 format (separate `actor_state_dict` / `critic_state_dict`).

IsaacLab's published pretrained checkpoints (`--use_pretrained_checkpoint`)
are still in the old format and fail under the installed rsl-rl-lib 5.0.1
with `KeyError: 'actor_state_dict'`. Weights, parameter order, and optimizer
state are unchanged; only the keys move.

    scripts/launch.sh scripts/locomotion/convert_rsl_rl_checkpoint.py IN.pt OUT.pt
"""

import sys
from pathlib import Path

import torch

OLD_TO_NEW = {
    "std": ("actor", "distribution.std_param"),
    "log_std": ("actor", "distribution.log_std_param"),
    "actor_obs_normalizer.": ("actor", "obs_normalizer."),
    "critic_obs_normalizer.": ("critic", "obs_normalizer."),
    "actor.": ("actor", "mlp."),
    "critic.": ("critic", "mlp."),
}


def convert(old: dict) -> dict:
    if "actor_state_dict" in old:
        raise SystemExit("already in the new format")
    new = {"actor": {}, "critic": {}}
    for key, value in old["model_state_dict"].items():
        for prefix, (model, new_prefix) in OLD_TO_NEW.items():
            if key == prefix or (prefix.endswith(".") and key.startswith(prefix)):
                new[model][new_prefix + key[len(prefix):]] = value
                break
        else:
            raise SystemExit(f"unmapped key: {key}")
    out = {k: v for k, v in old.items() if k != "model_state_dict"}
    out["actor_state_dict"] = new["actor"]
    out["critic_state_dict"] = new["critic"]
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    converted = convert(torch.load(src, map_location="cpu", weights_only=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, dst)
    for model in ("actor", "critic"):
        print(f"{model}: {', '.join(converted[f'{model}_state_dict'])}")
    print(f"wrote {dst}")
