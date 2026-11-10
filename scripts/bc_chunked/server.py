"""Policy server for robomimic 0.5 checkpoints (runs in the robomimic 0.5 venv, not in Isaac's python).

Started by scripts/bc/eval_bc.py --backend server, and connects back to its Listener. Protocol (pickled over
multiprocessing.connection): first message ("init", n_envs) -> reply ("ok", obs_keys);
then ("reset",) -> start_episode on every policy; ("act", {key: array[n, ...]}) -> array[n, 32].
"""
import argparse
from collections import deque
from multiprocessing.connection import Client

import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--port", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed)
dev = torch.device("cuda")
conn = Client(("127.0.0.1", a.port), authkey=b"bc_chunked")
policies, keys, hist, To = [], None, [], 2
while True:
    msg = conn.recv()
    if msg[0] == "init":
        policies = [FileUtils.policy_from_checkpoint(ckpt_path=a.checkpoint, device=dev)[0] for _ in range(msg[1])]
        keys = list(policies[0].policy.global_config.observation.modalities.obs.low_dim)
        To = policies[0].policy.algo_config.horizon.observation_horizon
        hist = [deque(maxlen=To) for _ in policies]
        conn.send(("ok", keys))
    elif msg[0] == "reset":
        for q in policies:
            q.start_episode()
        for h in hist:
            h.clear()
        conn.send(("ok",))
    elif msg[0] == "act":
        obs = msg[1]
        out = []
        for i, q in enumerate(policies):
            h = hist[i]
            cur = {k: obs[k][i] for k in keys}
            if not h:  # episode start: repeat the first frame (training used pad_frame_stack=True)
                h.extend([cur] * To)
            else:
                h.append(cur)
            out.append(np.asarray(q({k: np.stack([f[k] for f in h]) for k in keys})))
        conn.send(np.stack(out))
    elif msg[0] == "quit":
        break
