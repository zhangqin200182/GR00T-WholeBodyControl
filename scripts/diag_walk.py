#!/usr/bin/env python3
"""Zero-shot walking diagnostics: single env, SONIC policy, log every fall.

Usage: python scripts/diag_walk.py [clip] [ema_alpha] [beta] [n_steps]
  clip: action saturation bound (Isaac trained under ±1 saturation)
  ema_alpha: low-pass factor, 0 = off (a_t = α·a_t + (1-α)·a_{t-1})
  beta: action amplitude scale (1.0 = full; smaller = smaller strides, slower gait)
"""
import os
import sys
from importlib import util as iu

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

spec = iu.spec_from_file_location(
    "tms", os.path.join(REPO_ROOT, "scripts", "train_mujoco_sonic.py")
)
tms = iu.module_from_spec(spec)
spec.loader.exec_module(tms)

from gear_sonic.envs.mujoco_env import MuJoCoEnv  # noqa: E402

clip = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
ema = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
beta = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

env = MuJoCoEnv(model_xml=tms.XML, pkl_dir=tms.PKL,
                config={"ignore_terminations": True})
actor, _, _ = tms.build_model_from_release_config("cpu")
ckpt = torch.load(tms.CKPT, map_location="cpu", weights_only=False)
actor.load_state_dict(ckpt["policy_state_dict"], strict=False)
actor.eval()
actor.init_rollout()

obs = env.reset()
pelvis = env._body_idx["pelvis"]
a_prev = None
falls = 0
ep_step = 0
path = 0.0
prev_xy = env.data.xpos[pelvis][:2].copy()
print(f"=== clip={clip} ema={ema} beta={beta} ===")
for step in range(n_steps):
    with torch.no_grad():
        a = actor.act_inference(
            {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in obs.items()},
            cur_dones=None,
        ).squeeze(0).numpy()
    a = np.clip(a, -clip, clip) * beta
    if ema > 0:
        a = ema * a + (1 - ema) * (a_prev if a_prev is not None else a)
        a_prev = a.copy()
    obs, r, done, info = env.step(a)
    ep_step += 1
    h = env.data.xpos[pelvis][2]
    xy = env.data.xpos[pelvis][:2]
    upright = h > 0.45  # real-fall threshold: pelvis below 45cm
    if upright:
        path += float(np.linalg.norm(xy - prev_xy))
    prev_xy = xy.copy()
    if not upright:
        falls += 1
        ep_step = 0
    if step % 500 == 499:
        print(f"  step {step+1}: root_h={h:.3f} path={path:.2f}m falls={falls}")
print(f"TOTAL: {falls} real falls / {n_steps} steps, {path:.1f} m walked "
      f"({path/(n_steps*0.02):.2f} m/s)")
