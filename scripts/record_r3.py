#!/usr/bin/env python3
"""Record the SONIC policy tracking a SPECIFIC motion (round-3 multi-motion test).

Same honest-mode pipeline as record_walk.py, plus:
  SONIC_MOTION_FILE=<path.pkl>  pick the motion by PKL file (name-based selection)
Others (SONIC_STOCH / SONIC_REC_NO_TERM / SONIC_NOCLIP ...) identical to record_walk.py.
"""
import os
import sys
from importlib import util as iu

os.environ.setdefault("MUJOCO_GL", "cgl")

import imageio.v2 as imageio  # noqa: E402
import joblib  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

spec = iu.spec_from_file_location(
    "tms", os.path.join(REPO_ROOT, "scripts", "train_mujoco_sonic.py")
)
tms = iu.module_from_spec(spec)
spec.loader.exec_module(tms)

from gear_sonic.envs.mujoco_env import MuJoCoEnv  # noqa: E402

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
out = sys.argv[2] if len(sys.argv) > 2 else "/Users/max/code/ai/embody/g1_r3.mp4"
no_term = os.environ.get("SONIC_REC_NO_TERM") == "1"
stochastic = os.environ.get("SONIC_STOCH") == "1"
novideo = os.environ.get("SONIC_NOVIDEO") == "1"
motion_file = os.environ.get("SONIC_MOTION_FILE")

env = MuJoCoEnv(model_xml=tms.XML, pkl_dir=tms.PKL,
                config={"ignore_terminations": no_term})
if motion_file:
    v = joblib.load(motion_file)
    entry = v[next(k for k in v if isinstance(v[k], dict) and "dof" in v[k])]
    env.motions = [entry]
    print(f"[motion] {os.path.basename(motion_file)}: "
          f"{len(entry['dof'])/entry['fps']:.1f}s")
actor, _, _ = tms.build_model_from_release_config("cpu")
ckpt = torch.load(tms.CKPT, map_location="cpu", weights_only=False)
actor.load_state_dict(ckpt["policy_state_dict"], strict=False)
actor.eval()
actor.init_rollout()

renderer = mujoco.Renderer(env.model, 480, 640)
cam = mujoco.MjvCamera()
cam.distance = 3.5
cam.azimuth = 100.0
cam.elevation = -10.0

pelvis = env._body_idx["pelvis"]
obs = env.reset()
path = 0.0
prev_xy = env.data.xpos[pelvis][:2].copy()
writer = imageio.get_writer(out, fps=50, quality=8)
motion_len = len(env._ref_dof) / env._ref_fps
fell_at = None
speeds = []
ref_speeds = []
for step in range(n_steps):
    with torch.no_grad():
        act_out = actor.act_inference(
            {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in obs.items()},
            cur_dones=None,
        )
    if stochastic and hasattr(actor, "distribution") and actor.distribution is not None:
        a = act_out.sample().squeeze(0).numpy()
    else:
        a = act_out.squeeze(0).numpy()
    if os.environ.get("SONIC_NOCLIP") != "1":
        a = np.clip(a, -1.0, 1.0)
    obs, r, done, info = env.step(a)

    xy = env.data.xpos[pelvis][:2]
    h = float(env.data.xpos[pelvis][2])
    if step > 0:
        speeds.append(float(np.linalg.norm(xy - prev_xy)) / env.ctrl_dt)
    nxt_t = min(env._ref_time + env.ctrl_dt, motion_len - 1e-3)
    nxt_idx = min(int(nxt_t * env._ref_fps), len(env._ref_root_trans) - 1)
    cur_idx = min(int(env._ref_time * env._ref_fps), len(env._ref_root_trans) - 1)
    ref_speeds.append(
        float(np.linalg.norm(env._ref_root_trans[nxt_idx][:2]
                             - env._ref_root_trans[cur_idx][:2])) / env.ctrl_dt
    )
    if h > 0.45:
        path += float(np.linalg.norm(xy - prev_xy))
    prev_xy = xy.copy()

    cam.lookat[:] = env.data.xpos[pelvis]
    if not novideo:
        renderer.update_scene(env.data, camera=cam)
        writer.append_data(renderer.render())
    if step % 200 == 199:
        print(f"step {step+1}/{n_steps}, path {path:.1f} m, pelvis_h {h:.2f}, "
              f"v {np.mean(speeds[-100:]):.2f} m/s (ref {np.mean(ref_speeds[-100:]):.2f})")
    if done and not no_term:
        fell_at = step + 1
        print(f"episode terminated at step {fell_at} (pelvis_h {h:.2f})")
        break
if not novideo:
    writer.close()
    print(f"saved {out}: {len(open(out,'rb').read())//1024} KB, {step+1} steps, "
          f"path {path:.2f} m, fell_at {'none' if fell_at is None else fell_at}")
else:
    print(f"[novideo] {step+1} steps, path {path:.2f} m, "
          f"fell_at {'none' if fell_at is None else fell_at}")
print(f"mean speed {np.mean(speeds):.2f} m/s vs ref {np.mean(ref_speeds):.2f} m/s; "
      f"min pelvis height {min(0.45, h):.2f}")
