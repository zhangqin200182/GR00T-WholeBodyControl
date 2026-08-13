#!/usr/bin/env python3
"""3D MuJoCo EGL rendering of Phase D ref PD rollout."""
import os, sys
os.chdir("/")
sys.path.insert(0, "/")
os.environ["MUJOCO_GL"] = "egl"
os.environ["SONIC_MUJOCO_ENV"] = "1"

import numpy as np, mujoco, av

MODEL_XML = "/gear_sonic_deploy/g1/g1_29dof.xml"
PKL_DIR = "/sample_data/robot_filtered"
OUT_DIR = "/sonic-data/renders"
os.makedirs(OUT_DIR, exist_ok=True)
W, H = 1280, 720
FPS = 25

# ── Create env ──
from gear_sonic.envs.mujoco_env import MuJoCoEnv
env = MuJoCoEnv(MODEL_XML, PKL_DIR, config={"ignore_terminations": True, "alive_bonus": 0.5})

# ── Renderer ──
renderer = mujoco.Renderer(env.model, W, H)
camera = mujoco.MjvCamera()
camera.type = 1  # tracking
camera.trackbodyid = env._body_idx["pelvis"]
camera.distance = 3.0
camera.elevation = -15
camera.azimuth = 90

# ── Render episodes ──
for ep in range(3):
    out_path = f"{OUT_DIR}/phaseD_3D_ep{ep+1}.mp4"
    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=FPS)
    stream.width = W; stream.height = H
    stream.pix_fmt = "yuv420p"

    env.reset()
    total_rew = 0

    for step in range(300):
        # Ref PD action
        idx = int(env._ref_time * env._ref_fps)
        ref_q = env._ref_dof[idx % len(env._ref_dof)].astype(np.float64)
        action = np.clip((ref_q - env.jm) / env.jh, -1.0, 1.0)

        obs, rew, done, info = env.step(action.astype(np.float32))
        total_rew += rew

        renderer.update_scene(env.data, camera=camera)
        frame = av.VideoFrame.from_ndarray(renderer.render(), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

        if done and step > 5:
            break

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"  {out_path} (rew={total_rew:.0f}, steps={step+1})")

renderer.close()
print(f"Done. Files in {OUT_DIR}/")
