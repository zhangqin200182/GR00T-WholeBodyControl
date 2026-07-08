#!/usr/bin/env python3
"""Phase B Layer 0: Observation dimension and sanity checks."""
import sys, os, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env import MuJoCoEnv, ACTOR_DIM, CRITIC_DIM, TOKENIZER_DIM, NUM_DOF

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"


def test_dims():
    print("1. Observation dimensions")
    env = MuJoCoEnv(XML, PKL)
    obs = env.reset()
    assert obs["actor_obs"].shape == (ACTOR_DIM,), f"actor {obs['actor_obs'].shape}"
    assert obs["critic_obs"].shape == (CRITIC_DIM,), f"critic {obs['critic_obs'].shape}"
    assert obs["tokenizer"].shape == (TOKENIZER_DIM,), f"tokenizer {obs['tokenizer'].shape}"
    print(f"  actor={ACTOR_DIM} critic={CRITIC_DIM} tokenizer={TOKENIZER_DIM}: OK")


def test_no_nan_inf():
    print("2. No NaN / Inf")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    for i in range(50):
        o, r, d, info = env.step(np.random.uniform(-0.2, 0.2, NUM_DOF))
        for k, v in o.items():
            assert not np.any(np.isnan(v)), f"NaN in {k} at step {i}"
            assert not np.any(np.isinf(v)), f"Inf in {k} at step {i}"
    print("  50 steps, no NaN/Inf: OK")


def test_history_buffer():
    print("3. History buffer correctness")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    for _ in range(15):
        env.step(np.random.uniform(-0.2, 0.2, NUM_DOF))
    jph_last = env._jph[-1]
    qpos_now = env.data.qpos[7:]
    err = np.max(np.abs(jph_last - qpos_now))
    assert err < 1e-3, f"joint_pos_history mismatch: {err}"
    ah_last = env._ah[-1]
    assert np.max(np.abs(ah_last)) < 1.0, f"action_history out of range"
    gdh_last = env._gdh[-1]
    assert abs(np.linalg.norm(gdh_last) - 1.0) < 0.1, f"gravity not unit: {np.linalg.norm(gdh_last)}"
    print(f"  jph max_err={err:.6f}, ah range ok, gravity unit: OK")


def test_encoder_index():
    print("4. Encoder index one-hot")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    for _ in range(20):
        o, *_ = env.step(np.random.uniform(-0.2, 0.2, NUM_DOF))
        enc = o["tokenizer"][:3]
        active = np.sum(enc > 0.9)
        assert active == 1, f"encoder_index not one-hot: sum={active}, enc={enc}"
    print("  20 steps, all one-hot: OK")


def test_fk_consistency():
    print("5. FK reference body consistency")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    env._compute_ref_body_state()
    rp = env._ref_data.xpos[env._body_idx["pelvis"]]
    assert np.linalg.norm(rp) > 0.1, f"ref pelvis pos too small: {rp}"
    bp = env._ref_data.xpos[env._body_indices]
    assert not np.any(np.isnan(bp)), "NaN in ref body pos"
    print(f"  pelvis pos=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f}), {len(env._body_indices)} bodies: OK")


if __name__ == "__main__":
    print("Phase B: Observation verification")
    test_dims()
    test_no_nan_inf()
    test_history_buffer()
    test_encoder_index()
    test_fk_consistency()
    print("\nPhase B: PASS")
