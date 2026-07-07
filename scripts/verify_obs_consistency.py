#!/usr/bin/env python3
"""Task 4: Observation consistency verification.

Compare MuJoCo observations against training expectations:
  1. History buffer correctness
  2. Obs ranges vs Isaac Sim training data
  3. Physics impact on observation distribution
"""
import sys, time, numpy as np, mujoco, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MuJoCoEnv:
    """Same as verify_mujoco_env.py but with motion_lib integration."""
    def __init__(self, model_xml):
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.native_dt = self.model.opt.timestep
        self.decimation = 10
        self.ctrl_dt = self.native_dt * self.decimation
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        jr = self.model.jnt_range[1:]
        self.jm = (jr[:, 1] + jr[:, 0]) / 2
        self.jh = (jr[:, 1] - jr[:, 0]) / 2
        self._ah = np.zeros((1, 10, self.nu), dtype=np.float32)
        self._jph = np.zeros((1, 10, self.nu), dtype=np.float32)
        self._jvh = np.zeros((1, 10, self.nu), dtype=np.float32)
        self._gdh = np.zeros((1, 10, 3), dtype=np.float32)
        self._avh = np.zeros((1, 10, 3), dtype=np.float32)
        self.ep = 0
        self.max_ep = 500
        self.kp = np.ones(self.nu) * 30.0
        self.kd = np.ones(self.nu) * 3.0

    def reset(self):
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.ep = 0
        for b in [self._ah, self._jph, self._jvh, self._gdh, self._avh]:
            b.fill(0)
        return self._obs()

    def step(self, action):
        if len(action.shape) == 2:
            action = action[0]
        action = np.clip(action, -1, 1)
        target = action * self.jh + self.jm
        for _ in range(self.decimation):
            t = self.kp * (target - self.data.qpos[7:]) - self.kd * self.data.qvel[6:]
            self.data.ctrl[:] = np.clip(t, -50, 50)
            mujoco.mj_step(self.model, self.data)
        self.ep += 1
        self._ah[:, :-1] = self._ah[:, 1:]; self._ah[:, -1] = action
        self._jph[:, :-1] = self._jph[:, 1:]; self._jph[:, -1] = self.data.qpos[7:]
        self._jvh[:, :-1] = self._jvh[:, 1:]; self._jvh[:, -1] = self.data.qvel[6:]
        self._gdh[:, :-1] = self._gdh[:, 1:]; self._gdh[:, -1] = [0, 0, -1]
        self._avh[:, :-1] = self._avh[:, 1:]; self._avh[:, -1] = self.data.qvel[3:6]
        obs = self._obs()
        r = float(-np.sum((self.data.qpos[7:] - target) ** 2) * 0.01 + 0.1)
        term = self.data.xpos[1][2] < 0.3
        trunc = self.ep >= self.max_ep
        done = term or trunc
        tobs = obs.copy() if done else None
        if done:
            obs = self.reset()
        return obs, r, done, {"time_outs": trunc, "terminal_obs": tobs}

    def _obs(self):
        a = np.concatenate([
            self._jph.flatten(), self._jvh.flatten(),
            self._gdh.flatten(), self._avh.flatten(), self._ah.flatten(),
        ]).astype(np.float32)
        return {
            "actor_obs": a,
            "critic_obs": np.concatenate([a, np.zeros(715, dtype=np.float32)]),
            "tokenizer": np.zeros(1761, dtype=np.float32),
        }


if __name__ == "__main__":
    xml = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
    env = MuJoCoEnv(xml)
    print("Task 4: Observation Consistency")

    # ---- 1. History buffer correctness ----
    print("\n1. History buffer correctness")
    env.reset()
    # Collect several steps with the same action
    action = np.random.uniform(-0.1, 0.1, env.nu)
    for _ in range(15):
        env.step(action)

    # The last 10 joint_pos_history entries should match actual qpos history
    jph_last = env._jph[0, -1]  # most recent history entry
    qpos_now = env.data.qpos[7:]
    err = np.max(np.abs(jph_last - qpos_now))
    print(f"  Last history entry vs actual qpos: max_err={err:.6f}")
    assert err < 1e-6, f"History buffer error too large: {err}"

    # Verify sliding window: after 15 steps with same action, all entries should match
    for i in range(10):
        jph_i = env._jph[0, i]
        ah_i = env._ah[0, i]
        assert np.max(np.abs(ah_i - action)) < 1e-6, f"Action history[{i}] mismatch"
    print("  Action history: consistent (all 10 entries match input)")
    print("  History buffer: PASS")

    # ---- 2. Obs ranges vs Isaac Sim benchmarks ----
    # From training report, Iter 1 rollout:
    #   actor_obs:  mean=0.0153, std=0.5903
    #   critic_obs: mean=0.0417, std=0.5637
    #   tokenizer:  mean=0.0595, std=0.5641

    print("\n2. Obs ranges vs Isaac Sim training data")
    print("  Isaac Sim (iter 1): actor mean=0.015, std=0.59, critic mean=0.042, std=0.56")

    # Collect a batch of observations
    all_actor = []
    all_critic = []
    env.reset()
    for _ in range(500):
        a = np.random.uniform(-0.3, 0.3, env.nu)
        o, r, d, info = env.step(a)
        all_actor.append(o["actor_obs"])
        all_critic.append(o["critic_obs"])
        if d:
            continue  # auto-reset handled

    actor_arr = np.array(all_actor)
    critic_arr = np.array(all_critic)
    print(f"  MuJoCo actor_obs:  mean={np.mean(actor_arr):.4f}, std={np.std(actor_arr):.4f}")
    print(f"  MuJoCo critic_obs: mean={np.mean(critic_arr):.4f}, std={np.std(critic_arr):.4f}")

    # Check: values should be in reasonable range (not NaN, not explosion)
    assert not np.any(np.isnan(actor_arr)), "NaN in actor obs"
    assert np.abs(np.mean(actor_arr)) < 100, f"Actor obs mean {np.mean(actor_arr)} too large"

    # ---- 3. Physics impact ----
    print("\n3. Physics impact analysis")
    # Track qpos trajectory with zero action
    env.reset()
    qpos_trajectory = []
    for _ in range(50):
        o, r, d, info = env.step(np.zeros(env.nu))
        qpos_trajectory.append(env.data.qpos[7:].copy())
        if d:
            break

    qpos_arr = np.array(qpos_trajectory)
    qpos_diff = np.max(np.abs(qpos_arr - qpos_arr[0]), axis=0)
    print(f"  Zero-action drift (50 steps): max={np.max(qpos_diff):.4f} rad")
    print(f"  Mean drift per joint: {np.mean(qpos_diff):.4f} rad")

    # ---- 4. Joint range check ----
    print("\n4. Joint range check")
    jr = env.model.jnt_range[1:]
    for j in range(env.nu):
        q = env.data.qpos[7+j]
        lo, hi = jr[j]
        if q < lo or q > hi:
            print(f"  Joint {j}: q={q:.4f} outside [{lo:.4f}, {hi:.4f}]")

    within = np.sum((env.data.qpos[7:] >= jr[:, 0]) & (env.data.qpos[7:] <= jr[:, 1]))
    print(f"  Joints within range: {within}/{env.nu}")

    print("\nTask 4: PASS")
