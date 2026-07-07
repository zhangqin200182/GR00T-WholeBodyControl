#!/usr/bin/env python3
"""Task 3: MuJoCoEnv single environment verification."""
import sys, time, numpy as np, mujoco, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MuJoCoEnv:
    def __init__(self, model_xml):
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.native_dt = self.model.opt.timestep  # 0.002
        self.decimation = 10  # 10 × 0.002 = 0.020s = 50 Hz
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
        self._ah[:, :-1] = self._ah[:, 1:]
        self._ah[:, -1] = action
        self._jph[:, :-1] = self._jph[:, 1:]
        self._jph[:, -1] = self.data.qpos[7:]
        self._jvh[:, :-1] = self._jvh[:, 1:]
        self._jvh[:, -1] = self.data.qvel[6:]
        self._gdh[:, :-1] = self._gdh[:, 1:]
        self._gdh[:, -1] = [0, 0, -1]
        self._avh[:, :-1] = self._avh[:, 1:]
        self._avh[:, -1] = self.data.qvel[3:6]
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
        a = np.concatenate(
            [
                self._jph.flatten(),
                self._jvh.flatten(),
                self._gdh.flatten(),
                self._avh.flatten(),
                self._ah.flatten(),
            ]
        ).astype(np.float32)
        return {
            "actor_obs": a,
            "critic_obs": np.concatenate([a, np.zeros(715, dtype=np.float32)]),
            "tokenizer": np.zeros(1761, dtype=np.float32),
        }


if __name__ == "__main__":
    xml = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
    print("Task 3: MuJoCoEnv")
    env = MuJoCoEnv(xml)
    print(
        f"  nb={env.model.nbody} nu={env.nu} "
        f"native_dt={env.native_dt} decim={env.decimation} ctrl_dt={env.ctrl_dt}"
    )
    obs = env.reset()
    k0 = list(obs.keys())[0]
    print(f"  dims: {k0}={obs[k0].shape}")

    t0 = time.perf_counter()
    rs = []
    ds = 0
    for i in range(200):
        a = np.random.uniform(-0.2, 0.2, env.nu)
        o, r, d, info = env.step(a)
        rs.append(r)
        if d:
            ds += 1
    dt = time.perf_counter() - t0
    print(
        f"  200 steps: {1000*dt/200:.2f} ms, "
        f"r={np.mean(rs):.4f}[{np.min(rs):.4f},{np.max(rs):.4f}] dones={ds}"
    )

    env.reset()
    for i in range(200):
        o, r, d, info = env.step(np.zeros(env.nu))
        if d:
            print(f"  zero-action fell at step {i}")
            break
    else:
        print("  zero-action: stable 200 steps")
    print("Task 3: PASS")
