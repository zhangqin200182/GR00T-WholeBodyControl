"""MuJoCo environment for SONIC training — single env, ISAAC-SIM compatible."""
import os, glob, numpy as np, mujoco, joblib

NUM_DOF = 29
NUM_FUTURE = 10  # num_future_frames from SONIC config


class MuJoCoEnv:
    """Single MuJoCo G1 environment with motion reference tracking."""

    def __init__(self, model_xml, pkl_dir, config=None):
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
        self.max_ep = getattr(config, "max_episode_length", 500) if config else 500
        self.kp = np.ones(self.nu) * 30.0
        self.kd = np.ones(self.nu) * 3.0
        # Load motion data
        self.motions = []
        for p in sorted(glob.glob(os.path.join(pkl_dir, "**/*.pkl"), recursive=True)):
            if os.path.basename(p).startswith("._"):
                continue
            data = joblib.load(p)
            for v in data.values():
                if isinstance(v, dict) and "dof" in v:
                    self.motions.append(v)

    def reset(self):
        m = self.motions[np.random.randint(len(self.motions))]
        dof = m["dof"]
        start = np.random.randint(0, max(1, len(dof) - self.max_ep))
        self._ref_dof = dof
        self._ref_start = start
        self._ref_idx = start
        self.data.qpos[7:] = dof[start].astype(np.float64)
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
        self._ref_idx = getattr(self, "_ref_idx", 0) + 1
        self._ah[:, :-1] = self._ah[:, 1:]; self._ah[:, -1] = action
        self._jph[:, :-1] = self._jph[:, 1:]; self._jph[:, -1] = self.data.qpos[7:]
        self._jvh[:, :-1] = self._jvh[:, 1:]; self._jvh[:, -1] = self.data.qvel[6:]
        self._gdh[:, :-1] = self._gdh[:, 1:]; self._gdh[:, -1] = [0, 0, -1]
        self._avh[:, :-1] = self._avh[:, 1:]; self._avh[:, -1] = self.data.qvel[3:6]
        obs = self._obs()
        ref = self._ref_dof[min(self._ref_idx, len(self._ref_dof) - 1)]
        r = float(-np.sum((self.data.qpos[7:] - ref) ** 2) * 0.01 + 0.1)
        term = self.data.xpos[1][2] < 0.3
        trunc = self.ep >= self.max_ep
        done = term or trunc
        tobs = obs.copy() if done else None
        if done:
            obs = self.reset()
        return obs, r, done, {"time_outs": trunc, "terminal_obs": tobs}

    # ---- Observation helpers ----
    def _future_dof(self):
        """Get future NUM_FUTURE frames from motion reference."""
        idx = self._ref_idx
        dof = self._ref_dof
        n = len(dof)
        indices = np.clip(np.arange(idx, idx + NUM_FUTURE), 0, n - 1)
        return dof[indices].astype(np.float32)  # (10, 29)

    def _obs(self):
        actor = np.concatenate([
            self._jph.flatten(), self._jvh.flatten(),
            self._gdh.flatten(), self._avh.flatten(), self._ah.flatten(),
        ]).astype(np.float32)  # 930

        # Critic obs: actor + motion reference targets + contact placeholder
        future_dof = self._future_dof()  # (10, 29)
        motion_targets = np.concatenate([future_dof.flatten(), np.zeros(17 + 700, dtype=np.float32)])
        critic = np.concatenate([actor, motion_targets[:715]]).astype(np.float32)  # 1645

        # Tokenizer obs: minimal encoder input from motion reference
        tokenizer = self._build_tokenizer(actor).astype(np.float32)  # 1761

        return {"actor_obs": actor, "critic_obs": critic, "tokenizer": tokenizer}

    def _build_tokenizer(self, actor):
        """Minimal tokenizer obs for encoder compatibility.
        Key field: command_multi_future_nonflat (flat joints ×NUM_FUTURE frames).
        """
        future = self._future_dof()  # (10, 29)
        # command_multi_future_nonflat: joints + zero velocity → (10, 58)
        cmd_future = np.concatenate([future, np.zeros_like(future)], axis=-1).flatten()  # 580
        # Pad to 1761 total
        result = np.zeros(1761, dtype=np.float32)
        result[:580] = cmd_future
        # encoder_index: random g1/teleop/smpl (uniform for now)
        result[580] = np.random.randint(3)
        return result
