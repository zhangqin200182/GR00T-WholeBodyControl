"""MuJoCo G1 environment — full SONIC-compatible observation/reward/termination."""
import os, glob, numpy as np, mujoco, joblib
from gear_sonic.envs.mujoco_math import (
    quat_mul, quat_inv, quat_apply, quat_error_magnitude,
    quat_to_matrix, subtract_frame_transforms, quat_diff_to_angvel,
)

# ── Constants ──────────────────────────────────────────────────────────
NUM_DOF = 29
NUM_FUTURE = 10
NUM_SMPL_FUTURE = 10
FUTURE_DT_REF = 0.1     # dt_future_ref_frames from StubEnv (seconds between future frames)

BODY_NAMES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link",
    "right_elbow_link", "right_wrist_yaw_link",
)
VR_3POINT_BODY = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_3POINT_OFFSETS = np.array([[0.18, -0.025, 0.0], [0.18, 0.025, 0.0],
                               [0.0, 0.0, 0.35]], dtype=np.float32)
LOWER_JOINT_INDICES = list(range(12))
ACTOR_DIM = 930; CRITIC_DIM = 1645; TOKENIZER_DIM = 1761
HIST = 10  # history_length


class MuJoCoEnv:
    """Single MuJoCo G1 environment — SONIC-compatible obs/reward/termination."""

    def __init__(self, model_xml, pkl_dir, config=None):
        # ── MuJoCo simulation model ──
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)
        # ── FK reference model (independent kinematics) ──
        self._ref_model = mujoco.MjModel.from_xml_path(model_xml)
        self._ref_data = mujoco.MjData(self._ref_model)
        # ── Simulation params ──
        self.native_dt = self.model.opt.timestep   # 0.002
        self.decimation = 10; self.ctrl_dt = self.native_dt * self.decimation
        # ── Action space ──
        # Use Isaac Sim joint limits (from WBC config g1_29dof_sonic_model12.yaml),
        # not MuJoCo XML jnt_range.  Pretrained SONIC policy learned actions under
        # Isaac Sim normalization; mismatched jm/jh maps actions to wrong joint angles.
        self.nu = self.model.nu
        self.jm = np.array([
             0.1745,  1.2217,  0.0000,  1.3963, -0.1745,  0.0000,   # left  leg
             0.1745, -1.2217,  0.0000,  1.3963, -0.1745,  0.0000,   # right leg
             0.0000,  0.0000,  0.0000,                                # waist
            -0.2094,  0.3317,  0.0000,  0.5236,  0.0000,  0.0000,  0.0000,  # left  arm
            -0.2094, -0.3317,  0.0000,  0.5236,  0.0000,  0.0000,  0.0000,  # right arm
        ], dtype=np.float64)
        self.jh = np.array([
            2.7052, 1.7453, 2.7576, 1.4835, 0.6981, 0.2618,  # left  leg
            2.7052, 1.7453, 2.7576, 1.4835, 0.6981, 0.2618,  # right leg
            2.6180, 0.5200, 0.5200,                            # waist
            2.8798, 1.9199, 2.6180, 1.5708, 1.9722, 1.6144, 1.6144,  # left  arm
            2.8798, 1.9199, 2.6180, 1.5708, 1.9722, 1.6144, 1.6144,  # right arm
        ], dtype=np.float64)
        # ── PD gains ──
        self.kp = np.ones(self.nu) * 100.0; self.kd = np.ones(self.nu) * 5.0
        # ── Per-joint torque limits ──
        # actuator_forcelimited=False in this XML → parse joint actuatorfrcrange
        torque = np.ones(self.nu) * 50.0  # default fallback
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]  # joint id for this actuator
            if jid >= 0 and self.model.jnt_actfrcrange is not None:
                hi = self.model.jnt_actfrcrange[jid][1]
                if hi > 1e-6: torque[i] = hi
        self._torque_limit = torque
        # ── MuJoCo solver: more iterations for QACC stability ──
        self.model.opt.iterations = 200
        # ── Body indices ──
        self._body_idx = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in BODY_NAMES}
        self._body_indices = np.array([self._body_idx[n] for n in BODY_NAMES])
        # ── Episode ──
        self.ep = 0
        self.max_ep = getattr(config, "max_episode_length", 500) if config else 500
        self.ignore_terminations = getattr(config, "ignore_terminations", False) if config else False
        # ── Motions ──
        self.motions = self._load_motions(pkl_dir)
        # ── History buffers ──
        self._init_history()
        # ── Prev frame cache ──
        self._prev_action = np.zeros(self.nu, dtype=np.float32)
        self._prev_root_pos = np.zeros(3, dtype=np.float64)
        self._prev_body_pos = np.zeros((14, 3), dtype=np.float64)
        self._prev_body_quat = np.zeros((14, 4), dtype=np.float64)
        self._prev_ref_body_pos = np.zeros((14, 3), dtype=np.float64)
        self._prev_ref_body_quat = np.zeros((14, 4), dtype=np.float64)
        self._prev_joint_vel = np.zeros(self.nu, dtype=np.float64)

    # ═══════════════════════════════════════════════════════════════════
    # Motion loading
    # ═══════════════════════════════════════════════════════════════════
    def _load_motions(self, pkl_dir):
        motions = []
        for p in sorted(glob.glob(os.path.join(pkl_dir, "**/*.pkl"), recursive=True)):
            if os.path.basename(p).startswith("._"): continue
            for v in joblib.load(p).values():
                if isinstance(v, dict) and "dof" in v: motions.append(v)
        if not motions: raise RuntimeError(f"No motion PKLs in {pkl_dir}")
        return motions

    def _sample_motion(self):
        m = self.motions[np.random.randint(len(self.motions))]
        dof = m["dof"]; n = len(dof)
        self._ref_dof = dof
        self._ref_root_rot = m["root_rot"].astype(np.float64)       # (T, 4) quat [x,y,z,w]
        self._ref_root_trans = m["root_trans_offset"].astype(np.float64)  # (T, 3)
        self._ref_fps = m.get("fps", 30.0)
        self._ref_dt = 1.0 / self._ref_fps
        max_time = (n - self.max_ep - 1) * self._ref_dt
        self._ref_time = np.random.uniform(0, max(0.001, max_time))

    def _advance_motion_time(self):
        self._ref_time += self.ctrl_dt

    def _future_dof(self, field="dof", n=NUM_FUTURE, dt_ref=FUTURE_DT_REF):
        dof = self._ref_dof; fps = self._ref_fps; end = len(dof)
        times = self._ref_time + np.arange(n) * dt_ref
        indices = np.clip((times * fps).astype(int), 0, end - 1)
        return dof[indices].astype(np.float32)

    def _future_dof_vel(self, n=NUM_FUTURE, dt_ref=FUTURE_DT_REF):
        dof = self._ref_dof; fps = self._ref_fps; end = len(dof)
        times = self._ref_time + np.arange(n) * dt_ref
        t0 = np.clip((times * fps).astype(int), 0, end - 2)
        t1 = np.clip(t0 + 1, 0, end - 1)
        return ((dof[t1] - dof[t0]) / self._ref_dt).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # FK reference body state
    # ═══════════════════════════════════════════════════════════════════
    def _compute_ref_body_state(self):
        idx = min(int(self._ref_time * self._ref_fps), len(self._ref_dof) - 1)
        self._ref_data.qpos[7:] = self._ref_dof[idx].astype(np.float64)
        self._ref_data.qpos[:3] = self._ref_root_trans[idx]
        # PKL quat is [x,y,z,w] → MuJoCo is [w,x,y,z]
        pk = self._ref_root_rot[idx]
        self._ref_data.qpos[3:7] = [pk[3], pk[0], pk[1], pk[2]]
        self._ref_data.qvel[:] = 0
        mujoco.mj_kinematics(self._ref_model, self._ref_data)

    def _ref_root_pos(self):
        self._compute_ref_body_state()
        return self._ref_data.xpos[self._body_idx["pelvis"]].copy()

    def _ref_root_quat(self):
        self._compute_ref_body_state()
        return self._ref_data.xquat[self._body_idx["pelvis"]].copy()

    def _ref_body_pos(self):
        self._compute_ref_body_state()
        return self._ref_data.xpos[self._body_indices].copy()

    def _ref_body_quat(self):
        self._compute_ref_body_state()
        return self._ref_data.xquat[self._body_indices].copy()

    def _future_ref_root_pos(self, n=NUM_FUTURE, dt_ref=FUTURE_DT_REF):
        fps = self._ref_fps; end = len(self._ref_dof)
        times = self._ref_time + np.arange(n) * dt_ref
        indices = np.clip((times * fps).astype(int), 0, end - 1)
        return self._ref_root_trans[indices].astype(np.float32)

    def _future_ref_root_quat(self, n=NUM_FUTURE, dt_ref=FUTURE_DT_REF):
        fps = self._ref_fps; end = len(self._ref_dof)
        times = self._ref_time + np.arange(n) * dt_ref
        indices = np.clip((times * fps).astype(int), 0, end - 1)
        pk = self._ref_root_rot[indices]                      # (n, 4) [x,y,z,w]
        ret = np.zeros((n, 4), dtype=np.float32)
        ret[:, 0] = pk[:, 3]; ret[:, 1:] = pk[:, :3]         # → [w,x,y,z]
        return ret

    # ═══════════════════════════════════════════════════════════════════
    # Physics
    # ═══════════════════════════════════════════════════════════════════
    def _pd_control(self, action):
        target = action * self.jh + self.jm
        torque = self.kp * (target - self.data.qpos[7:]) - self.kd * self.data.qvel[6:]
        self.data.ctrl[:] = np.clip(torque, -self._torque_limit, self._torque_limit)

    def _physics_step(self):
        for _ in range(self.decimation): mujoco.mj_step(self.model, self.data)

    # ═══════════════════════════════════════════════════════════════════
    # History buffers
    # ═══════════════════════════════════════════════════════════════════
    def _init_history(self):
        self._gdh = np.zeros((HIST, 3), dtype=np.float32)
        self._avh = np.zeros((HIST, 3), dtype=np.float32)
        self._jph = np.zeros((HIST, self.nu), dtype=np.float32)
        self._jvh = np.zeros((HIST, self.nu), dtype=np.float32)
        self._ah  = np.zeros((HIST, self.nu), dtype=np.float32)
        self._lvh = np.zeros((HIST, 3), dtype=np.float32)

    def _shift_histories(self):
        for b in [self._gdh, self._avh, self._jph, self._jvh, self._lvh]:
            b[:-1] = b[1:]

    # ═══════════════════════════════════════════════════════════════════
    # Observations
    # ═══════════════════════════════════════════════════════════════════
    def _obs(self):
        return {"actor_obs": self._compute_actor_obs(),
                "critic_obs": self._compute_critic_obs(),
                "tokenizer": self._build_tokenizer()}

    def _compute_actor_obs(self):
        root_quat = self.data.xquat[self._body_idx["pelvis"]]
        g_body = quat_apply(quat_inv(root_quat), np.array([0, 0, -1]))
        ang_vel = self.data.qvel[3:6].copy()
        jpos = self.data.qpos[7:].copy(); jvel = self.data.qvel[6:].copy()
        self._shift_histories()
        self._gdh[-1] = g_body; self._avh[-1] = ang_vel
        self._jph[-1] = jpos; self._jvh[-1] = jvel
        return np.concatenate([b.flatten() for b in [self._gdh, self._avh, self._jph, self._jvh, self._ah]]).astype(np.float32)

    def _compute_critic_obs(self):
        root_pos = self.data.xpos[self._body_idx["pelvis"]]
        root_quat = self.data.xquat[self._body_idx["pelvis"]]
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()

        future_pos = self._future_dof(); future_vel = self._future_dof_vel()
        cmd_mf = np.concatenate([future_pos.flatten(), future_vel.flatten()])  # 580
        # anchor pos/ori in robot frame
        pos_b, ori_b = subtract_frame_transforms(root_pos, root_quat, ref_root_pos, ref_root_quat)
        mat = quat_to_matrix(ori_b); ori_6d = mat[..., :2].flatten()  # 6
        # body pos in robot frame
        body_pos_w = self.data.xpos[self._body_indices]
        body_pos_b = quat_apply(quat_inv(root_quat), body_pos_w - root_pos).flatten()  # 42
        # body ori in robot frame (6D)
        body_quat_w = self.data.xquat[self._body_indices]
        body_quat_b = quat_mul(np.tile(quat_inv(root_quat), (14, 1)), body_quat_w)
        ori_parts = [quat_to_matrix(q)[:2].flatten() for q in body_quat_b]
        body_ori_flat = np.concatenate(ori_parts)  # 84
        # lin vel history
        lin_vel = (root_pos - self._prev_root_pos) / self.ctrl_dt
        self._lvh[-1] = lin_vel; self._prev_root_pos = root_pos.copy()

        parts = [cmd_mf, pos_b.flatten(), ori_6d, body_pos_b, body_ori_flat] + \
                [b.flatten() for b in [self._lvh, self._avh, self._jph, self._jvh, self._ah]]
        return np.concatenate(parts).astype(np.float32)

    def _build_tokenizer(self):
        root_quat = self.data.xquat[self._body_idx["pelvis"]]
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()

        future_pos = self._future_dof(); future_vel = self._future_dof_vel()
        cmd_nonflat = np.concatenate([future_pos, future_vel], axis=-1).flatten()  # 580
        future_rp = self._future_ref_root_pos(); cmd_z_mf = future_rp[:, 2:3].flatten()  # 10
        future_rq = self._future_ref_root_quat()
        root_q_e = np.tile(root_quat, (NUM_FUTURE, 1))
        rot_diff = quat_mul(quat_inv(root_q_e), future_rq)
        ori_mf = np.stack([quat_to_matrix(q) for q in rot_diff])[:, :2, :].flatten()  # 60
        lower_pos = future_pos[:, LOWER_JOINT_INDICES]; lower_vel = future_vel[:, LOWER_JOINT_INDICES]
        cmd_lower = np.concatenate([lower_pos.flatten(), lower_vel.flatten()])  # 240

        # VR 3-point
        ref_body_pos = self._ref_body_pos(); ref_body_quat = self._ref_body_quat()
        vr_idx = [list(BODY_NAMES).index(n) for n in VR_3POINT_BODY]
        vr_pos = ref_body_pos[vr_idx]; vr_quat = ref_body_quat[vr_idx]
        vr_w = vr_pos + quat_apply(vr_quat, VR_3POINT_OFFSETS)
        ref_rq_3p = np.tile(ref_root_quat, (3, 1))
        vr_local = quat_apply(quat_inv(ref_rq_3p), vr_w - ref_root_pos).flatten()  # 9
        vr_orn = quat_mul(quat_inv(ref_rq_3p), vr_quat).flatten()  # 12

        # Single-frame anchor ori
        _, ori_b = subtract_frame_transforms(
            self.data.xpos[self._body_idx["pelvis"]], root_quat, ref_root_pos, ref_root_quat)
        mat_s = quat_to_matrix(ori_b); ori_b_6d = mat_s[..., :2].flatten()  # 6
        cmd_z = np.array([ref_root_pos[2]], dtype=np.float32)  # 1

        # Encoder index + SMPL (zeros)
        enc_idx = np.zeros(3, dtype=np.float32); enc_idx[0] = 1.0  # g1
        smpl_j = np.zeros(NUM_SMPL_FUTURE * 24 * 3, dtype=np.float32)   # 720
        smpl_o = np.zeros(NUM_SMPL_FUTURE * 6, dtype=np.float32)        # 60
        smpl_w = np.zeros(NUM_SMPL_FUTURE * 6, dtype=np.float32)        # 60

        return np.concatenate([enc_idx, cmd_nonflat, cmd_z_mf, ori_mf, cmd_lower,
                               vr_local, vr_orn, ori_b_6d, cmd_z,
                               smpl_j, smpl_o, smpl_w]).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # Reward
    # ═══════════════════════════════════════════════════════════════════
    def _compute_reward(self, action):
        root_pos = self.data.xpos[self._body_idx["pelvis"]]
        root_quat = self.data.xquat[self._body_idx["pelvis"]]
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()

        # 1. tracking_anchor_pos (w=0.5, σ=0.3)
        err = np.linalg.norm(root_pos - ref_root_pos)
        r1 = 0.5 * np.exp(-err**2 / 0.09)
        # 2. tracking_anchor_ori (w=0.5, σ=0.4)
        r2 = 0.5 * np.exp(-quat_error_magnitude(ref_root_quat, root_quat)**2 / 0.16)
        # 3. tracking_relative_body_pos (w=1.0, σ=0.3)
        ref_body_pos = self._ref_body_pos()
        ref_aligned = ref_body_pos - ref_root_pos + root_pos
        body_pos_w = self.data.xpos[self._body_indices]
        r3 = 1.0 * np.exp(-np.sum((body_pos_w - ref_aligned)**2, axis=-1).mean() / 0.09)
        # 4. tracking_relative_body_ori (w=1.0, σ=0.4)
        ref_body_quat = self._ref_body_quat()
        body_quat_w = self.data.xquat[self._body_indices]
        ang_errs = np.array([quat_error_magnitude(ref_body_quat[i], body_quat_w[i]) for i in range(14)])
        r4 = 1.0 * np.exp(-(ang_errs**2).mean() / 0.16)
        # 5. tracking_body_linvel (w=1.0, σ=1.0)
        body_lin_vel = (body_pos_w - self._prev_body_pos) / self.ctrl_dt
        ref_lin_vel = (ref_body_pos - self._prev_ref_body_pos) / self.ctrl_dt
        r5 = 1.0 * np.exp(-np.sum((body_lin_vel - ref_lin_vel)**2, axis=-1).mean() / 1.0)
        self._prev_body_pos = body_pos_w.copy(); self._prev_ref_body_pos = ref_body_pos.copy()
        # 6. tracking_body_angvel (w=1.0, σ=3.14)
        body_ang_vel = quat_diff_to_angvel(self._prev_body_quat, body_quat_w, self.ctrl_dt)
        ref_ang_vel = quat_diff_to_angvel(self._prev_ref_body_quat, ref_body_quat, self.ctrl_dt)
        r6 = 1.0 * np.exp(-np.sum((body_ang_vel - ref_ang_vel)**2, axis=-1).mean() / 9.86)
        self._prev_body_quat = body_quat_w.copy(); self._prev_ref_body_quat = ref_body_quat.copy()
        # 7. action_rate_l2 (w=-0.1)
        r7 = -0.1 * np.sum((action - self._prev_action)**2); self._prev_action = action.copy()
        # 8. joint_limit (w=-10.0)
        q = self.data.qpos[7:]
        r8 = -10.0 * np.sum(np.maximum(np.abs(q - self.jm) - self.jh, 0))
        # 9. undesired_contacts (w=-0.1)
        r9 = -0.1 * self._undesired_contact()
        # 10. anti_shake (w=-0.005)
        r10 = -0.005 * self._anti_shake()
        # 11. tracking_vr_local (w=2.0, σ=0.1)
        r11 = 2.0 * self._vr_local_error()
        # 12. feet_acc (w=-2.5e-6)
        r12 = -2.5e-6 * self._feet_acc()

        return float(r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12)

    def _undesired_contact(self):
        excluded = {"left_ankle_roll_link", "right_ankle_roll_link",
                    "left_wrist_yaw_link", "right_wrist_yaw_link",
                    "left_elbow_link", "right_elbow_link"}
        total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[c.geom2])
            if b1 not in excluded and b2 not in excluded:
                force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, force)
                total += np.linalg.norm(force[:3])
        return max(total - 1.0, 0.0)

    def _anti_shake(self):
        target = ("left_wrist_yaw_link", "right_wrist_yaw_link", "head_link")
        excesses = []
        for name in target:
            idx = self._body_idx.get(name)
            if idx is None: continue
            dof_adr = self.model.body_dofadr[idx]
            if dof_adr < 0: continue
            w = self.data.cvel[dof_adr + 3: dof_adr + 6]
            excesses.append(max(np.linalg.norm(w) - 1.5, 0))
        return float(np.mean(np.array(excesses)**2)) if excesses else 0.0

    def _vr_local_error(self):
        pt_bodies = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
        pt_offsets = np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        n_pts = len(pt_bodies)

        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()
        ref_body_pos = self._ref_body_pos(); ref_body_quat = self._ref_body_quat()
        ref_idx = [list(BODY_NAMES).index(n) for n in pt_bodies]
        ref_pt_w = ref_body_pos[ref_idx] + quat_apply(ref_body_quat[ref_idx], pt_offsets)
        ref_local = quat_apply(quat_inv(np.tile(ref_root_quat, (n_pts, 1))), ref_pt_w - ref_root_pos)

        root_pos = self.data.xpos[self._body_idx["pelvis"]]
        root_quat = self.data.xquat[self._body_idx["pelvis"]]
        rob_idx = [self._body_idx[n] for n in pt_bodies]
        rob_pt_w = self.data.xpos[rob_idx] + quat_apply(self.data.xquat[rob_idx], pt_offsets)
        rob_local = quat_apply(quat_inv(np.tile(root_quat, (n_pts, 1))), rob_pt_w - root_pos)

        err = np.sum((rob_local - ref_local)**2)
        return float(np.exp(-err / (n_pts * 0.01)))  # σ²=0.01

    def _feet_acc(self):
        ankle_names = ("left_ankle_pitch_joint", "left_ankle_roll_joint",
                       "right_ankle_pitch_joint", "right_ankle_roll_joint")
        indices = []
        for name in ankle_names:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                dadr = self.model.jnt_dofadr[jid]
                if dadr >= 6: indices.append(dadr - 6)
            except Exception: pass
        if not indices: return 0.0
        ankle_vel = self.data.qvel[6:][indices]
        acc = (ankle_vel - self._prev_joint_vel[indices]) / self.ctrl_dt
        self._prev_joint_vel = self.data.qvel[6:].copy()
        return float(np.sum(acc**2))

    # ═══════════════════════════════════════════════════════════════════
    # Termination
    # ═══════════════════════════════════════════════════════════════════
    def _check_termination(self):
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()
        root_pos = self.data.xpos[self._body_idx["pelvis"]]
        root_quat = self.data.xquat[self._body_idx["pelvis"]]
        ref_h = ref_root_pos[2]; root_h = root_pos[2]
        term = False; h_thresh = 0.75 if ref_h < 0.5 else 0.15

        if abs(ref_h - root_h) > h_thresh: term = True
        if quat_error_magnitude(ref_root_quat, root_quat)**2 > 0.2: term = True

        ref_body_pos = self._ref_body_pos()
        for name in ("left_ankle_roll_link", "right_ankle_roll_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link"):
            idx = list(BODY_NAMES).index(name)
            if abs(ref_body_pos[idx, 2] - self.data.xpos[self._body_idx[name]][2]) > h_thresh:
                term = True; break

        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            idx = list(BODY_NAMES).index(name)
            ref_aligned = ref_body_pos[idx] - ref_root_pos + root_pos
            if np.linalg.norm(ref_aligned - self.data.xpos[self._body_idx[name]]) > 0.2:
                term = True; break

        trunc = int(self._ref_time * self._ref_fps) >= len(self._ref_dof) - 1
        return term, trunc

    # ═══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════
    def reset(self):
        self._sample_motion()
        ref_q0 = self._ref_dof[int(self._ref_time * self._ref_fps)]
        self.data.qpos[7:] = ref_q0.astype(np.float64); self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        for b in (self._gdh, self._avh, self._jph, self._jvh, self._ah, self._lvh): b.fill(0)
        self._prev_action.fill(0)
        self._prev_root_pos = self.data.xpos[self._body_idx["pelvis"]].copy()
        self._prev_body_pos = self.data.xpos[self._body_indices].copy()
        self._prev_body_quat = self.data.xquat[self._body_indices].copy()
        self._prev_ref_body_pos = self._prev_body_pos.copy()
        self._prev_ref_body_quat = self._prev_body_quat.copy()
        self._prev_joint_vel = self.data.qvel[6:].copy()
        self.ep = 0
        self._compute_ref_body_state()
        return self._obs()

    def step(self, action):
        if action.ndim == 2: action = action[0]
        action = np.clip(action, -1, 1).astype(np.float64)
        # Update action history before PD
        self._ah[:-1] = self._ah[1:]; self._ah[-1] = action
        self._pd_control(action); self._physics_step()
        self._advance_motion_time()
        self._compute_ref_body_state()
        obs = self._obs()
        reward = self._compute_reward(action)
        terminated, truncated = self._check_termination()
        done = terminated or truncated
        self.ep += 1
        terminal_obs = None
        if done:
            terminal_obs = {k: v.copy() for k, v in obs.items()}
            obs = self.reset()
        if self.ignore_terminations:
            # Track original termination for GAE bootstrapping
            info = {"time_outs": truncated, "terminal_obs": terminal_obs,
                    "_orig_done": done}
            done = False
        else:
            info = {"time_outs": truncated, "terminal_obs": terminal_obs}
        return obs, reward, done, info
