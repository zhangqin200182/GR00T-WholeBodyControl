"""PhysX G1 environment — SONIC-compatible observation/reward/termination.

Drop-in replacement for MuJoCoEnv using PhysX 5 physics engine.
"""
import os, glob, sys, numpy as np, joblib

# Ensure gear_sonic is importable (needed on NPU server where package may not be installed)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_this_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Ensure mujoco_math is importable (gear_sonic is not a package — no __init__.py)
_this_dir = os.path.dirname(os.path.abspath(__file__))  # .../gear_sonic/envs/
_physx_dir = os.path.join(_this_dir, "physx")
_repo_root = os.path.dirname(os.path.dirname(_this_dir))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
import mujoco_math
quat_mul = mujoco_math.quat_mul
quat_inv = mujoco_math.quat_inv
quat_apply = mujoco_math.quat_apply
quat_error_magnitude = mujoco_math.quat_error_magnitude
quat_to_matrix = mujoco_math.quat_to_matrix
subtract_frame_transforms = mujoco_math.subtract_frame_transforms
quat_diff_to_angvel = mujoco_math.quat_diff_to_angvel
# Same for physx loader and FK
if _physx_dir not in sys.path:
    sys.path.insert(0, _physx_dir)
import physx_loader, physx_fk
load_g1 = physx_loader.load_g1
G1ForwardKinematics = physx_fk.G1ForwardKinematics

# ── Constants (identical to MuJoCoEnv) ─────────────────────────────────
NUM_DOF = 29
NUM_FUTURE = 10
NUM_SMPL_FUTURE = 10
FUTURE_DT_REF = 0.1
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
HIST = 10


class PhysXEnv:
    """Single PhysX G1 environment — SONIC-compatible obs/reward/termination."""

    def __init__(self, px, model_xml, pkl_dir, config=None,
                 native_dt=0.002, decimation=10, pos_iters=8, vel_iters=1,
                 static_pose=False, root_z_offset=0.0, standing_prob=0.0,
                 drive_type="ACCELERATION"):
        self.px = px
        self._static_pose = static_pose
        self._root_z_offset = root_z_offset
        self._standing_prob = standing_prob  # per-episode prob (0=never, 1=always)

        # ── PhysX scene + articulation ──
        self.art = load_g1(px, model_xml, pos_iters=pos_iters, vel_iters=vel_iters,
                             drive_type=drive_type)
        self.scene = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
        mat = self.scene.create_material(0.6, 0.5, 0.0)
        self.scene.add_ground_plane(mat, np.array([0,0,1], dtype=np.float32))
        self.scene.add_articulation(self.art)

        # ── Python FK (replaces mj_kinematics) ──
        self._fk = G1ForwardKinematics(model_xml)

        # ── Simulation params ──
        self.native_dt = native_dt; self.decimation = decimation
        self.ctrl_dt = self.native_dt * self.decimation  # 0.020 = 50Hz

        # ── Action space (Isaac WBC config — same as MuJoCoEnv) ──
        self.nu = 29
        self.jm = np.array([
             0.1745,  1.2217,  0.0000,  1.3963, -0.1745,  0.0000,
             0.1745, -1.2217,  0.0000,  1.3963, -0.1745,  0.0000,
             0.0000,  0.0000,  0.0000,
            -0.2094,  0.3317,  0.0000,  0.5236,  0.0000,  0.0000,  0.0000,
            -0.2094, -0.3317,  0.0000,  0.5236,  0.0000,  0.0000,  0.0000,
        ], dtype=np.float64)
        self.jh = np.array([
            2.7052, 1.7453, 2.7576, 1.4835, 0.6981, 0.2618,
            2.7052, 1.7453, 2.7576, 1.4835, 0.6981, 0.2618,
            2.6180, 0.5200, 0.5200,
            2.8798, 1.9199, 2.6180, 1.5708, 1.9722, 1.6144, 1.6144,
            2.8798, 1.9199, 2.6180, 1.5708, 1.9722, 1.6144, 1.6144,
        ], dtype=np.float64)

        # ── PD drive params (set once during load_g1, can override here) ──
        # Already set to kp=100, kd=5 in loader

        # ── Body indices (by name) ──
        self._body_idx = {n: self.art.get_link_index(n) for n in BODY_NAMES}
        self._body_indices = np.array([self._body_idx[n] for n in BODY_NAMES])

        # ── Episode ──
        self.ep = 0
        self.max_ep = getattr(config, "max_episode_length", 500) if config else 500
        self.ignore_terminations = getattr(config, "ignore_terminations", False) if config else False
        self.skip_termination = getattr(config, "skip_termination", False) if config else False
        self.alive_bonus = getattr(config, "alive_bonus", 0.0) if config else 0.0
        self.ori_thresh = getattr(config, "ori_thresh", 0.2) if config else 0.2
        self.ank_pos_thresh = getattr(config, "ank_pos_thresh", 0.2) if config else 0.2
        self.ank_h_mult = getattr(config, "ank_h_mult", 1.0) if config else 1.0
        self.action_trust = getattr(config, "action_trust", 1.0) if config else 1.0
        self.height_hinge_weight = getattr(config, "height_hinge_weight", 0.0) if config else 0.0
        self.ankle_hinge_weight = getattr(config, "ankle_hinge_weight", 0.0) if config else 0.0
        self.ankle_vel_penalty_weight = getattr(config, "ankle_vel_penalty_weight", 0.0) if config else 0.0

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
    # Motion loading (identical to MuJoCoEnv)
    # ═══════════════════════════════════════════════════════════════════
    def _load_motions(self, pkl_dir):
        pkls = [p for p in glob.glob(os.path.join(pkl_dir, "**/*.pkl"), recursive=True)
                if not os.path.basename(p).startswith("._")]
        if not pkls: raise RuntimeError(f"No motion PKLs in {pkl_dir}")
        rng = np.random.RandomState(os.getpid() + id(self) % 100000)
        rng.shuffle(pkls)
        motions = []
        for p in pkls:
            v = joblib.load(p)
            if isinstance(v, dict):
                if "dof" in v:
                    motions.append(v); continue
                for key in v:
                    if isinstance(v[key], dict) and "dof" in v[key]:
                        motions.append(v[key]); break
            if len(motions) >= 500: break
        if not motions: raise RuntimeError(f"No motion PKLs in {pkl_dir}")
        return motions

    def _sample_motion(self):
        m = self.motions[np.random.randint(len(self.motions))]
        dof = m["dof"]; n = len(dof)
        self._ref_dof = dof
        self._ref_root_rot = m["root_rot"].astype(np.float64)
        self._ref_root_trans = m["root_trans_offset"].astype(np.float64)
        self._ref_fps = m.get("fps", 30.0)
        self._ref_dt = 1.0 / self._ref_fps
        # Offset applies in walking mode too: mocap root-z is not physically
        # consistent with G1 leg geometry (feet hover 25-35mm above ground at
        # reset). Without it every episode starts with a free-fall transient
        # that eats into the height-termination budget.
        if self._root_z_offset != 0.0:
            self._ref_root_trans[:, 2] += self._root_z_offset
        if self._static_pose:
            self._ref_time = 0.0
        else:
            max_time = (n - self.max_ep - 1) * self._ref_dt
            self._ref_time = np.random.uniform(0, max(0.001, max_time))

    def _advance_motion_time(self):
        if not self._static_pose:
            self._ref_time += self.ctrl_dt

    def _future_dof(self, n=NUM_FUTURE, dt_ref=FUTURE_DT_REF):
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
    # Reference body state via Python FK
    # ═══════════════════════════════════════════════════════════════════
    def _compute_ref_body_state(self):
        idx = min(int(self._ref_time * self._ref_fps), len(self._ref_dof) - 1)
        ref_qpos = self._ref_dof[idx].astype(np.float64)
        ref_root_pos = self._ref_root_trans[idx]
        pk_quat = self._ref_root_rot[idx]  # [x,y,z,w] from PKL
        ref_root_quat = np.array([pk_quat[3], pk_quat[0], pk_quat[1], pk_quat[2]], dtype=np.float64)

        all_poses = self._fk.compute(ref_root_pos, ref_root_quat, ref_qpos)
        # Use tracked_indices to map FK link order → BODY_NAMES order
        tracked = [all_poses[i] for i in self._fk._tracked_indices]
        self._cached_ref_body_pos = np.array([t[0] for t in tracked], dtype=np.float64)
        self._cached_ref_body_quat = np.array([t[1] for t in tracked], dtype=np.float64)

    def _ref_root_pos(self):
        self._compute_ref_body_state()
        return self._cached_ref_body_pos[0].copy()

    def _ref_root_quat(self):
        self._compute_ref_body_state()
        return self._cached_ref_body_quat[0].copy()

    def _ref_body_pos(self):
        self._compute_ref_body_state()
        return self._cached_ref_body_pos.copy()

    def _ref_body_quat(self):
        self._compute_ref_body_state()
        return self._cached_ref_body_quat.copy()

    def _future_ref_root_pos(self, n=NUM_FUTURE, dt_ref=FUTURE_DT_REF):
        fps = self._ref_fps; end = len(self._ref_dof)
        times = self._ref_time + np.arange(n) * dt_ref
        indices = np.clip((times * fps).astype(int), 0, end - 1)
        return self._ref_root_trans[indices].astype(np.float32)

    def _future_ref_root_quat(self, n=NUM_FUTURE, dt_ref=FUTURE_DT_REF):
        fps = self._ref_fps; end = len(self._ref_dof)
        times = self._ref_time + np.arange(n) * dt_ref
        indices = np.clip((times * fps).astype(int), 0, end - 1)
        pk = self._ref_root_rot[indices]
        ret = np.zeros((n, 4), dtype=np.float32)
        ret[:, 0] = pk[:, 3]; ret[:, 1:] = pk[:, :3]
        return ret

    # ═══════════════════════════════════════════════════════════════════
    # Body state via Python FK (replaces get_link_world_pose everywhere)
    # ═══════════════════════════════════════════════════════════════════
    def _get_body_state_fk(self):
        """Return (body_pos, body_quat) for BODY_NAMES via Python FK, cached per step."""
        step_id = (id(self), self.ep, self._ref_time)
        if getattr(self, '_fk_cache_key', None) == step_id:
            return self._fk_body_pos, self._fk_body_quat
        root_pos, root_quat = self.art.get_root_world_pose()
        actual_qpos = self.art.get_joint_positions()
        tracked = self._fk.get_tracked_poses(root_pos, root_quat, actual_qpos)
        self._fk_body_pos = np.array([t[0] for t in tracked], dtype=np.float64)
        self._fk_body_quat = np.array([t[1] for t in tracked], dtype=np.float64)
        self._fk_cache_key = step_id
        return self._fk_body_pos, self._fk_body_quat

    # ═══════════════════════════════════════════════════════════════════
    # Physics
    # ═══════════════════════════════════════════════════════════════════
    def _pd_control(self, action):
        target = action * self.jh + self.jm
        self.art.set_joint_drive_targets(target.astype(np.float32))

    def _physics_step(self):
        for _ in range(self.decimation):
            self.scene.simulate(self.native_dt)
            self.scene.fetch_results()

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
                "tokenizer": self._build_tokenizer(),
                "ref_action": (self.get_current_ref_qpos() - self.jm) / self.jh}

    def _compute_actor_obs(self):
        root_quat = self.art.get_root_world_pose()[1]
        g_body = quat_apply(quat_inv(root_quat), np.array([0, 0, -1]))
        # PhysX getRootAngularVelocity → world frame; convert to body frame (MuJoCo convention)
        ang_vel_world = self.art.get_root_world_velocity()[1]
        ang_vel = quat_apply(quat_inv(root_quat), ang_vel_world)
        jpos = self.art.get_joint_positions(); jvel = self.art.get_joint_velocities()
        self._shift_histories()
        self._gdh[-1] = g_body; self._avh[-1] = ang_vel
        self._jph[-1] = jpos; self._jvh[-1] = jvel
        return np.concatenate([b.flatten() for b in
            [self._gdh, self._avh, self._jph, self._jvh, self._ah]]).astype(np.float32)

    def _compute_critic_obs(self):
        root_pos, root_quat = self.art.get_root_world_pose()
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()

        future_pos = self._future_dof(); future_vel = self._future_dof_vel()
        cmd_mf = np.concatenate([future_pos.flatten(), future_vel.flatten()])
        pos_b, ori_b = subtract_frame_transforms(root_pos, root_quat, ref_root_pos, ref_root_quat)
        mat = quat_to_matrix(ori_b); ori_6d = mat[..., :2].flatten()

        body_pos_w, body_quat_w = self._get_body_state_fk()
        body_pos_b = quat_apply(quat_inv(root_quat), body_pos_w - root_pos).flatten()
        body_quat_b = quat_mul(np.tile(quat_inv(root_quat), (14, 1)), body_quat_w)
        ori_parts = [quat_to_matrix(q)[:2].flatten() for q in body_quat_b]
        body_ori_flat = np.concatenate(ori_parts)

        lin_vel = (root_pos - self._prev_root_pos) / self.ctrl_dt
        self._lvh[-1] = lin_vel; self._prev_root_pos = root_pos.copy()

        parts = [cmd_mf, pos_b.flatten(), ori_6d, body_pos_b, body_ori_flat] + \
                [b.flatten() for b in [self._lvh, self._avh, self._jph, self._jvh, self._ah]]
        return np.concatenate(parts).astype(np.float32)

    def _build_tokenizer(self):
        root_quat = self.art.get_root_world_pose()[1]
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()

        future_pos = self._future_dof(); future_vel = self._future_dof_vel()
        cmd_nonflat = np.concatenate([future_pos, future_vel], axis=-1).flatten()
        future_rp = self._future_ref_root_pos(); cmd_z_mf = future_rp[:, 2:3].flatten()
        future_rq = self._future_ref_root_quat()
        root_q_e = np.tile(root_quat, (NUM_FUTURE, 1))
        rot_diff = quat_mul(quat_inv(root_q_e), future_rq)
        ori_mf = np.stack([quat_to_matrix(q) for q in rot_diff])[:, :2, :].flatten()
        lower_pos = future_pos[:, LOWER_JOINT_INDICES]
        lower_vel = future_vel[:, LOWER_JOINT_INDICES]
        cmd_lower = np.concatenate([lower_pos.flatten(), lower_vel.flatten()])

        ref_body_pos = self._ref_body_pos(); ref_body_quat = self._ref_body_quat()
        vr_idx = [list(BODY_NAMES).index(n) for n in VR_3POINT_BODY]
        vr_pos = ref_body_pos[vr_idx]; vr_quat = ref_body_quat[vr_idx]
        vr_w = vr_pos + quat_apply(vr_quat, VR_3POINT_OFFSETS)
        ref_rq_3p = np.tile(ref_root_quat, (3, 1))
        vr_local = quat_apply(quat_inv(ref_rq_3p), vr_w - ref_root_pos).flatten()
        vr_orn = quat_mul(quat_inv(ref_rq_3p), vr_quat).flatten()

        _, ori_b = subtract_frame_transforms(
            self.art.get_root_world_pose()[0], root_quat, ref_root_pos, ref_root_quat)
        mat_s = quat_to_matrix(ori_b); ori_b_6d = mat_s[..., :2].flatten()
        cmd_z = np.array([ref_root_pos[2]], dtype=np.float32)

        enc_idx = np.zeros(3, dtype=np.float32); enc_idx[0] = 1.0
        smpl_j = np.zeros(NUM_SMPL_FUTURE * 24 * 3, dtype=np.float32)
        smpl_o = np.zeros(NUM_SMPL_FUTURE * 6, dtype=np.float32)
        smpl_w = np.zeros(NUM_SMPL_FUTURE * 6, dtype=np.float32)

        return np.concatenate([enc_idx, cmd_nonflat, cmd_z_mf, ori_mf, cmd_lower,
                               vr_local, vr_orn, ori_b_6d, cmd_z,
                               smpl_j, smpl_o, smpl_w]).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # Reward (identical math to MuJoCoEnv)
    # ═══════════════════════════════════════════════════════════════════
    def _compute_reward(self, action):
        root_pos, root_quat = self.art.get_root_world_pose()
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()

        err = np.linalg.norm(root_pos - ref_root_pos)
        r1 = 0.5 * np.exp(-err**2 / 0.09)
        r2 = 0.5 * np.exp(-quat_error_magnitude(ref_root_quat, root_quat)**2 / 0.16)

        ref_body_pos = self._ref_body_pos()
        ref_aligned = ref_body_pos - ref_root_pos + root_pos
        body_pos_w, body_quat_w = self._get_body_state_fk()
        r3 = 1.0 * np.exp(-np.sum((body_pos_w - ref_aligned)**2, axis=-1).mean() / 0.09)

        ref_body_quat = self._ref_body_quat()
        ang_errs = np.array([quat_error_magnitude(ref_body_quat[i], body_quat_w[i]) for i in range(14)])
        r4 = 1.0 * np.exp(-(ang_errs**2).mean() / 0.16)

        body_lin_vel = (body_pos_w - self._prev_body_pos) / self.ctrl_dt
        ref_lin_vel = (ref_body_pos - self._prev_ref_body_pos) / self.ctrl_dt
        r5 = 1.0 * np.exp(-np.sum((body_lin_vel - ref_lin_vel)**2, axis=-1).mean() / 1.0)
        self._prev_body_pos = body_pos_w.copy(); self._prev_ref_body_pos = ref_body_pos.copy()

        body_ang_vel = quat_diff_to_angvel(self._prev_body_quat, body_quat_w, self.ctrl_dt)
        ref_ang_vel = quat_diff_to_angvel(self._prev_ref_body_quat, ref_body_quat, self.ctrl_dt)
        r6 = 1.0 * np.exp(-np.sum((body_ang_vel - ref_ang_vel)**2, axis=-1).mean() / 9.86)
        self._prev_body_quat = body_quat_w.copy(); self._prev_ref_body_quat = ref_body_quat.copy()

        r7 = -0.1 * np.sum((action - self._prev_action)**2); self._prev_action = action.copy()
        q = self.art.get_joint_positions()
        r8 = -10.0 * np.sum(np.maximum(np.abs(q - self.jm) - self.jh, 0))
        r9 = -0.1 * self._undesired_contact()
        r10 = -0.005 * self._anti_shake()
        r11 = 2.0 * self._vr_local_error()
        r12 = -2.5e-6 * self._feet_acc()
        r13 = self.alive_bonus

        # Height hinge: r1's Gaussian (sigma 0.09) still pays 39% of max at
        # dh=150mm, so root-height sink is barely penalized — yet it is the
        # dominant termination cause under PhysX (75% of falls).  This term
        # rewards full value at perfect height and ramps linearly to 0 at the
        # active termination threshold, giving a constant gradient where r1's
        # vanishes.  Default 0.0 keeps eval scripts on the original reward.
        r14 = 0.0
        if self.height_hinge_weight > 0.0:
            h_thresh = 0.75 if ref_root_pos[2] < 0.5 else 0.15
            dh = abs(root_pos[2] - ref_root_pos[2])
            r14 = self.height_hinge_weight * max(0.0, 1.0 - dh / h_thresh)

        # Ankle hinge: the 0.20-threshold death diagnosis (08-16) showed the
        # dominant failure is ankle horizontal-position error, marginally over
        # the strict threshold (0.201-0.241 vs 0.200).  Mirror the height-hinge
        # structure: full reward at zero error, linear ramp to 0 at the
        # TRAINING ank_pos_thresh (0.35) — constant gradient where the existing
        # tracking rewards flatten out.  Default 0.0 keeps eval scripts on the
        # original reward.
        r15 = 0.0
        if self.ankle_hinge_weight > 0.0:
            actual_body_pos, _ = self._get_body_state_fk()
            ref_body_pos = self._ref_body_pos()
            ankle_errs = []
            for _name in ("left_ankle_roll_link", "right_ankle_roll_link"):
                _idx = list(BODY_NAMES).index(_name)
                _ref_aligned = ref_body_pos[_idx] - ref_root_pos + root_pos
                ankle_errs.append(float(np.linalg.norm(_ref_aligned - actual_body_pos[_idx])))
            ankle_err = float(np.mean(ankle_errs))
            r15 = self.ankle_hinge_weight * max(0.0, 1.0 - ankle_err / self.ank_pos_thresh)

        # Ankle velocity penalty: the ankle hinge taught the policy to pump
        # the ankles at high frequency (jitter 32-116 mrad/step vs 19-28 for
        # the pre-hinge policy — render visibly shaky).  Punish ankle joint
        # speed so tight ankle tracking must come from smooth control.
        # Typical values: mean_sq 3.0 (normal) vs 4.4 (hinge-jittery).
        r16 = 0.0
        if self.ankle_vel_penalty_weight > 0.0:
            qvel = self.art.get_joint_velocities()
            ankle_msq = float(np.mean(qvel[[4, 5, 10, 11]] ** 2))
            r16 = -self.ankle_vel_penalty_weight * ankle_msq

        return float(r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16)

    def _undesired_contact(self):
        # TODO: implement via scene.get_contacts() once T3 adds the API
        return 0.0

    def _anti_shake(self):
        # head_link is not in BODY_NAMES (no tracked body index), so only
        # the two wrist links are checked.  Per-link angular velocity is
        # estimated via quaternion differencing.
        target = ("left_wrist_yaw_link", "right_wrist_yaw_link")
        excesses = []
        _, all_quat = self._get_body_state_fk()
        for name in target:
            bi = list(BODY_NAMES).index(name)
            curr_quat = all_quat[bi]
            prev_quat = self._prev_body_quat[bi]
            ang_vel = quat_diff_to_angvel(prev_quat, curr_quat, self.ctrl_dt)
            # quat_diff_to_angvel expects (N,4) batches; passing (4,) works
            # because the ellipsis indexing (q[..., 1:]) degrades cleanly.
            excesses.append(max(np.linalg.norm(ang_vel) - 1.5, 0))
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

        root_pos, root_quat = self.art.get_root_world_pose()
        all_pos, all_quat = self._get_body_state_fk()
        rob_bidx = [list(BODY_NAMES).index(n) for n in pt_bodies]
        rob_pt_w = all_pos[rob_bidx]
        rob_quat_w = all_quat[rob_bidx]
        rob_pt_w = rob_pt_w + quat_apply(rob_quat_w, pt_offsets)
        rob_local = quat_apply(quat_inv(np.tile(root_quat, (n_pts, 1))), rob_pt_w - root_pos)

        err = np.sum((rob_local - ref_local)**2)
        return float(np.exp(-err / (n_pts * 0.01)))

    def _feet_acc(self):
        ankle_indices = [4, 5, 10, 11]  # left/right ankle pitch/roll in 29-DOF order
        ankle_vel = self.art.get_joint_velocities()[ankle_indices]
        acc = (ankle_vel - self._prev_joint_vel[ankle_indices]) / self.ctrl_dt
        self._prev_joint_vel = self.art.get_joint_velocities().copy()
        return float(np.sum(acc**2))

    # ═══════════════════════════════════════════════════════════════════
    # Termination — use consistent FK for both robot and reference.
    # get_link_world_pose reflects PhysX's internal FK (world-pose createLink)
    # which disagrees with MJCF local-pose FK.  Until this is reconciled, compute
    # robot body positions via Python FK from actual joint angles.
    # ═══════════════════════════════════════════════════════════════════
    def _check_termination(self):
        if self.skip_termination:
            return False, False

        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()
        root_pos, root_quat = self.art.get_root_world_pose()
        ref_h = ref_root_pos[2]; root_h = root_pos[2]
        term = False; h_thresh = 0.75 if ref_h < 0.5 else 0.15
        term_reason = ""

        dh = abs(ref_h - root_h)
        if dh > h_thresh:
            term = True; term_reason = f"height({dh:.3f}>{h_thresh:.3f})"

        ori_err = quat_error_magnitude(ref_root_quat, root_quat)**2
        if ori_err > self.ori_thresh:
            term = True
            term_reason += f" ori({ori_err:.3f}>{self.ori_thresh:.3f})"

        # Compute robot body positions via Python FK for consistent reference comparison
        actual_qpos = self.art.get_joint_positions()
        tracked = self._fk.get_tracked_poses(root_pos, root_quat, actual_qpos)
        actual_body_pos = np.array([t[0] for t in tracked], dtype=np.float64)
        ref_body_pos = self._ref_body_pos()

        for name in ("left_ankle_roll_link", "right_ankle_roll_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link"):
            idx = list(BODY_NAMES).index(name)
            err = abs(ref_body_pos[idx, 2] - actual_body_pos[idx][2])
            h_limit = h_thresh * self.ank_h_mult
            if err > h_limit:
                term = True
                term_reason += f" {name}_h({err:.3f}>{h_limit:.3f})"
                break

        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            idx = list(BODY_NAMES).index(name)
            ref_aligned = ref_body_pos[idx] - ref_root_pos + root_pos
            err = np.linalg.norm(ref_aligned - actual_body_pos[idx])
            if err > self.ank_pos_thresh:
                term = True
                term_reason += f" {name}_pos({err:.3f}>{self.ank_pos_thresh:.3f})"
                break

        if term and self.ep == 0:
            pass  # Debug print removed — use action_trust to bootstrap

        trunc = (int(self._ref_time * self._ref_fps) >= len(self._ref_dof) - 1
                 or self.ep >= self.max_ep)
        return term, trunc

    # ═══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════
    def get_current_ref_qpos(self):
        idx = min(int(self._ref_time * self._ref_fps), len(self._ref_dof) - 1)
        return self._ref_dof[idx].astype(np.float32)

    def reset(self):
        if self._standing_prob > 0:
            self._static_pose = np.random.random() < self._standing_prob
        self._sample_motion()
        idx = int(self._ref_time * self._ref_fps)
        ref_q0 = self._ref_dof[idx]
        self.art.set_root_world_pose(
            self._ref_root_trans[idx].astype(np.float32),
            np.array([self._ref_root_rot[idx][3], self._ref_root_rot[idx][0],
                      self._ref_root_rot[idx][1], self._ref_root_rot[idx][2]], dtype=np.float32))
        self.art.set_joint_positions(ref_q0.astype(np.float32))
        self.art.set_joint_velocities(np.zeros(self.nu, dtype=np.float32))
        self.art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                          np.zeros(3, dtype=np.float32))
        # Reset drive targets to match initial pose — otherwise stale targets
        # from the previous episode cause extreme torque transients.
        self.art.set_joint_drive_targets(ref_q0.astype(np.float32))
        for b in (self._gdh, self._avh, self._jph, self._jvh, self._ah, self._lvh): b.fill(0)
        self._prev_action.fill(0)
        root_pos, root_quat = self.art.get_root_world_pose()
        self._prev_root_pos = root_pos.copy()
        fk_pos, fk_quat = self._get_body_state_fk()
        self._prev_body_pos = fk_pos.copy()
        self._prev_body_quat = fk_quat.copy()
        self._prev_ref_body_pos = self._prev_body_pos.copy()
        self._prev_ref_body_quat = self._prev_body_quat.copy()
        self._prev_joint_vel = self.art.get_joint_velocities()
        self.ep = 0
        self._compute_ref_body_state()
        return self._obs()

    def step(self, action):
        if action.ndim == 2: action = action[0]
        action = np.clip(action, -1, 1).astype(np.float64)

        # Blend model action with ref_action: action_trust=0 means pure ref_action
        # NaN guard: 0.0 * NaN = NaN in IEEE 754 — model NaNs would leak
        # through PD targets → physics → observations → all envs.
        if self.action_trust < 1.0:
            ref_action = (self.get_current_ref_qpos() - self.jm) / self.jh
            if not np.isfinite(action).all():
                action = ref_action.copy()
            else:
                action = self.action_trust * action + (1.0 - self.action_trust) * ref_action

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
            info = {"time_outs": truncated, "terminal_obs": terminal_obs, "_orig_done": done}
            done = False
        else:
            info = {"time_outs": truncated, "terminal_obs": terminal_obs}
        return obs, reward, done, info
