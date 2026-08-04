"""PhysX G1 environment using ovphysx backend — SONIC-compatible obs/reward/termination.

Drop-in replacement for MuJoCoEnv using ovphysx (pip-installable PhysX 5 + USD bridge).
"""

import os, glob, sys, numpy as np, joblib

# Ensure gear_sonic is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_this_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Ensure mujoco_math is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_physx_dir = os.path.join(_this_dir, "physx")
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

# FK
if _physx_dir not in sys.path:
    sys.path.insert(0, _physx_dir)
import physx_fk
G1ForwardKinematics = physx_fk.G1ForwardKinematics

# ovphysx imports
import ovphysx, ovstage
from ovphysx.types import TensorType

# ── DOF permutation: BFS (ovphysx kinematic tree) ↔ actuator (MJCF/PKL) order ──
DOF2ACT = np.array([0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28], dtype=np.int32)
ACT2DOF = np.zeros(29, dtype=np.int32)
for d, a in enumerate(DOF2ACT):
    ACT2DOF[a] = d


def _xyzw_to_wxyz(q):
    """Convert xyzw quaternion (ovphysx/USD) to wxyz (mujoco_math)."""
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    out = np.zeros_like(q)
    out[..., 0] = q[..., 3]
    out[..., 1] = q[..., 0]
    out[..., 2] = q[..., 1]
    out[..., 3] = q[..., 2]
    return out


def _wxyz_to_xyzw(q):
    """Convert wxyz quaternion (mujoco_math) to xyzw (ovphysx/USD)."""
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    out = np.zeros_like(q, dtype=np.float32)
    out[..., 0] = q[..., 1]
    out[..., 1] = q[..., 2]
    out[..., 2] = q[..., 3]
    out[..., 3] = q[..., 0]
    return out


# ── Constants (identical to PhysXEnv / MuJoCoEnv) ─────────────────────
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


class TensorBinding:
    """Wraps an ovphysx tensor binding for convenient numpy read/write."""

    def __init__(self, binding):
        self._b = binding
        self._buf = np.zeros(binding.shape, dtype=np.float32)

    def read(self):
        self._b.read(self._buf)
        return self._buf.copy()

    def write(self, data):
        self._buf[:] = data
        self._b.write(self._buf)

    def destroy(self):
        self._b.destroy()


def prepare_usd(robot_usd_path, output_path=None):
    """Embed PhysicsScene + GroundPlane into robot USD.

    Args:
        robot_usd_path: Path to robot-only USD file (e.g. g1_29dof_physx_v3.usda)
        output_path: Output path (default: /tmp/g1_combined_{pid}.usda)

    Returns:
        Path to combined USD file
    """
    if output_path is None:
        output_path = f"/tmp/g1_combined_{os.getpid()}.usda"

    with open(robot_usd_path, "r") as f:
        original = f.read()

    world_open = 'def Xform "World"'
    insert_pos = original.index(world_open) + len(world_open)
    brace_pos = original.index("{", insert_pos)

    SCENE = """
    def PhysicsScene "physicsScene"
    {
        float3 gravity = (0, 0, -9.81)
    }

    def Xform "GroundPlane"
    {
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        def Plane "CollisionPlane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            uniform token axis = "Z"
            uniform token purpose = "guide"
        }
    }

"""
    combined = original[:brace_pos + 1] + SCENE + original[brace_pos + 1:]
    with open(output_path, "w") as f:
        f.write(combined)
    return output_path


class PhysXEnvOv:
    """Single PhysX G1 environment using ovphysx backend — SONIC-compatible."""

    # Class-level counter for recreation scheduling across all envs
    _total_resets = 0

    def __init__(self, px, stage, robot_usd_path, model_xml, pkl_dir, config=None,
                 native_dt=0.001961, decimation=17, recreate_every=15):
        self._robot_usd_path = robot_usd_path
        self._model_xml = model_xml
        self._pkl_dir = pkl_dir
        self._recreate_every = recreate_every
        self._px = px
        self._stage = stage
        self._config = config

        self._init_physics()

        # Python FK (still needs MJCF XML for kinematic tree)
        self._fk = G1ForwardKinematics(model_xml)

        # Simulation params
        self.native_dt = native_dt
        self.decimation = decimation
        self.ctrl_dt = native_dt * decimation  # ~0.033 = 30Hz

        # Action space (Isaac WBC config — same as MuJoCoEnv / PhysXEnv)
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

        # Episode
        self.ep = 0
        self.max_ep = getattr(config, "max_episode_length", 500) if config else 500
        self.ignore_terminations = getattr(config, "ignore_terminations", False) if config else False
        self.alive_bonus = getattr(config, "alive_bonus", 0.0) if config else 0.0

        # Motions
        self.motions = self._load_motions(pkl_dir)

        # History buffers
        self._init_history()

        # Prev frame cache
        self._prev_action = np.zeros(self.nu, dtype=np.float32)
        self._prev_root_pos = np.zeros(3, dtype=np.float64)
        self._prev_root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
        self._prev_body_pos = np.zeros((14, 3), dtype=np.float64)
        self._prev_body_quat = np.zeros((14, 4), dtype=np.float64)
        self._prev_ref_body_pos = np.zeros((14, 3), dtype=np.float64)
        self._prev_ref_body_quat = np.zeros((14, 4), dtype=np.float64)
        self._prev_joint_vel = np.zeros(self.nu, dtype=np.float64)

        # Episode counter for periodic physics recreation
        self._episode_count = 0

    def _init_physics(self):
        """(Re)create ovphysx physics state: load USD, attach, create tensor bindings."""
        ovstage.population.open_usd(self._stage, self._robot_usd_path, ordinal=1,
                                     domains=ovstage.PopulationDomain.PHYSICS)
        self._px.attach_ovstage(self._stage, read_ordinal=1)
        self._px.wait_all()

        pattern = "/World/G1/*"
        self._b_pos = TensorBinding(
            self._px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
        self._b_tgt = TensorBinding(
            self._px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
        self._b_vel = TensorBinding(
            self._px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))
        self._b_root = TensorBinding(
            self._px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_ROOT_POSE))

    def _recreate_physics(self):
        """Tear down and recreate physics state to clear accumulated numerical drift."""
        for b in [self._b_pos, self._b_tgt, self._b_vel, self._b_root]:
            try: b.destroy()
            except Exception: pass
        try:
            self._px.detach_ovstage()
        except Exception:
            pass
        self._init_physics()
        PhysXEnvOv._total_resets += 1

    # ═══════════════════════════════════════════════════════════════════
    # Tensor read helpers — convert BFS→actuator order on read
    # ═══════════════════════════════════════════════════════════════════

    def _read_root_pose(self):
        """Return (root_pos[3], root_quat_wxyz[4]) from tensor binding."""
        rt = self._b_root.read().ravel()  # (7,) = [px, py, pz, qx, qy, qz, qw]
        pos = rt[0:3].astype(np.float64)
        quat_xyzw = rt[3:7]
        quat = _xyzw_to_wxyz(quat_xyzw)
        return pos, quat

    def _write_root_pose(self, pos, quat_wxyz):
        """Write root pose in xyzw format to tensor binding."""
        rt = np.zeros((1, 7), dtype=np.float32)
        rt[0, 0:3] = pos
        rt[0, 3:7] = _wxyz_to_xyzw(quat_wxyz)
        self._b_root.write(rt)

    def _read_joint_positions(self):
        """Read DOF positions, convert BFS→actuator order."""
        return self._b_pos.read().ravel()[ACT2DOF].astype(np.float64)

    def _read_joint_velocities(self):
        """Read DOF velocities, convert BFS→actuator order."""
        return self._b_vel.read().ravel()[ACT2DOF].astype(np.float64)

    def _write_joint_positions(self, qpos_actuator_order):
        """Write DOF positions, convert actuator→BFS order."""
        self._b_pos.write(qpos_actuator_order[DOF2ACT].astype(np.float32).reshape(1, -1))

    def _write_joint_velocities(self, qvel_actuator_order):
        """Write DOF velocities, convert actuator→BFS order."""
        self._b_vel.write(qvel_actuator_order[DOF2ACT].astype(np.float32).reshape(1, -1))

    def _write_joint_targets(self, target_actuator_order):
        """Write PD drive targets, convert actuator→BFS order."""
        self._b_tgt.write(target_actuator_order[DOF2ACT].astype(np.float32).reshape(1, -1))

    # ═══════════════════════════════════════════════════════════════════
    # Motion loading (identical to PhysXEnv / MuJoCoEnv)
    # ═══════════════════════════════════════════════════════════════════

    def _load_motions(self, pkl_dir):
        pkls = [p for p in glob.glob(os.path.join(pkl_dir, "**/*.pkl"), recursive=True)
                if not os.path.basename(p).startswith("._")]
        if not pkls:
            raise RuntimeError(f"No motion PKLs in {pkl_dir}")
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
            if len(motions) >= 500:
                break
        if not motions:
            raise RuntimeError(f"No motion PKLs in {pkl_dir}")
        return motions

    def _sample_motion(self):
        m = self.motions[np.random.randint(len(self.motions))]
        dof = m["dof"]; n = len(dof)
        self._ref_dof = dof
        self._ref_root_rot = m["root_rot"].astype(np.float64)
        self._ref_root_trans = m["root_trans_offset"].astype(np.float64)
        self._ref_fps = m.get("fps", 30.0)
        self._ref_dt = 1.0 / self._ref_fps
        max_time = (n - self.max_ep - 1) * self._ref_dt
        # Skip first ~100 frames (unstable PD init from static pose) and
        # leave 2s margin at end for safe tracking range
        min_frame = max(100, int(0.10 * n))
        min_time = min_frame * self._ref_dt
        if max_time > min_time + 1.0:
            safe_start = min(min_time, max_time - 1.0)
            safe_end = max(min_time + 1.0, max_time - 2.0)
            self._ref_time = np.random.uniform(safe_start, safe_end)
        else:
            self._ref_time = max(0.0, max_time * 0.5)

    def _advance_motion_time(self):
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
    # Body state via Python FK
    # ═══════════════════════════════════════════════════════════════════

    def _get_body_state_fk(self):
        """Return (body_pos, body_quat) for BODY_NAMES via Python FK, cached per step."""
        step_id = (id(self), self.ep, self._ref_time)
        if getattr(self, '_fk_cache_key', None) == step_id:
            return self._fk_body_pos, self._fk_body_quat
        root_pos, root_quat = self._read_root_pose()
        actual_qpos = self._read_joint_positions()
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
        self._write_joint_targets(target.astype(np.float64))

    def _physics_step(self):
        for _ in range(self.decimation):
            self._px.step(self.native_dt)
        self._px.wait_all()

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
        root_pos, root_quat = self._read_root_pose()
        g_body = quat_apply(quat_inv(root_quat), np.array([0, 0, -1]))
        # Angular velocity from quaternion difference (ovphysx has no direct root velocity tensor)
        ang_vel_world = quat_diff_to_angvel(
            self._prev_root_quat, root_quat, self.ctrl_dt)
        ang_vel = quat_apply(quat_inv(root_quat), ang_vel_world)
        self._prev_root_quat = root_quat.copy()

        jpos = self._read_joint_positions()
        jvel = self._read_joint_velocities()
        self._shift_histories()
        self._gdh[-1] = g_body; self._avh[-1] = ang_vel
        self._jph[-1] = jpos; self._jvh[-1] = jvel
        return np.concatenate([b.flatten() for b in
            [self._gdh, self._avh, self._jph, self._jvh, self._ah]]).astype(np.float32)

    def _compute_critic_obs(self):
        root_pos, root_quat = self._read_root_pose()
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
        root_pos, root_quat = self._read_root_pose()
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

        _, ori_b = subtract_frame_transforms(root_pos, root_quat, ref_root_pos, ref_root_quat)
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
    # Reward (identical math to PhysXEnv / MuJoCoEnv)
    # ═══════════════════════════════════════════════════════════════════

    def _compute_reward(self, action):
        root_pos, root_quat = self._read_root_pose()
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
        q = self._read_joint_positions()
        r8 = -10.0 * np.sum(np.maximum(np.abs(q - self.jm) - self.jh, 0))
        r9 = -0.1 * self._undesired_contact()
        r10 = -0.005 * self._anti_shake()
        r11 = 2.0 * self._vr_local_error()
        r12 = -2.5e-9 * self._feet_acc()
        r13 = self.alive_bonus

        return float(r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13)

    def _undesired_contact(self):
        return 0.0

    def _anti_shake(self):
        target = ("left_wrist_yaw_link", "right_wrist_yaw_link")
        excesses = []
        _, all_quat = self._get_body_state_fk()
        for name in target:
            bi = list(BODY_NAMES).index(name)
            curr_quat = all_quat[bi]
            prev_quat = self._prev_body_quat[bi]
            ang_vel = quat_diff_to_angvel(prev_quat, curr_quat, self.ctrl_dt)
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

        root_pos, root_quat = self._read_root_pose()
        all_pos, all_quat = self._get_body_state_fk()
        rob_bidx = [list(BODY_NAMES).index(n) for n in pt_bodies]
        rob_pt_w = all_pos[rob_bidx]
        rob_quat_w = all_quat[rob_bidx]
        rob_pt_w = rob_pt_w + quat_apply(rob_quat_w, pt_offsets)
        rob_local = quat_apply(quat_inv(np.tile(root_quat, (n_pts, 1))), rob_pt_w - root_pos)

        err = np.sum((rob_local - ref_local)**2)
        return float(np.exp(-err / (n_pts * 0.01)))

    def _feet_acc(self):
        ankle_indices = [4, 5, 10, 11]
        ankle_vel = self._read_joint_velocities()[ankle_indices]
        acc = (ankle_vel - self._prev_joint_vel[ankle_indices]) / self.ctrl_dt
        self._prev_joint_vel = self._read_joint_velocities().copy()
        return float(np.sum(acc**2))

    # ═══════════════════════════════════════════════════════════════════
    # Termination
    # ═══════════════════════════════════════════════════════════════════

    def _check_termination(self):
        ref_root_pos = self._ref_root_pos(); ref_root_quat = self._ref_root_quat()
        root_pos, root_quat = self._read_root_pose()
        ref_h = ref_root_pos[2]; root_h = root_pos[2]
        term = False; h_thresh = 0.75 if ref_h < 0.5 else 0.15

        _ORI_THRESH = 0.5
        _ANK_POS_THRESH = 0.5
        _ANK_H_MULT = 2.0

        if abs(ref_h - root_h) > h_thresh:
            term = True
        if quat_error_magnitude(ref_root_quat, root_quat)**2 > _ORI_THRESH:
            term = True

        actual_qpos = self._read_joint_positions()
        tracked = self._fk.get_tracked_poses(root_pos, root_quat, actual_qpos)
        actual_body_pos = np.array([t[0] for t in tracked], dtype=np.float64)
        ref_body_pos = self._ref_body_pos()

        for name in ("left_ankle_roll_link", "right_ankle_roll_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link"):
            idx = list(BODY_NAMES).index(name)
            err = abs(ref_body_pos[idx, 2] - actual_body_pos[idx][2])
            if err > h_thresh * _ANK_H_MULT:
                term = True; break

        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            idx = list(BODY_NAMES).index(name)
            ref_aligned = ref_body_pos[idx] - ref_root_pos + root_pos
            err = np.linalg.norm(ref_aligned - actual_body_pos[idx])
            if err > _ANK_POS_THRESH:
                term = True; break

        trunc = int(self._ref_time * self._ref_fps) >= len(self._ref_dof) - 1
        return term, trunc

    # ═══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════

    def get_current_ref_qpos(self):
        idx = min(int(self._ref_time * self._ref_fps), len(self._ref_dof) - 1)
        return self._ref_dof[idx].astype(np.float32)

    def reset(self):
        # Periodic physics recreation to clear accumulated numerical drift
        self._episode_count += 1
        if self._episode_count % self._recreate_every == 0:
            self._recreate_physics()

        # Try up to 5 motion frames to find a stable starting point
        MAX_RETRIES = 5
        for retry in range(MAX_RETRIES):
            self._sample_motion()
            idx = int(self._ref_time * self._ref_fps)
            ref_q0 = self._ref_dof[idx]

            # Write joint state only (NOT root pose — root stays at USD default to
            # avoid "Invalid PhysX transform" warnings from ovphysx root pose
            # manipulation).
            self._write_joint_positions(ref_q0.astype(np.float32))
            self._write_joint_velocities(np.zeros(self.nu, dtype=np.float32))
            self._write_joint_targets(ref_q0.astype(np.float32))

            # Short warmup for joint constraints to settle
            for _ in range(10):
                self._px.step(self.native_dt)
            self._px.wait_all()

            q = self._read_joint_positions()
            if not np.any(np.isnan(q)):
                break
        else:
            # All retries failed — recreate physics as last resort
            self._recreate_physics()
            self._sample_motion()
            idx = int(self._ref_time * self._ref_fps)
            ref_q0 = self._ref_dof[idx]
            self._write_joint_positions(ref_q0.astype(np.float32))
            self._write_joint_velocities(np.zeros(self.nu, dtype=np.float32))
            self._write_joint_targets(ref_q0.astype(np.float32))
            for _ in range(10):
                self._px.step(self.native_dt)
            self._px.wait_all()

        for b in (self._gdh, self._avh, self._jph, self._jvh, self._ah, self._lvh):
            b.fill(0)
        self._prev_action.fill(0)
        root_pos, root_quat = self._read_root_pose()
        self._prev_root_pos = root_pos.copy()
        self._prev_root_quat = root_quat.copy()
        fk_pos, fk_quat = self._get_body_state_fk()
        self._prev_body_pos = fk_pos.copy()
        self._prev_body_quat = fk_quat.copy()
        self._prev_ref_body_pos = self._prev_body_pos.copy()
        self._prev_ref_body_quat = self._prev_body_quat.copy()
        self._prev_joint_vel = self._read_joint_velocities()
        self.ep = 0
        self._compute_ref_body_state()
        return self._obs()

    def step(self, action):
        if action.ndim == 2:
            action = action[0]
        action = np.clip(action, -1, 1).astype(np.float64)
        self._ah[:-1] = self._ah[1:]; self._ah[-1] = action
        self._pd_control(action); self._physics_step()
        self._advance_motion_time()
        self._compute_ref_body_state()
        obs = self._obs()
        reward = self._compute_reward(action)

        # NaN guard + root drift guard (root Z drifts without ground contact,
        # causing numerical instability after ~100+ steps)
        root_pos, _ = self._read_root_pose()
        if np.isnan(reward) or np.any(np.isnan(obs["actor_obs"])) or abs(root_pos[2]) > 20.0:
            obs = self.reset()
            return obs, 0.0, False, {"time_outs": False, "terminal_obs": None, "_orig_done": False}

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

    def close(self):
        """Release tensor bindings and detach from physics."""
        for b in [self._b_pos, self._b_tgt, self._b_vel, self._b_root]:
            try:
                b.destroy()
            except Exception:
                pass
