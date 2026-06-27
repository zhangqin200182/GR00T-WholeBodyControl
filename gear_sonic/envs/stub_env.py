"""Stub environment for training SONIC on NPU without Isaac Sim.

Uses MotionLib directly (no isaaclab dependency) to provide realistic
motion data observations. Replaces ManagerEnvWrapper for the PPO trainer.
"""

from __future__ import annotations

import easydict
import numpy as np
import torch

from gear_sonic.isaac_utils import rotations
from gear_sonic.trl.utils import torch_transform
from gear_sonic.trl.utils.torch_transform import quat_apply, quat_inv, quat_mul
from gear_sonic.utils.motion_lib import motion_lib_robot


# ---------------------------------------------------------------------------
# Standalone rotation utilities (replaces isaaclab.utils.math functions)
# ---------------------------------------------------------------------------

def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w,x,y,z) to 3x3 rotation matrices."""
    w = quaternions[..., 0]
    x = quaternions[..., 1]
    y = quaternions[..., 2]
    z = quaternions[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    mat = torch.stack(
        [
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy),
        ],
        dim=-1,
    ).reshape(*quaternions.shape[:-1], 3, 3)
    return mat


def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor, q02: torch.Tensor
):
    """Compute (t02, q02) expressed in frame (t01, q01)."""
    q01_inv = quat_inv(q01)
    q_rel = quat_mul(q01_inv, q02)
    t_rel = quat_apply(q01_inv, t02 - t01)
    return t_rel, q_rel


def quat_apply_yaw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply only the yaw (heading) component of quaternion to a vector."""
    heading_q = torch_transform.get_heading_q(q.reshape(-1, 4)).reshape(q.shape)
    return quat_apply(heading_q, v)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_yaw_link",
]

VR_3POINT_BODY = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
VR_3POINT_BODY_OFFSET = [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]]

ANCHOR_BODY = "pelvis"
NUM_BODIES = len(BODY_NAMES)
NUM_DOF = 29
NUM_SMPL_JOINTS = 24

LOWER_JOINT_INDICES_MUJOCO = list(range(12))

# G1 joint/body mapping constants (from gear_sonic/envs/manager_env/robots/g1.py,
# hardcoded here to avoid isaaclab import chain)
G1_ISAACLAB_JOINTS = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
    "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
G1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
    21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]
G1_MUJOCO_TO_ISAACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
G1_ISAACLAB_TO_MUJOCO_BODY = [
    0, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 3, 6, 9, 12, 16,
    20, 22, 24, 26, 28, 13, 17, 21, 23, 25, 27, 29,
]
G1_MUJOCO_TO_ISAACLAB_BODY = [
    0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 5, 11, 17, 24,
    6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29,
]


class StubEnv:
    """Stub environment using real motion data without physics simulation.

    Provides the same interface as ManagerEnvWrapper for PPO training.
    """

    def __init__(self, config, env_config, device):
        self._device = device
        self._num_envs = config.num_envs
        self.is_manager_env = True
        self.is_evaluating = False
        self._extras = {"episode": {}, "to_log": {}, "log": {}}
        self.env = self  # trainer accesses env.env
        # Motion lib config
        motion_cfg = env_config.commands.motion
        motion_lib_cfg_raw = dict(motion_cfg.motion_lib_cfg)
        self._num_future_frames = motion_cfg.get("num_future_frames", 10)
        self._dt_future_ref_frames = motion_cfg.get("dt_future_ref_frames", 0.1)
        self._smpl_num_future_frames = motion_cfg.get("smpl_num_future_frames", self._num_future_frames)
        self._smpl_dt_future_ref_frames = motion_cfg.get("smpl_dt_future_ref_frames", self._dt_future_ref_frames)
        self._teleop_sample_prob_when_smpl = motion_cfg.get("teleop_sample_prob_when_smpl", 0.5)
        self._target_fps = motion_lib_cfg_raw.get("target_fps", 50)

        # Setup joint ordering (hardcoded G1 mappings, no isaaclab import needed)
        motion_lib_cfg_raw.update({
            "mujoco_to_isaaclab_body": G1_MUJOCO_TO_ISAACLAB_BODY,
            "mujoco_to_isaaclab_dof": G1_MUJOCO_TO_ISAACLAB_DOF,
            "isaaclab_to_mujoco_body": G1_ISAACLAB_TO_MUJOCO_BODY,
            "isaaclab_to_mujoco_dof": G1_ISAACLAB_TO_MUJOCO_DOF,
        })
        motion_lib_cfg_raw["body_indexes_data"] = [
            G1_ISAACLAB_JOINTS.index(name) for name in BODY_NAMES
        ]

        self._mujoco_to_isaaclab_dof = G1_MUJOCO_TO_ISAACLAB_DOF
        self._isaaclab_to_mujoco_dof = G1_ISAACLAB_TO_MUJOCO_DOF
        self._lower_joint_isaaclab_indices = [
            G1_ISAACLAB_TO_MUJOCO_DOF[i] for i in LOWER_JOINT_INDICES_MUJOCO
        ]

        # Load motion library
        ml_cfg = easydict.EasyDict(motion_lib_cfg_raw)
        self.motion_lib = motion_lib_robot.MotionLibRobot(ml_cfg, self._num_envs, device)
        max_num_motions = ml_cfg.get("max_num_motions", None)
        self.motion_lib.load_motions_for_training(max_num_seqs=max_num_motions)

        # Body indices
        self._anchor_body_index = BODY_NAMES.index(ANCHOR_BODY)
        self._vr_3point_indices = [BODY_NAMES.index(n) for n in VR_3POINT_BODY]
        self._vr_3point_offsets = (
            torch.tensor(VR_3POINT_BODY_OFFSET, dtype=torch.float32, device=device)
            .view(1, -1, 3)
            .repeat(self._num_envs, 1, 1)
        )
        self._down_dir = (
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
            .view(1, -1)
            .repeat(self._num_envs, 1)
        )

        # Timing
        self._frame_skips = self._dt_future_ref_frames // (1.0 / self._target_fps)
        self._smpl_frame_skips = self._smpl_dt_future_ref_frames // (1.0 / self._target_fps)

        self._future_time_steps_init = (
            (torch.arange(self._num_future_frames, device=device, dtype=torch.long) * self._frame_skips)
            .view(1, -1)
            .repeat(self._num_envs, 1)
        )
        self._smpl_future_time_steps_init = (
            (torch.arange(self._smpl_num_future_frames, device=device, dtype=torch.long) * self._smpl_frame_skips)
            .view(1, -1)
            .repeat(self._num_envs, 1)
        )

        # Motion state
        self.motion_ids = torch.zeros(self._num_envs, dtype=torch.long, device=device)
        self.time_steps = torch.zeros(self._num_envs, dtype=torch.long, device=device)
        self.motion_start_time_steps = torch.zeros(self._num_envs, dtype=torch.long, device=device)
        self.motion_num_steps = torch.zeros(self._num_envs, dtype=torch.long, device=device)

        # Encoder routing
        self._setup_encoder_routing(env_config)

        # History buffers
        self._history_length = config.get("actor_prop_history_length", 10)
        self._critic_history_length = config.get("critic_prop_history_length", 10)
        self._actions_history_length = config.get("actor_actions_history_length", 10)
        self._critic_actions_history_length = config.get("critic_actions_history_length", 10)

        # Actor obs components
        self._gravity_dir_history = torch.zeros(self._num_envs, self._history_length, 3, device=device)
        self._ang_vel_history = torch.zeros(self._num_envs, self._history_length, 3, device=device)
        self._joint_pos_history = torch.zeros(self._num_envs, self._history_length, NUM_DOF, device=device)
        self._joint_vel_history = torch.zeros(self._num_envs, self._history_length, NUM_DOF, device=device)
        self._action_history = torch.zeros(self._num_envs, self._actions_history_length, NUM_DOF, device=device)

        # Critic obs components (some shared with actor, some with different history)
        self._lin_vel_history = torch.zeros(self._num_envs, self._critic_history_length, 3, device=device)
        self._critic_ang_vel_history = torch.zeros(self._num_envs, self._critic_history_length, 3, device=device)
        self._critic_joint_pos_history = torch.zeros(self._num_envs, self._critic_history_length, NUM_DOF, device=device)
        self._critic_joint_vel_history = torch.zeros(self._num_envs, self._critic_history_length, NUM_DOF, device=device)
        self._critic_action_history = torch.zeros(self._num_envs, self._critic_actions_history_length, NUM_DOF, device=device)

        # Previous root quat (for angular velocity approximation)
        self._prev_root_quat = torch.zeros(self._num_envs, 4, device=device)
        self._prev_root_quat[:, 0] = 1.0  # identity quaternion
        self._prev_root_pos = torch.zeros(self._num_envs, 3, device=device)

        # Build config dict that PPO trainer reads
        self._build_config(config, env_config)

        # Do initial motion sampling so observation computation works
        all_ids = torch.arange(self._num_envs, device=device)
        self._resample_motion(all_ids)

        # Compute actor/critic obs dims by doing a trial observation
        trial_obs = self._compute_obs_dict(flatten_dict_obs=True)
        actor_dim = trial_obs["actor_obs"].shape[-1]
        critic_dim = trial_obs["critic_obs"].shape[-1]

        # observation_space / action_space (accessed by trainer as env.env.observation_space)
        import gymnasium as gym
        self.observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Box(-np.inf, np.inf, shape=(actor_dim,)),
            "critic": gym.spaces.Box(-np.inf, np.inf, shape=(critic_dim,)),
            "tokenizer": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(NUM_DOF,))

    def _setup_encoder_routing(self, env_config):
        """Setup encoder sampling probabilities from config."""
        try:
            encoder_sample_probs_dict = dict(
                env_config.commands.motion.encoder_sample_probs
            )
        except Exception:
            encoder_sample_probs_dict = {"g1": 1.0}

        self._encoder_names = list(encoder_sample_probs_dict.keys())
        self._num_encoders = len(self._encoder_names)
        probs = torch.tensor(
            [encoder_sample_probs_dict[k] for k in self._encoder_names],
            dtype=torch.float32,
            device=self._device,
        )
        self._encoder_sample_probs = probs / probs.sum()

        self._g1_encoder_idx = self._encoder_names.index("g1") if "g1" in self._encoder_names else 0
        self._smpl_encoder_idx = self._encoder_names.index("smpl") if "smpl" in self._encoder_names else None
        self._teleop_encoder_idx = self._encoder_names.index("teleop") if "teleop" in self._encoder_names else None

        # Build no-smpl probs (set smpl to 0, redistribute to g1)
        self._encoder_sample_probs_no_smpl = self._encoder_sample_probs.clone()
        if self._smpl_encoder_idx is not None:
            self._encoder_sample_probs_no_smpl[self._smpl_encoder_idx] = 0
            self._encoder_sample_probs_no_smpl[self._g1_encoder_idx] = 1.0
            remaining = self._encoder_sample_probs_no_smpl.sum()
            if remaining > 0:
                self._encoder_sample_probs_no_smpl /= remaining

        self.encoder_index = torch.zeros(
            self._num_envs, self._num_encoders, dtype=torch.long, device=self._device
        )

    def _build_config(self, config, env_config):
        """Build the config dict that the PPO trainer reads."""
        from omegaconf import OmegaConf

        self.config = OmegaConf.create({
            "num_envs": self._num_envs,
            "obs": {
                "obs_dims": {},
                "obs_dict": {},
                "group_obs_dims": {},
                "group_obs_names": {},
            },
            "robot": {
                "actions_dim": NUM_DOF,
                "num_joints": NUM_DOF,
                "algo_obs_dim_dict": {},
            },
            "rewards": {
                "num_critics": 1,
            },
            "tokenizer_action_dim": 64,
            "meta_action_dim": 64 + 2,
            "experiment_dir": config.get("experiment_dir", "outputs/stub_train"),
        })
        # Merge any extra config entries
        for key in ["save_rendering_dir", "experiment_dir"]:
            if key in env_config.get("config", {}):
                self.config[key] = env_config.config[key]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def device(self):
        return self._device

    @property
    def extras(self):
        return self._extras

    # ------------------------------------------------------------------
    # Motion state helpers
    # ------------------------------------------------------------------

    @property
    def _current_time(self):
        return self.motion_start_time_steps + self.time_steps

    @property
    def _future_motion_ids(self):
        return self.motion_ids.repeat_interleave(self._num_future_frames)

    @property
    def _future_time_steps(self):
        return (
            torch.clip(
                self._future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    @property
    def _smpl_future_motion_ids(self):
        return self.motion_ids.repeat_interleave(self._smpl_num_future_frames)

    @property
    def _smpl_future_time_steps(self):
        return (
            torch.clip(
                self._smpl_future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    def _robot_anchor_quat_w(self):
        """Robot anchor quaternion — uses motion root (perfect tracking)."""
        return self.motion_lib.get_root_quat_w(self.motion_ids, self._current_time)

    def _robot_anchor_pos_w(self):
        """Robot anchor position — uses motion root (perfect tracking)."""
        return self.motion_lib.get_root_pos_w(self.motion_ids, self._current_time)

    def _anchor_quat_w(self):
        """Reference motion anchor quat (same as robot in offline)."""
        return self.motion_lib.get_root_quat_w(self.motion_ids, self._current_time)

    def _anchor_pos_w(self):
        """Reference motion anchor position."""
        return self.motion_lib.get_body_pos_w(
            self.motion_ids, self._current_time
        )[:, self._anchor_body_index]

    # ------------------------------------------------------------------
    # Observation computation
    # ------------------------------------------------------------------

    def _compute_tokenizer_obs_dict(self):
        """Compute all 12 tokenizer observation terms as a dict."""
        N = self._num_envs
        F = self._num_future_frames
        SF = self._smpl_num_future_frames
        fmi = self._future_motion_ids
        fts = self._future_time_steps
        sfmi = self._smpl_future_motion_ids
        sfts = self._smpl_future_time_steps

        robot_anchor_quat = self._robot_anchor_quat_w()

        obs = {}

        # 1. encoder_index (N, num_encoders)
        obs["encoder_index"] = self.encoder_index.float()

        # 2. command_multi_future_nonflat (N, F, 2*29)
        joint_pos_mf = self.motion_lib.get_dof_pos(fmi, fts).view(N, F, NUM_DOF)
        joint_vel_mf = self.motion_lib.get_dof_vel(fmi, fts).view(N, F, NUM_DOF)
        obs["command_multi_future_nonflat"] = torch.cat([joint_pos_mf, joint_vel_mf], dim=-1)

        # 3. command_z_multi_future_nonflat (N, F, 1)
        root_pos_mf = self.motion_lib.get_root_pos_w(fmi, fts).view(N, F, 3)
        obs["command_z_multi_future_nonflat"] = root_pos_mf[..., 2:3]

        # 4. motion_anchor_ori_b_mf_nonflat (N, F, 6)
        ref_root_quat_mf = self.motion_lib.get_root_quat_w(fmi, fts).view(N, F, 4)
        robot_q_expanded = robot_anchor_quat.view(N, 1, 4).expand(N, F, 4)
        root_rot_dif = quat_mul(quat_inv(robot_q_expanded), ref_root_quat_mf)
        mat = matrix_from_quat(root_rot_dif)
        obs["motion_anchor_ori_b_mf_nonflat"] = mat[..., :2].reshape(N, F, 6)

        # 5. command_multi_future_lower_body (N, F*12*2)
        lower_pos = joint_pos_mf[..., self._lower_joint_isaaclab_indices]
        lower_vel = joint_vel_mf[..., self._lower_joint_isaaclab_indices]
        obs["command_multi_future_lower_body"] = torch.cat(
            [lower_pos.reshape(N, -1), lower_vel.reshape(N, -1)], dim=-1
        )

        # 6. vr_3point_local_target (N, 9)
        anchor_pos = self._anchor_pos_w()
        anchor_quat = self._anchor_quat_w()
        body_pos = self.motion_lib.get_body_pos_w(
            self.motion_ids, self._current_time
        )[:, self._vr_3point_indices]  # (N, 3, 3)
        body_quat = self.motion_lib.get_body_quat_w(
            self.motion_ids, self._current_time
        )[:, self._vr_3point_indices]  # (N, 3, 4)
        vr_pos_w = body_pos + quat_apply(body_quat, self._vr_3point_offsets)
        ref_root_quat_3p = anchor_quat.view(N, 1, 4).expand(N, 3, 4)
        ref_3point_diff = vr_pos_w - anchor_pos[:, None, :]
        obs["vr_3point_local_target"] = quat_apply(
            quat_inv(ref_root_quat_3p), ref_3point_diff
        ).view(N, -1)

        # 7. vr_3point_local_orn_target (N, 12)
        ref_3point_quat = quat_mul(quat_inv(ref_root_quat_3p), body_quat)
        obs["vr_3point_local_orn_target"] = ref_3point_quat.view(N, -1)

        # 8. motion_anchor_ori_b (N, 6)
        _, ori_rel = subtract_frame_transforms(
            self._robot_anchor_pos_w(), robot_anchor_quat,
            anchor_pos, anchor_quat,
        )
        mat_single = matrix_from_quat(ori_rel)
        obs["motion_anchor_ori_b"] = mat_single[..., :2].reshape(N, -1)

        # 9. command_z (N, 1)
        obs["command_z"] = self.motion_lib.get_root_pos_w(
            self.motion_ids, self._current_time
        )[:, 2:3]

        # 10-12: SMPL observations (zeros for non-SMPL motions)
        has_smpl = hasattr(self.motion_lib, "motion_has_smpl") and self.motion_lib.motion_has_smpl.any()

        if has_smpl:
            try:
                # smpl_joints_multi_future_local_nonflat (N, SF, num_joints*3)
                smpl_joints = self.motion_lib.get_smpl_joints(sfmi, sfts)
                smpl_joints = smpl_joints.view(N, SF, *smpl_joints.shape[1:])

                smpl_pose = self.motion_lib.get_smpl_pose(sfmi, sfts)
                smpl_root_quat = torch_transform.angle_axis_to_quaternion(
                    smpl_pose[..., :3]
                ).view(-1, 4)
                if getattr(self.motion_lib, "smpl_y_up", False):
                    smpl_root_quat = rotations.smpl_root_ytoz_up(smpl_root_quat)
                smpl_root_quat = rotations.remove_smpl_base_rot(smpl_root_quat, w_last=False)
                smpl_root_quat_mf = smpl_root_quat.view(N, SF, 4)

                smpl_root_quat_expanded = smpl_root_quat_mf.unsqueeze(-2).expand(
                    N, SF, smpl_joints.shape[-2], 4
                )
                smpl_joints_local = quat_apply(quat_inv(smpl_root_quat_expanded), smpl_joints)
                obs["smpl_joints_multi_future_local_nonflat"] = smpl_joints_local.reshape(N, SF, -1)

                # smpl_root_ori_b_multi_future (N, SF, 6)
                robot_q_smpl = robot_anchor_quat.view(N, 1, 4).expand(N, SF, 4)
                smpl_rot_dif = quat_mul(quat_inv(robot_q_smpl), smpl_root_quat_mf)
                mat_smpl = matrix_from_quat(smpl_rot_dif)
                obs["smpl_root_ori_b_multi_future"] = mat_smpl[..., :2].reshape(N, SF, 6)

                # joint_pos_multi_future_wrist_for_smpl (N, SF, 6)
                wrist_indices = [23, 24, 25, 26, 27, 28]
                smpl_joint_pos = self.motion_lib.get_dof_pos(sfmi, sfts).view(N, SF, NUM_DOF)
                obs["joint_pos_multi_future_wrist_for_smpl"] = smpl_joint_pos[..., wrist_indices]

            except Exception:
                has_smpl = False

        if not has_smpl:
            obs["smpl_joints_multi_future_local_nonflat"] = torch.zeros(
                N, SF, NUM_SMPL_JOINTS * 3, device=self._device
            )
            obs["smpl_root_ori_b_multi_future"] = torch.zeros(N, SF, 6, device=self._device)
            obs["joint_pos_multi_future_wrist_for_smpl"] = torch.zeros(N, SF, 6, device=self._device)

        return obs

    def _compute_actor_obs(self):
        """Compute proprioception-based actor observations with history."""
        N = self._num_envs
        robot_quat = self._robot_anchor_quat_w()

        # Gravity direction in body frame
        gravity_dir = quat_apply(quat_inv(robot_quat), self._down_dir)

        # Angular velocity (finite difference approximation)
        dq = quat_mul(robot_quat, quat_inv(self._prev_root_quat))
        ang_vel = 2.0 * dq[:, 1:4] * self._target_fps
        ang_vel = torch.clamp(ang_vel, -10.0, 10.0)

        # Joint state from motion lib (perfect tracking + noise)
        joint_pos = self.motion_lib.get_dof_pos(self.motion_ids, self._current_time)
        joint_vel = self.motion_lib.get_dof_vel(self.motion_ids, self._current_time)
        joint_pos = joint_pos + torch.randn_like(joint_pos) * 0.01
        joint_vel = joint_vel + torch.randn_like(joint_vel) * 0.5

        # Update histories (shift left, insert new at end)
        self._gravity_dir_history = torch.roll(self._gravity_dir_history, -1, dims=1)
        self._gravity_dir_history[:, -1] = gravity_dir + torch.randn_like(gravity_dir) * 0.05
        self._ang_vel_history = torch.roll(self._ang_vel_history, -1, dims=1)
        self._ang_vel_history[:, -1] = ang_vel + torch.randn_like(ang_vel) * 0.2
        self._joint_pos_history = torch.roll(self._joint_pos_history, -1, dims=1)
        self._joint_pos_history[:, -1] = joint_pos
        self._joint_vel_history = torch.roll(self._joint_vel_history, -1, dims=1)
        self._joint_vel_history[:, -1] = joint_vel

        self._prev_root_quat = robot_quat.clone()

        # Concatenate all with history: (N, history * dim_per_term)
        actor_obs = torch.cat([
            self._gravity_dir_history.reshape(N, -1),
            self._ang_vel_history.reshape(N, -1),
            self._joint_pos_history.reshape(N, -1),
            self._joint_vel_history.reshape(N, -1),
            self._action_history.reshape(N, -1),
        ], dim=-1)
        return actor_obs

    def _compute_critic_obs(self):
        """Compute critic observations (privileged info + history)."""
        N = self._num_envs
        robot_quat = self._robot_anchor_quat_w()
        robot_pos = self._robot_anchor_pos_w()

        # command_multi_future (N, 580) = (N, 10*29 + 10*29)
        fmi = self._future_motion_ids
        fts = self._future_time_steps
        joint_pos_mf = self.motion_lib.get_dof_pos(fmi, fts).view(N, -1)
        joint_vel_mf = self.motion_lib.get_dof_vel(fmi, fts).view(N, -1)
        command_mf = torch.cat([joint_pos_mf, joint_vel_mf], dim=-1)

        # motion_anchor_pos_b (N, 3)
        anchor_pos = self._anchor_pos_w()
        anchor_quat = self._anchor_quat_w()
        pos_b, _ = subtract_frame_transforms(robot_pos, robot_quat, anchor_pos, anchor_quat)

        # motion_anchor_ori_b (N, 6)
        _, ori_b = subtract_frame_transforms(robot_pos, robot_quat, anchor_pos, anchor_quat)
        mat = matrix_from_quat(ori_b)
        ori_b_6d = mat[..., :2].reshape(N, -1)

        # body_pos (N, 42) and body_ori (N, 84) — robot body in body frame
        body_pos_w = self.motion_lib.get_body_pos_w(self.motion_ids, self._current_time)
        body_quat_w = self.motion_lib.get_body_quat_w(self.motion_ids, self._current_time)
        rp_expanded = robot_pos[:, None, :].expand_as(body_pos_w)
        rq_expanded = robot_quat[:, None, :].expand_as(body_quat_w)
        body_pos_b, body_quat_b = subtract_frame_transforms(
            rp_expanded, rq_expanded, body_pos_w, body_quat_w
        )
        body_pos_flat = body_pos_b.view(N, -1)
        body_ori_mat = matrix_from_quat(body_quat_b)
        body_ori_flat = body_ori_mat[..., :2].reshape(N, -1)

        # Linear velocity (finite difference)
        lin_vel = (robot_pos - self._prev_root_pos) * self._target_fps
        lin_vel = torch.clamp(lin_vel, -10.0, 10.0)
        self._prev_root_pos = robot_pos.clone()

        # Update critic histories
        joint_pos_cur = self.motion_lib.get_dof_pos(self.motion_ids, self._current_time)
        joint_vel_cur = self.motion_lib.get_dof_vel(self.motion_ids, self._current_time)
        ang_vel = self._ang_vel_history[:, -1]

        self._lin_vel_history = torch.roll(self._lin_vel_history, -1, dims=1)
        self._lin_vel_history[:, -1] = lin_vel
        self._critic_ang_vel_history = torch.roll(self._critic_ang_vel_history, -1, dims=1)
        self._critic_ang_vel_history[:, -1] = ang_vel
        self._critic_joint_pos_history = torch.roll(self._critic_joint_pos_history, -1, dims=1)
        self._critic_joint_pos_history[:, -1] = joint_pos_cur
        self._critic_joint_vel_history = torch.roll(self._critic_joint_vel_history, -1, dims=1)
        self._critic_joint_vel_history[:, -1] = joint_vel_cur

        critic_obs = torch.cat([
            command_mf,
            pos_b.view(N, -1),
            ori_b_6d,
            body_pos_flat,
            body_ori_flat,
            self._lin_vel_history.reshape(N, -1),
            self._critic_ang_vel_history.reshape(N, -1),
            self._critic_joint_pos_history.reshape(N, -1),
            self._critic_joint_vel_history.reshape(N, -1),
            self._critic_action_history.reshape(N, -1),
        ], dim=-1)
        return critic_obs

    def _compute_obs_dict(self, flatten_dict_obs=True):
        """Compute the full observation dict."""
        actor_obs = self._compute_actor_obs()
        critic_obs = self._compute_critic_obs()
        tokenizer_obs_dict = self._compute_tokenizer_obs_dict()

        obs_dict = {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
        }

        if flatten_dict_obs:
            tokenizer_names = list(tokenizer_obs_dict.keys())
            flat_parts = [
                tokenizer_obs_dict[k].reshape(self._num_envs, -1) for k in tokenizer_names
            ]
            obs_dict["tokenizer"] = torch.cat(flat_parts, dim=-1)
        else:
            obs_dict["tokenizer"] = tokenizer_obs_dict

        return obs_dict

    # ------------------------------------------------------------------
    # Encoder routing
    # ------------------------------------------------------------------

    def _resample_encoder(self, env_ids):
        """Resample encoder index for given envs based on motion_has_smpl."""
        if self._num_encoders <= 1:
            self.encoder_index[env_ids] = 0
            self.encoder_index[env_ids, 0] = 1
            return

        has_smpl = self.motion_lib.motion_has_smpl[self.motion_ids[env_ids]]
        smpl_ids = env_ids[has_smpl]
        no_smpl_ids = env_ids[~has_smpl]

        # Sample for envs with SMPL data
        if len(smpl_ids) > 0:
            enc_idx = torch.multinomial(
                self._encoder_sample_probs, len(smpl_ids), replacement=True
            ).to(self._device)
            self.encoder_index[smpl_ids] = 0
            self.encoder_index[smpl_ids, enc_idx] = 1

        # Sample for envs without SMPL data
        if len(no_smpl_ids) > 0:
            enc_idx = torch.multinomial(
                self._encoder_sample_probs_no_smpl, len(no_smpl_ids), replacement=True
            ).to(self._device)
            self.encoder_index[no_smpl_ids] = 0
            self.encoder_index[no_smpl_ids, enc_idx] = 1

        # SMPL-native envs also activate G1 encoder (for latent alignment)
        if self._smpl_encoder_idx is not None:
            use_smpl = self.encoder_index[env_ids, self._smpl_encoder_idx].bool()
            smpl_env_ids = env_ids[use_smpl]
            if len(smpl_env_ids) > 0:
                self.encoder_index[smpl_env_ids, self._g1_encoder_idx] = 1
                # Also sample teleop with probability
                if self._teleop_encoder_idx is not None and self._teleop_sample_prob_when_smpl > 0:
                    sample_teleop = (
                        torch.rand(len(smpl_env_ids), device=self._device)
                        < self._teleop_sample_prob_when_smpl
                    )
                    self.encoder_index[smpl_env_ids, self._teleop_encoder_idx] = (
                        self.encoder_index[smpl_env_ids, self._teleop_encoder_idx]
                        | sample_teleop.long()
                    )

    # ------------------------------------------------------------------
    # Motion sampling
    # ------------------------------------------------------------------

    def _resample_motion(self, env_ids):
        """Sample new motion clips for given environment indices."""
        if len(env_ids) == 0:
            return

        self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
        self.motion_start_time_steps[env_ids] = self.motion_lib.sample_time_steps(
            self.motion_ids[env_ids], truncate_time=None
        ).long()
        self.time_steps[env_ids] = 0
        self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
            self.motion_ids[env_ids]
        ).long()
        self._resample_encoder(env_ids)

        # Reset histories for resampled envs
        self._gravity_dir_history[env_ids] = 0
        self._ang_vel_history[env_ids] = 0
        self._joint_pos_history[env_ids] = 0
        self._joint_vel_history[env_ids] = 0
        self._action_history[env_ids] = 0
        self._lin_vel_history[env_ids] = 0
        self._critic_ang_vel_history[env_ids] = 0
        self._critic_joint_pos_history[env_ids] = 0
        self._critic_joint_vel_history[env_ids] = 0
        self._critic_action_history[env_ids] = 0

        # Reset prev root state
        self._prev_root_quat[env_ids] = self.motion_lib.get_root_quat_w(
            self.motion_ids[env_ids],
            self.motion_start_time_steps[env_ids],
        )
        self._prev_root_pos[env_ids] = self.motion_lib.get_root_pos_w(
            self.motion_ids[env_ids],
            self.motion_start_time_steps[env_ids],
        )

    # ------------------------------------------------------------------
    # Environment interface (matches ManagerEnvWrapper API)
    # ------------------------------------------------------------------

    def reset_all(self, global_rank=0):
        return self.reset()

    def reset(self, flatten_dict_obs=True):
        all_ids = torch.arange(self._num_envs, device=self._device)
        self._resample_motion(all_ids)
        obs_dict = self._compute_obs_dict(flatten_dict_obs=flatten_dict_obs)
        return obs_dict

    def step(self, policy_state_dict):
        # Extract action for history tracking
        if isinstance(policy_state_dict, dict) and "actions" in policy_state_dict:
            actions = policy_state_dict["actions"]
            if actions.shape[-1] >= NUM_DOF:
                self._action_history = torch.roll(self._action_history, -1, dims=1)
                self._action_history[:, -1] = actions[:, :NUM_DOF]
                self._critic_action_history = torch.roll(self._critic_action_history, -1, dims=1)
                self._critic_action_history[:, -1] = actions[:, :NUM_DOF]

        # Advance time
        self.time_steps += 1

        # Check for episode end
        remaining = self.motion_num_steps - self.motion_start_time_steps - self.time_steps
        done_mask = remaining <= self._num_future_frames
        random_early_term = torch.rand(self._num_envs, device=self._device) < 0.002
        dones = (done_mask | random_early_term).float()

        # Time out (no early termination failure)
        time_outs = done_mask.float()

        # Resample done envs
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self._resample_motion(done_ids)

        # Compute observations
        obs_dict = self._compute_obs_dict(flatten_dict_obs=True)

        # Compute simple reward (small random — no physics-based reward possible)
        rewards = torch.randn(self._num_envs, device=self._device) * 0.1

        infos = {
            "episode": {},
            "time_outs": time_outs,
            "to_log": {},
            "log": {},
        }

        return obs_dict, rewards, dones, infos

    # ------------------------------------------------------------------
    # No-op methods (required by PPO trainer / callbacks)
    # ------------------------------------------------------------------

    def set_is_evaluating(self, is_evaluating=True, **kwargs):
        self.is_evaluating = is_evaluating

    def set_is_training(self):
        self.is_evaluating = False

    def resample_motion(self):
        pass

    def sync_and_compute_adaptive_sampling(self, *args, **kwargs):
        pass

    def load_env_state_dict(self, state_dict):
        pass

    def get_env_state_dict(self):
        return {}

    def reinit_dr(self):
        pass

    def render_results(self, *args, **kwargs):
        pass

    def end_render_results(self, *args, **kwargs):
        pass
