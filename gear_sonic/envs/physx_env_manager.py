"""PhysX parallel environment manager — SHM + Barrier sync.

Manages N PhysXEnv instances across M worker processes.
Implements the same step(actions) → (obs, rewards, dones, info)
interface as MuJoCoEnvManager for PPO trainer compatibility.

Key differences from MuJoCoEnvManager:
  - Workers import physx_core + call init_foundation() after fork
  - Each env gets its own PxScene + PxArticulation + FK
  - NaN guard on env.step() output (eACCELERATION can explode numerically)
  - torch imported lazily (not at module level — avoids HCCL fork inheritance)
"""
import math, os, time, signal, logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from threading import BrokenBarrierError
import numpy as np

logger = logging.getLogger(__name__)

# SHM layout per env — identical to mujoco_env_manager.py
OBS_DIM = 4365       # actor(930) + critic(1645) + tokenizer(1761) + ref_action(29)
ACT_DIM = 29
OBS_BYTES = OBS_DIM * 4       # float32
ACT_BYTES = ACT_DIM * 4
REW_BYTES = 4                 # float32
DONE_BYTES = 1                # uint8
TIMEOUT_BYTES = 1             # uint8
ORIG_DONE_BYTES = 1           # uint8 — _orig_done for ignore_terminations


class EnvSharedMemory:
    """Manages 7 shared memory regions. Names are passed to workers as strings."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self._shm = {}
        self._names = {}

        layouts = {
            "obs": (num_envs * OBS_BYTES, np.float32, (num_envs, OBS_DIM)),
            "terminal": (num_envs * OBS_BYTES, np.float32, (num_envs, OBS_DIM)),
            "actions": (num_envs * ACT_BYTES, np.float32, (num_envs, ACT_DIM)),
            "rewards": (num_envs * REW_BYTES, np.float32, (num_envs,)),
            "dones": (num_envs * DONE_BYTES, np.uint8, (num_envs,)),
            "timeouts": (num_envs * TIMEOUT_BYTES, np.uint8, (num_envs,)),
            "orig_dones": (num_envs * ORIG_DONE_BYTES, np.uint8, (num_envs,)),
        }
        for name, (size, dtype, shape) in layouts.items():
            shm = SharedMemory(create=True, size=size)
            self._shm[name] = shm
            self._names[name] = shm.name
            setattr(self, f"_{name}", np.ndarray(shape, dtype=dtype, buffer=shm.buf))

    @property
    def names(self):
        return self._names.copy()

    def write_obs(self, env_id, obs_dict):
        flat = np.concatenate([obs_dict["actor_obs"], obs_dict["critic_obs"], obs_dict["tokenizer"], obs_dict["ref_action"]])
        self._obs[env_id] = flat

    def write_terminal(self, env_id, obs_dict):
        if obs_dict is not None:
            flat = np.concatenate([obs_dict["actor_obs"], obs_dict["critic_obs"], obs_dict["tokenizer"], obs_dict["ref_action"]])
            self._terminal[env_id] = flat

    def read_obs(self):
        return {
            "actor_obs": self._obs[:, :930].copy(),
            "critic_obs": self._obs[:, 930:2575].copy(),
            "tokenizer": self._obs[:, 2575:4336].copy(),
            "ref_action": self._obs[:, 4336:4365].copy(),
        }

    def read_terminal(self):
        return {
            "actor_obs": self._terminal[:, :930].copy(),
            "critic_obs": self._terminal[:, 930:2575].copy(),
            "tokenizer": self._terminal[:, 2575:4336].copy(),
            "ref_action": self._terminal[:, 4336:4365].copy(),
        }

    def read_rewards(self):
        return np.ndarray(self.num_envs, dtype=np.float32, buffer=self._rewards.data).copy()

    def read_dones(self):
        return np.ndarray(self.num_envs, dtype=np.bool_, buffer=self._dones.data).copy()

    def read_timeouts(self):
        return np.ndarray(self.num_envs, dtype=np.bool_, buffer=self._timeouts.data).copy()

    def read_orig_dones(self):
        return np.ndarray(self.num_envs, dtype=np.bool_, buffer=self._orig_dones.data).copy()

    def close(self):
        for shm in self._shm.values():
            shm.close()
            shm.unlink()

    @staticmethod
    def attach(names):
        """Worker-side: re-attach via names dict."""
        obj = EnvSharedMemory.__new__(EnvSharedMemory)
        obj.num_envs = None  # set by caller
        obj._shm = {}
        obj._names = names
        layouts = {
            "obs": OBS_BYTES, "terminal": OBS_BYTES,
            "actions": ACT_BYTES, "rewards": REW_BYTES,
            "dones": DONE_BYTES, "timeouts": TIMEOUT_BYTES,
            "orig_dones": ORIG_DONE_BYTES,
        }
        for name, stride in layouts.items():
            shm = SharedMemory(name=names[name])
            obj._shm[name] = shm
        return obj


def _to_numpy(x):
    """Convert TensorDict / NPU tensor / torch.Tensor → np.ndarray."""
    import torch
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "storage"):
        try:
            raw = x.storage()
            if isinstance(raw, np.ndarray):
                r = raw.reshape(x.shape) if hasattr(x, "shape") else raw
                if isinstance(r, np.ndarray):
                    return r
            if isinstance(raw, torch.Tensor):
                arr = raw.detach().cpu().numpy()
                r = arr.reshape(x.shape) if hasattr(x, "shape") else arr
                if isinstance(r, np.ndarray):
                    return r
        except Exception:
            pass
    if hasattr(x, "to"):
        try:
            tmp = x.to("cpu")
            if hasattr(tmp, "numpy"):
                r = tmp.numpy()
                if isinstance(r, np.ndarray):
                    return r
                if isinstance(r, dict) and "actions" in r:
                    a = r["actions"]
                    if isinstance(a, np.ndarray):
                        return a
        except Exception:
            pass
    if hasattr(x, "__iter__") and hasattr(x, "__len__"):
        try:
            r = np.fromiter(x, dtype=np.float32)
            if isinstance(r, np.ndarray):
                return r
        except Exception:
            pass
    if isinstance(x, dict) and "actions" in x:
        r = _to_numpy(x["actions"])
        if isinstance(r, np.ndarray):
            return r
    raise TypeError(
        f"Cannot convert actions to numpy: type={type(x)}, "
        f"module={type(x).__module__}, "
        f"has_cpu={hasattr(x, 'cpu')}, has_numpy={hasattr(x, 'numpy')}, "
        f"has_storage={hasattr(x, 'storage')}, has_to={hasattr(x, 'to')}"
    )


def _worker_loop(worker_id, start_env, num_envs, shm_names, barrier,
                 model_xml, pkl_dir, env_config=None,
                 static_pose=False, root_z_offset=0.0, standing_prob=0.0):
    """Worker process entry point.

    Each worker:
    1. Imports physx_core + init_foundation() (MUST run before any PhysX API call)
    2. Creates N PhysXEnv instances (each with independent Scene + Articulation + FK)
    3. Barriers-syncs with trainer: wait→step→wait loop

    To keep per-step latency low, keep num_envs per worker ≤ 8.
    Create many workers (512+) via physx_workers config instead.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # ── PhysX init (per-process, after fork) ──
    import sys
    _build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'physx', 'build')
    sys.path.insert(0, _build_dir)
    import physx_core
    px = physx_core
    px.init_foundation()

    # ── Delayed imports (depend on physx_core being available) ──
    from gear_sonic.envs.physx_env import PhysXEnv

    # ── Attach shared memory ──
    shm = EnvSharedMemory.attach(shm_names)

    # Rebuild numpy views for this worker's slice
    obs_buf = np.ndarray((num_envs, OBS_DIM), dtype=np.float32,
                         buffer=shm._shm["obs"].buf, offset=start_env * OBS_BYTES)
    terminal_buf = np.ndarray((num_envs, OBS_DIM), dtype=np.float32,
                              buffer=shm._shm["terminal"].buf, offset=start_env * OBS_BYTES)
    act_buf = np.ndarray((num_envs, ACT_DIM), dtype=np.float32,
                         buffer=shm._shm["actions"].buf, offset=start_env * ACT_BYTES)
    rew_buf = np.ndarray(num_envs, dtype=np.float32,
                         buffer=shm._shm["rewards"].buf, offset=start_env * REW_BYTES)
    done_buf = np.ndarray(num_envs, dtype=np.uint8,
                          buffer=shm._shm["dones"].buf, offset=start_env * DONE_BYTES)
    to_buf = np.ndarray(num_envs, dtype=np.uint8,
                        buffer=shm._shm["timeouts"].buf, offset=start_env * TIMEOUT_BYTES)
    orig_done_buf = np.ndarray(num_envs, dtype=np.uint8,
                               buffer=shm._shm["orig_dones"].buf, offset=start_env * ORIG_DONE_BYTES)

    # ── Create envs ──
    # Drive/contact stack is pinned by the LAUNCH SCRIPT via env vars
    # (E8: SONIC_PHYSX_DRIVE_TYPE=FORCE etc. — explicit, not defaults).
    # Legacy defaults here preserve old-train reproducibility.
    _drive_type = os.environ.get("SONIC_PHYSX_DRIVE_TYPE", "ACCELERATION")
    _vel_iters = int(os.environ.get("SONIC_PHYSX_VEL_ITERS", "1"))
    _native_dt = float(os.environ.get("SONIC_PHYSX_NATIVE_DT", "0.001961"))
    _decimation = int(os.environ.get("SONIC_PHYSX_DECIMATION", "10"))
    _root_z_offset = float(os.environ.get("SONIC_PHYSX_ROOT_Z_OFFSET",
                                          str(root_z_offset)))
    envs = [PhysXEnv(px, model_xml, pkl_dir, config=env_config,
                     native_dt=_native_dt, decimation=_decimation,
                     pos_iters=8, vel_iters=_vel_iters,
                     static_pose=static_pose, root_z_offset=_root_z_offset,
                     standing_prob=standing_prob, drive_type=_drive_type)
            for _ in range(num_envs)]

    # ── Initial reset ──
    for i, env in enumerate(envs):
        obs = env.reset()
        obs_buf[i] = np.concatenate([obs["actor_obs"], obs["critic_obs"], obs["tokenizer"], obs["ref_action"]])

    while True:
        barrier.wait()

        actions = act_buf.copy()

        for i, env in enumerate(envs):
            obs, reward, done, info = env.step(actions[i])

            if np.isnan(reward) or any(np.isnan(v).any() for v in obs.values()):
                obs = env.reset()
                reward = 0.0
                done = True
                info = {"time_outs": False, "_orig_done": True}

            flat = np.concatenate([obs["actor_obs"], obs["critic_obs"], obs["tokenizer"], obs["ref_action"]])
            obs_buf[i] = flat
            rew_buf[i] = reward
            done_buf[i] = int(done)
            to_buf[i] = int(info.get("time_outs", False))
            orig_done_buf[i] = int(info.get("_orig_done", done))
            if info.get("terminal_obs") is not None:
                terminal_buf[i] = np.concatenate([
                    info["terminal_obs"]["actor_obs"],
                    info["terminal_obs"]["critic_obs"],
                    info["terminal_obs"]["tokenizer"],
                    info["terminal_obs"]["ref_action"],
                ])

        barrier.wait()


class PhysXEnvManager:
    """Parallel PhysX environment manager.

    Presents the same step(actions) → (obs, rewards, dones, info) interface
    as MuJoCoEnvManager so the PPO trainer can use it without changes.
    """

    BARRIER_TIMEOUT = 120
    TOTAL_OBS_DIM = OBS_DIM

    def __init__(self, num_envs, num_workers, model_xml, pkl_dir, env_config=None,
                 static_pose=False, root_z_offset=0.0, standing_prob=0.0):
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.model_xml = model_xml
        self.pkl_dir = pkl_dir
        self.env_config = env_config
        self.static_pose = static_pose
        self.root_z_offset = root_z_offset
        self.standing_prob = standing_prob
        self.is_evaluating = False

        # Distribute envs across workers
        envs_per_worker = math.ceil(num_envs / num_workers)
        self._actual_workers = 0
        worker_args = []
        for i in range(num_workers):
            start = i * envs_per_worker
            n = min(envs_per_worker, num_envs - start)
            if n > 0:
                worker_args.append((i, start, n))
                self._actual_workers += 1

        if self._actual_workers == 0:
            raise ValueError(f"num_envs={num_envs} too small for num_workers={num_workers}")

        # Create SHM
        self._shm = EnvSharedMemory(num_envs)

        # Create Barrier (N workers + 1 trainer)
        self._barrier = mp.Barrier(self._actual_workers + 1)

        # Spawn workers (before torch/HCCL init in trainer script)
        self._workers = []
        for worker_id, start, n in worker_args:
            p = mp.Process(
                target=_worker_loop,
                args=(worker_id, start, n, self._shm.names, self._barrier,
                      model_xml, pkl_dir, env_config, self.static_pose, self.root_z_offset,
                      self.standing_prob),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Barrier sync: wait for all workers to finish initial reset
        try:
            self._barrier.wait(timeout=self.BARRIER_TIMEOUT)
            self._barrier.wait(timeout=self.BARRIER_TIMEOUT)
        except BrokenBarrierError:
            logger.error("Worker crashed during init!")
            self.close()
            raise RuntimeError("Worker crash during init")

        # Stub inner env for trainer compatibility
        self.env = self._EnvStub()
        self.extras = {}

        envs_per_w = math.ceil(num_envs / self._actual_workers) if self._actual_workers else 0
        logger.info(f"PhysXEnvManager: {num_envs} envs × {self._actual_workers} workers ({envs_per_w} envs/worker)")

    class _EnvStub:
        observation_space = {
            "policy": type("Space", (), {"shape": (930,)})(),
            "critic": type("Space", (), {"shape": (1645,)})(),
        }
        action_space = type("Space", (), {"shape": (29,)})()

    # ── Trainer-facing API ──────────────────────────────────────────────

    def reset(self, flatten_dict_obs=False):
        _ = flatten_dict_obs
        return {"tokenizer": np.zeros((self.num_envs, 12, 1))}

    def reset_all(self, global_rank=0):
        import torch
        obs = self._shm.read_obs()
        return {k: torch.from_numpy(v).float() for k, v in obs.items()}

    def step(self, policy_state_dict):
        """Trainer-side: distribute actions, wait, return results.

        Accepts both policy_state_dict (from ppo_trainer) and raw np.ndarray.
        Raises RuntimeError on worker crash — caller must discard current rollout.
        """
        import torch

        actions = _to_numpy(policy_state_dict if not isinstance(policy_state_dict, dict)
                            else policy_state_dict["actions"])
        act_buf = np.ndarray((self.num_envs, ACT_DIM), dtype=np.float32,
                             buffer=self._shm._actions.data)
        act_buf[:] = actions

        try:
            self._barrier.wait(timeout=self.BARRIER_TIMEOUT)
            self._barrier.wait(timeout=self.BARRIER_TIMEOUT)
        except BrokenBarrierError:
            logger.error("Worker crashed! Discarding current rollout, re-spawning...")
            self._handle_worker_crash()
            raise RuntimeError(
                "Worker crash during step — current rollout data is stale, "
                "trainer must discard and restart rollout from reset obs"
            )

        obs = self._shm.read_obs()
        rewards = self._shm.read_rewards()
        dones = self._shm.read_dones()
        timeouts = self._shm.read_timeouts()
        orig_dones = self._shm.read_orig_dones()
        terminal_obs = self._shm.read_terminal()

        return {k: torch.from_numpy(v).float() for k, v in obs.items()}, \
               torch.from_numpy(rewards).float(), \
               torch.from_numpy(dones).bool(), \
               {"time_outs": torch.from_numpy(timeouts).bool(),
                "_orig_done": torch.from_numpy(orig_dones).bool(),
                "episode": {},
                "to_log": {},
                "terminal_obs": {k: torch.from_numpy(v).float()
                                 for k, v in terminal_obs.items()} if terminal_obs is not None else None}

    # ── Stub methods for PPO trainer compatibility ───────────────────────

    def set_is_evaluating(self, is_evaluating=True, log_info=False, **kwargs):
        self.is_evaluating = is_evaluating

    def set_is_training(self, **_kwargs):
        self.is_evaluating = False

    def sync_and_compute_adaptive_sampling(self, *args, **kwargs):
        pass

    def resample_motion(self):
        pass

    def reinit_dr(self, **_kwargs):
        pass

    def load_env_state_dict(self, state_dict):
        pass

    def get_env_state_dict(self):
        return {}

    # ── Crash recovery ──────────────────────────────────────────────────

    def _handle_worker_crash(self):
        for p in self._workers:
            p.terminate()
            p.join(timeout=5)
        self._barrier = mp.Barrier(self._actual_workers + 1)
        self._workers = []
        envs_per_worker = math.ceil(self.num_envs / self.num_workers)
        for i in range(self.num_workers):
            start = i * envs_per_worker
            n = min(envs_per_worker, self.num_envs - start)
            if n > 0:
                p = mp.Process(target=_worker_loop,
                               args=(i, start, n, self._shm.names, self._barrier,
                                     self.model_xml, self.pkl_dir, self.env_config,
                                     self.static_pose, self.root_z_offset,
                                     self.standing_prob),
                               daemon=True)
                p.start()
                self._workers.append(p)

    def close(self):
        for p in self._workers:
            p.terminate()
            p.join(timeout=5)
        self._shm.close()
