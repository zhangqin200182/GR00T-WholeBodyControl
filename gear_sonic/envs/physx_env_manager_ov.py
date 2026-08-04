"""PhysX parallel environment manager using ovphysx — SHM + Barrier sync.

Manages N PhysXEnvOv instances across M worker processes.
Same SHM/Barrier protocol as MuJoCoEnvManager for PPO trainer compatibility.

Each worker process creates its own ovphysx.PhysX + ovstage.Stage and
loads the USD robot.  Environments within a worker run sequentially.
"""

import math, os, time, signal, logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from threading import BrokenBarrierError
import numpy as np
import torch

logger = logging.getLogger(__name__)

# SHM layout per env — identical to MuJoCoEnvManager
OBS_DIM = 4336       # actor(930) + critic(1645) + tokenizer(1761)
ACT_DIM = 29
OBS_BYTES = OBS_DIM * 4
ACT_BYTES = ACT_DIM * 4
REW_BYTES = 4
DONE_BYTES = 1
TIMEOUT_BYTES = 1
ORIG_DONE_BYTES = 1


class EnvSharedMemory:
    """Manages shared memory regions for ovphysx environments."""

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
        flat = np.concatenate([obs_dict["actor_obs"], obs_dict["critic_obs"], obs_dict["tokenizer"]])
        self._obs[env_id] = flat

    def write_terminal(self, env_id, obs_dict):
        if obs_dict is not None:
            flat = np.concatenate([obs_dict["actor_obs"], obs_dict["critic_obs"], obs_dict["tokenizer"]])
            self._terminal[env_id] = flat

    def write_action(self, env_id, action):
        self._actions[env_id] = action[:ACT_DIM]

    def write_reward(self, env_id, reward):
        self._rewards[env_id] = reward

    def write_done(self, env_id, done):
        self._dones[env_id] = int(done)

    def write_timeout(self, env_id, timeout):
        self._timeouts[env_id] = int(timeout)

    def write_orig_done(self, env_id, done):
        self._orig_dones[env_id] = int(done)

    def read_obs(self):
        return {
            "actor_obs": self._obs[:, :930].copy(),
            "critic_obs": self._obs[:, 930:2575].copy(),
            "tokenizer": self._obs[:, 2575:].copy(),
        }

    def read_terminal(self):
        return {
            "actor_obs": self._terminal[:, :930].copy(),
            "critic_obs": self._terminal[:, 930:2575].copy(),
            "tokenizer": self._terminal[:, 2575:].copy(),
        }

    def read_actions(self):
        return self._actions.copy()

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
        obj.num_envs = None
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
        f"module={type(x).__module__}"
    )


def _worker_loop(worker_id, start_env, num_envs, shm_names, barrier,
                 robot_usd_path, model_xml, pkl_dir, env_config=None):
    """Worker process entry point for ovphysx environments."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Import here — worker is a fresh process
    import ovphysx, ovstage
    from gear_sonic.envs.physx_env_ov import PhysXEnvOv

    # Attach shared memory
    shm = EnvSharedMemory.attach(shm_names)

    # Build numpy views for this worker's slice
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

    # Set LD_LIBRARY_PATH for ovphysx/ovstage shared libs
    for p in [
        "/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib",
        "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin",
        "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins",
        "/usr/lib/aarch64-linux-gnu",
    ]:
        e = os.environ.get("LD_LIBRARY_PATH", "")
        if p not in e:
            os.environ["LD_LIBRARY_PATH"] = f"{p}:{e}"

    # Create ovphysx environments (one PhysX + Stage per env for now)
    ovphysx.PhysX.set_cpu_mode(True)
    envs = []
    for i in range(num_envs):
        px = ovphysx.PhysX()
        stage = ovstage.Stage(f"worker{worker_id}_env{i}")
        env = PhysXEnvOv(px, stage, robot_usd_path, model_xml, pkl_dir, config=env_config)
        envs.append(env)

    # Initial reset
    for i, env in enumerate(envs):
        obs = env.reset()
        flat = np.concatenate([obs["actor_obs"], obs["critic_obs"], obs["tokenizer"]])
        obs_buf[i] = flat

    try:
        while True:
            # Wait for trainer to write actions
            barrier.wait()

            # Read actions for our envs
            actions = act_buf.copy()

            # Step each env
            for i, env in enumerate(envs):
                obs, reward, done, info = env.step(actions[i])
                flat = np.concatenate([obs["actor_obs"], obs["critic_obs"], obs["tokenizer"]])
                obs_buf[i] = flat
                rew_buf[i] = reward
                done_buf[i] = int(done)
                to_buf[i] = int(info.get("time_outs", False))
                orig_done_buf[i] = int(info.get("_orig_done", done))
                if info.get("terminal_obs") is not None:
                    tflat = np.concatenate([
                        info["terminal_obs"]["actor_obs"],
                        info["terminal_obs"]["critic_obs"],
                        info["terminal_obs"]["tokenizer"],
                    ])
                    terminal_buf[i] = tflat

            # Notify trainer
            barrier.wait()
    except (BrokenBarrierError, KeyboardInterrupt):
        pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


class PhysXEnvManagerOv:
    """Parallel ovphysx environment manager for SONIC PPO training.

    Same interface as MuJoCoEnvManager::

        manager = PhysXEnvManagerOv(num_envs=4, num_workers=2, ...)
        obs = manager.reset_all()
        while training:
            obs, rewards, dones, info = manager.step(actions)
    """

    BARRIER_TIMEOUT = 120
    TOTAL_OBS_DIM = OBS_DIM

    def __init__(self, num_envs, num_workers, robot_usd_path, model_xml,
                 pkl_dir, env_config=None):
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.robot_usd_path = robot_usd_path
        self.model_xml = model_xml
        self.pkl_dir = pkl_dir
        self.env_config = env_config

        # Calculate env distribution
        envs_per_worker = math.ceil(num_envs / num_workers)
        self._actual_workers = 0
        worker_args = []
        for i in range(num_workers):
            start = i * envs_per_worker
            n = min(envs_per_worker, num_envs - start)
            if n > 0:
                worker_args.append((i, start, n))
                self._actual_workers += 1

        # Create SHM
        self._shm = EnvSharedMemory(num_envs)

        # Create Barrier (N workers + 1 trainer)
        self._barrier = mp.Barrier(self._actual_workers + 1)

        # Spawn workers
        self._workers = []
        for worker_id, start, n in worker_args:
            p = mp.Process(
                target=_worker_loop,
                args=(worker_id, start, n, self._shm.names, self._barrier,
                      robot_usd_path, model_xml, pkl_dir, self.env_config),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Wait for all workers to finish initial reset (2 barriers: reset done + ready)
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

        logger.info(f"PhysXEnvManagerOv: {num_envs} envs x {self._actual_workers} workers")

    class _EnvStub:
        observation_space = {
            "policy": type("Space", (), {"shape": (930,)})(),
            "critic": type("Space", (), {"shape": (1645,)})(),
        }
        action_space = type("Space", (), {"shape": (29,)})()
        num_envs = 0

    # ── Stub methods for PPO trainer compatibility ─────────────────────

    def reset(self, flatten_dict_obs=False):
        _ = flatten_dict_obs
        return {"tokenizer": np.zeros((self.num_envs, 12, 1))}

    def reset_all(self, global_rank=0):
        obs = self._shm.read_obs()
        return {k: torch.from_numpy(v).float() for k, v in obs.items()}

    def set_is_evaluating(self, is_evaluating=True, log_info=False, **kwargs):
        self.is_evaluating = is_evaluating

    def set_is_training(self):
        pass

    def sync_and_compute_adaptive_sampling(self, *args, **kwargs):
        pass

    def resample_motion(self):
        pass

    def reinit_dr(self):
        pass

    def load_env_state_dict(self, state_dict):
        pass

    def get_env_state_dict(self):
        return {}

    def step(self, policy_state_dict):
        """Distribute actions, wait for workers, return results."""
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
                          self.robot_usd_path, self.model_xml, self.pkl_dir, self.env_config),
                    daemon=True)
                p.start()
                self._workers.append(p)

    def close(self):
        for p in self._workers:
            p.terminate()
            p.join(timeout=5)
        self._shm.close()
