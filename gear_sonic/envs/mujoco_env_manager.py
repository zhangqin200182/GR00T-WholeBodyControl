"""MuJoCo parallel environment manager — SHM + Barrier sync.

Manages N MuJoCoEnv instances across M worker processes.
Implements the same step(actions) → (obs, rewards, dones, info)
interface as ManagerEnvWrapper for PPO trainer compatibility.
"""
import math, os, time, signal, logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from collections import namedtuple
import numpy as np

from gear_sonic.envs.mujoco_env import MuJoCoEnv

logger = logging.getLogger(__name__)

# SHM layout per env
OBS_DIM = 4336       # actor(930) + critic(1645) + tokenizer(1761)
ACT_DIM = 29
OBS_BYTES = OBS_DIM * 4       # float32
ACT_BYTES = ACT_DIM * 4
REW_BYTES = 4                 # float32
DONE_BYTES = 1                # uint8
TIMEOUT_BYTES = 1             # uint8


class EnvSharedMemory:
    """Manages 6 shared memory regions. Names are passed to workers as strings."""

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
        }
        for name, (size, dtype, shape) in layouts.items():
            shm = SharedMemory(create=True, size=size)
            self._shm[name] = shm
            self._names[name] = shm.name
            # Create numpy view for fast access
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

    def read_obs(self):
        return {
            "actor_obs": self._obs[:, :930].copy(),
            "critic_obs": self._obs[:, 930:2575].copy(),
            "tokenizer": self._obs[:, 2575:].copy(),
        }

    def read_terminal(self):
        """Return terminal obs as dict of slices."""
        return {
            "actor_obs": self._terminal[:, :930].copy(),
            "critic_obs": self._terminal[:, 930:2575].copy(),
            "tokenizer": self._terminal[:, 2575:].copy(),
        }

    def read_actions(self):
        return np.ndarray((self.num_envs, ACT_DIM), dtype=np.float32, buffer=self._actions.data).copy()

    def read_rewards(self):
        return np.ndarray(self.num_envs, dtype=np.float32, buffer=self._rewards.data).copy()

    def read_dones(self):
        return np.ndarray(self.num_envs, dtype=np.bool_, buffer=self._dones.data).copy()

    def read_timeouts(self):
        return np.ndarray(self.num_envs, dtype=np.bool_, buffer=self._timeouts.data).copy()

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
        }
        for name, stride in layouts.items():
            shm = SharedMemory(name=names[name])
            obj._shm[name] = shm
        return obj


def _worker_loop(worker_id, start_env, num_envs, shm_names, barrier, model_xml, pkl_dir):
    """Worker process entry point."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # let parent handle Ctrl-C

    # Attach shared memory
    shm = EnvSharedMemory.attach(shm_names)

    # Rebuild numpy views for this worker
    obs_buf = np.ndarray((num_envs, OBS_DIM), dtype=np.float32, buffer=shm._shm["obs"].buf,
                          offset=start_env * OBS_BYTES)
    terminal_buf = np.ndarray((num_envs, OBS_DIM), dtype=np.float32, buffer=shm._shm["terminal"].buf,
                               offset=start_env * OBS_BYTES)
    act_buf = np.ndarray((num_envs, ACT_DIM), dtype=np.float32, buffer=shm._shm["actions"].buf,
                          offset=start_env * ACT_BYTES)
    rew_buf = np.ndarray(num_envs, dtype=np.float32, buffer=shm._shm["rewards"].buf)
    done_buf = np.ndarray(num_envs, dtype=np.uint8, buffer=shm._shm["dones"].buf)
    to_buf = np.ndarray(num_envs, dtype=np.uint8, buffer=shm._shm["timeouts"].buf)

    # Create local envs
    envs = [MuJoCoEnv(model_xml, pkl_dir) for _ in range(num_envs)]

    # Initial reset
    for i, env in enumerate(envs):
        obs = env.reset()
        flat = np.concatenate([obs["actor_obs"], obs["critic_obs"], obs["tokenizer"]])
        obs_buf[i] = flat

    while True:
        # Wait for trainer to write actions
        barrier.wait()

        # Read actions
        actions = act_buf.copy()

        # Step each env
        for i, env in enumerate(envs):
            obs, reward, done, info = env.step(actions[i])
            flat = np.concatenate([obs["actor_obs"], obs["critic_obs"], obs["tokenizer"]])
            obs_buf[i] = flat
            rew_buf[i] = reward
            done_buf[i] = int(done)
            to_buf[i] = int(info.get("time_outs", False))
            if info.get("terminal_obs") is not None:
                tflat = np.concatenate([
                    info["terminal_obs"]["actor_obs"],
                    info["terminal_obs"]["critic_obs"],
                    info["terminal_obs"]["tokenizer"],
                ])
                terminal_buf[i] = tflat

        # Notify trainer
        barrier.wait()


class MuJoCoEnvManager:
    """Parallel MuJoCo environment manager."""

    BARRIER_TIMEOUT = 60
    TOTAL_OBS_DIM = OBS_DIM

    def __init__(self, num_envs, num_workers, model_xml, pkl_dir):
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.model_xml = model_xml
        self.pkl_dir = pkl_dir

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

        # Spawn workers (before DDP/HCCL init!)
        self._workers = []
        for worker_id, start, n in worker_args:
            p = mp.Process(
                target=_worker_loop,
                args=(worker_id, start, n, self._shm.names, self._barrier, model_xml, pkl_dir),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        logger.info(f"MuJoCoEnvManager: {num_envs} envs × {self._actual_workers} workers")

    def step(self, actions):
        """Trainer-side: distribute actions, wait, return results."""
        # Write actions
        act_buf = np.ndarray((self.num_envs, ACT_DIM), dtype=np.float32,
                             buffer=self._shm._actions.data)
        act_buf[:] = actions

        # Sync 1: trigger workers
        self._barrier.wait()

        # Sync 2: wait for workers to finish
        self._barrier.wait()

        # Read results
        obs = self._shm.read_obs()
        rewards = self._shm.read_rewards()
        dones = self._shm.read_dones()
        timeouts = self._shm.read_timeouts()
        terminal_obs = self._shm.read_terminal()

        return obs, rewards, dones, {"time_outs": timeouts, "terminal_obs": terminal_obs}

    def close(self):
        for p in self._workers:
            p.terminate()
            p.join(timeout=5)
        self._shm.close()
