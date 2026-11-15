"""Vectorised env where each worker process owns its episode and does all per-step work.

The parent only batches observations (shared memory), runs the policy and sends back action
floats. See collect_rollout_workers in train.py."""

import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory

import numpy as np

from training.envs.action_space import DO_NOTHING, action_to_intents, diplomacy_from_code, pad_entities
from training.envs.bridge import OpenFrontBridge, _ensure_built
from training.envs.game_state import apply_step, episode_from_setup
from training.envs.reward import compute_reward

ACTION_STRIDE = 64  # policy.ACTION_STRIDE, duplicated to keep torch out of the workers
OBS_FIELDS = ("spatial", "scalar", "entities", "entity_mask", "owner_grid", "ratio_table")


@dataclass
class EnvConfig:
    map: str
    bots: int
    nations: int
    difficulties: list[str]
    max_episode_ticks: int
    act_every: int
    gamma_step: float


def decode_action_np(action) -> tuple[int, float, int]:
    """numpy twin of policy.decode_action for one action."""
    whole = np.floor(action).astype(np.int64)
    return int(whole % ACTION_STRIDE), float(np.float32(action - whole.astype(action.dtype))), int(whole // ACTION_STRIDE) - 1


class _EnvWorker:
    def __init__(self, idx: int, cfg: EnvConfig):
        self.idx = idx
        self.cfg = cfg
        self.bridge = OpenFrontBridge(map=cfg.map, bots=cfg.bots, nations=cfg.nations, build=False)
        self.episode = None
        self.slabs = {}
        self.views = {}

    def start_episode(self, seed: str | None = None, difficulty: str | None = None) -> None:
        difficulty = difficulty or random.choice(self.cfg.difficulties)
        seed = seed or str(random.randint(0, 2 ** 31 - 1))
        setup = self.bridge.reset(seed=seed, map=self.cfg.map, bots=self.cfg.bots, nations=self.cfg.nations, difficulty=difficulty)
        self.episode = episode_from_setup(setup)
        self.episode.difficulty = difficulty
        self.episode.seed = seed
        self.num_nations = sum(1 for t in self.episode.player_types.values() if t == "NATION")

    def observation(self) -> dict:
        ep = self.episode
        return {
            "spatial": ep.spatial, "scalar": ep.scalar, "entities": ep.entities, "entity_mask": ep.entity_mask,
            "owner_grid": ep.owner_grid, "ratio_table": ep.ratio_table,
            "agent_id": np.asarray(ep.agent_small_id, dtype=np.int16),
        }

    def attach(self, names: dict, layout: dict) -> None:
        for field, name in names.items():
            shm = shared_memory.SharedMemory(name=name)
            # the parent owns the block; stop this process's tracker from unlinking it at exit
            resource_tracker.unregister(shm._name, "shared_memory")
            shape, dtype = layout[field]
            self.slabs[field] = shm
            self.views[field] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self.write_observation()

    def write_observation(self) -> None:
        for field, value in self.observation().items():
            self.views[field][self.idx] = value

    def step(self, action) -> dict:
        cfg, ep = self.cfg, self.episode
        tgt_idx, size_frac, diplomacy_code = decode_action_np(action)
        intents = action_to_intents(
            target=ep.targets[tgt_idx] if tgt_idx < len(ep.targets) else DO_NOTHING,
            agent_client_id=ep.agent_client_id,
            troops=size_frac * ep.player_stats.get(ep.agent_small_id, {}).get("troops", 0.0),
            diplomacy=diplomacy_from_code(diplomacy_code, ep.targets),
        )
        step_result = self.bridge.step(intents)

        entity_features, prev_stats, curr_stats, done, won = apply_step(ep, step_result)
        ep.ticks += cfg.act_every
        ep.peak = max(ep.peak, float(ep.scalar[1]))
        capped = False
        if not done and ep.ticks >= cfg.max_episode_ticks:
            others = [v.get("tilesOwned", 0.0) for sid, v in ep.player_stats.items() if sid != ep.agent_small_id]
            won = curr_stats.get("tilesOwned", 0.0) > max(others, default=0.0)
            done, capped = True, True
        conquests = step_result.get("conquests", [])
        kills = sum(
            c["conqueror"] == ep.agent_small_id and c["type"] == "NATION"
            for c in conquests[ep.conquests_seen:]
        )
        ep.conquests_seen = len(conquests)
        reward = compute_reward(prev_stats, curr_stats, done, won, step_result["totalLand"], cfg.gamma_step, kills)
        ep.entities, ep.entity_mask = pad_entities(entity_features)

        info = {"reward": reward, "done": done, "won": won, "ticks": ep.ticks}
        if done:
            info.update(
                kind="cap" if capped else "win" if won else "other" if step_result.get("winner") else "elim",
                end_ticks=ep.ticks, peak=ep.peak, setup=f"{self.num_nations}{ep.difficulty[:1]}", capped=capped,
            )
            self.start_episode()
            info["ticks"] = self.episode.ticks
        self.write_observation()
        return info

    def close(self) -> None:
        for shm in self.slabs.values():
            shm.close()
        self.bridge.process.terminate()
        self.bridge.process.wait()


def _worker(conn, idx: int, cfg: EnvConfig):
    random.seed(int.from_bytes(os.urandom(8), "little"))  # forked workers would share the parent's RNG state
    worker = _EnvWorker(idx, cfg)
    while True:
        cmd, payload = conn.recv()
        if cmd == "close":
            worker.close()
            conn.close()
            return
        try:
            if cmd == "reset":
                worker.start_episode(**payload)
                obs = worker.observation()
                res = {
                    "layout": {k: (v.shape, v.dtype.str) for k, v in obs.items()},
                    "static_planes": np.stack([worker.episode.land_mask, worker.episode.impassable_mask]),
                }
            elif cmd == "attach":
                worker.attach(*payload)
                res = {}
            else:
                res = worker.step(payload)
        except Exception as e:
            import traceback
            res = {"cmd_success": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
        conn.send(res)


def _check(res: dict, idx: int, cmd: str) -> dict:
    if not res.get("cmd_success", True):
        raise RuntimeError(f"env {idx} error on {cmd}: {res.get('error')}")
    return res


class OwnedVecEnv:
    """Batched observation arrays (E, ...) live in shared memory; workers write their own row."""

    def __init__(self, num_envs: int, cfg: EnvConfig):
        _ensure_built()
        self.num_envs = num_envs
        self.conns, self.procs = [], []
        for i in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_worker, args=(child_conn, i, cfg), daemon=True)
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)
        self.shms: list[shared_memory.SharedMemory] = []
        self.obs: dict[str, np.ndarray] = {}
        self.static_planes = None

    def _call_all(self, cmd: str, payloads: list) -> list[dict]:
        for conn, payload in zip(self.conns, payloads):
            conn.send((cmd, payload))
        return [_check(conn.recv(), i, cmd) for i, conn in enumerate(self.conns)]

    def start(self, seeds: list[str] | None = None, difficulties: list[str] | None = None) -> None:
        payloads = [
            {"seed": seeds[i] if seeds else None, "difficulty": difficulties[i] if difficulties else None}
            for i in range(self.num_envs)
        ]
        metas = self._call_all("reset", payloads)
        layout = {k: ((self.num_envs, *shape), np.dtype(dt)) for k, (shape, dt) in metas[0]["layout"].items()}
        names = {}
        for field, (shape, dtype) in layout.items():
            shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * dtype.itemsize)
            self.shms.append(shm)
            names[field] = shm.name
            self.obs[field] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self.static_planes = metas[0]["static_planes"]
        self._call_all("attach", [(names, layout)] * self.num_envs)

    def step(self, actions: np.ndarray) -> list[dict]:
        return self._call_all("step", list(actions))

    def close(self) -> None:
        for conn in self.conns:
            conn.send(("close", None))
        for p in self.procs:
            p.join()
        self.obs = {}
        for shm in self.shms:
            shm.close()
            shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
