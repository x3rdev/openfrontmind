"""Ring buffer of single timesteps, used twice in train.py: as an anchor buffer of
high-troop-ratio states that led to good returns (self-imitation), and as a low-ratio
buffer that keeps the value head calibrated on depleted states.

The anchor loss is supervised toward a fixed (state, action, return) triple, weighted by
max(0, return - V(s)) with the weight detached, so old entries need no importance correction.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class AnchorBuffer:
    capacity: int
    bool_planes: list = field(default_factory=list)
    owner_grid: list = field(default_factory=list)
    ratio_table: list = field(default_factory=list)
    scalar: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    entity_mask: list = field(default_factory=list)
    action: list = field(default_factory=list)
    ret: list = field(default_factory=list)
    inserted_at: list = field(default_factory=list)
    _next_slot: int = 0

    def __len__(self) -> int:
        return len(self.action)

    def insert(
        self,
        bool_planes: np.ndarray, owner_grid: np.ndarray, ratio_table: np.ndarray,
        scalar: np.ndarray, entities: np.ndarray, entity_mask: np.ndarray,
        action: float, ret: float, update: int,
    ) -> None:
        """Overwrites the oldest slot once full."""
        bool_planes = bool_planes.copy()
        owner_grid = owner_grid.copy()
        ratio_table = ratio_table.copy()
        scalar = scalar.copy()
        entities = entities.copy()
        entity_mask = entity_mask.copy()
        if len(self) < self.capacity:
            self.bool_planes.append(bool_planes)
            self.owner_grid.append(owner_grid)
            self.ratio_table.append(ratio_table)
            self.scalar.append(scalar)
            self.entities.append(entities)
            self.entity_mask.append(entity_mask)
            self.action.append(action)
            self.ret.append(ret)
            self.inserted_at.append(update)
        else:
            slot = self._next_slot % self.capacity
            self.bool_planes[slot] = bool_planes
            self.owner_grid[slot] = owner_grid
            self.ratio_table[slot] = ratio_table
            self.scalar[slot] = scalar
            self.entities[slot] = entities
            self.entity_mask[slot] = entity_mask
            self.action[slot] = action
            self.ret[slot] = ret
            self.inserted_at[slot] = update
            self._next_slot += 1

    def insert_from_rollout(
        self,
        bool_planes_buf: np.ndarray, owner_grid_buf: np.ndarray, ratio_table_buf: np.ndarray,
        scalar_buf: np.ndarray, entities_buf: np.ndarray, entity_mask_buf: np.ndarray,
        actions: np.ndarray, raw_advantages: np.ndarray, returns: np.ndarray,
        high_ratio_threshold: float, update: int, max_insertions: int,
        high_ratio_ceiling: float = float("inf"),
    ) -> int:
        """Inserts a capped random sample of timesteps with a troop ratio inside
        [high_ratio_threshold, high_ratio_ceiling] and positive raw advantage.
        The ceiling stops the buffer rewarding ever-longer waits. Returns the number inserted."""
        candidates = np.nonzero(
            (scalar_buf[:, 1] >= high_ratio_threshold) & (scalar_buf[:, 1] <= high_ratio_ceiling)
            & (raw_advantages > 0)
        )[0]
        if candidates.size == 0:
            return 0
        if candidates.size > max_insertions:
            candidates = np.array(random.sample(list(candidates), max_insertions))
        for i in candidates:
            self.insert(
                bool_planes_buf[i], owner_grid_buf[i], ratio_table_buf[i],
                scalar_buf[i], entities_buf[i], entity_mask_buf[i],
                actions[i], returns[i], update,
            )
        return int(candidates.size)

    def insert_low_ratio_from_rollout(
        self,
        bool_planes_buf: np.ndarray, owner_grid_buf: np.ndarray, ratio_table_buf: np.ndarray,
        scalar_buf: np.ndarray, entities_buf: np.ndarray, entity_mask_buf: np.ndarray,
        actions: np.ndarray, returns: np.ndarray,
        low_ratio_threshold: float, update: int, max_insertions: int,
    ) -> int:
        """Inserts a capped random sample of timesteps below low_ratio_threshold,
        whatever the outcome. Returns the number inserted."""
        candidates = np.nonzero(scalar_buf[:, 1] < low_ratio_threshold)[0]
        if candidates.size == 0:
            return 0
        if candidates.size > max_insertions:
            candidates = np.array(random.sample(list(candidates), max_insertions))
        for i in candidates:
            self.insert(
                bool_planes_buf[i], owner_grid_buf[i], ratio_table_buf[i],
                scalar_buf[i], entities_buf[i], entity_mask_buf[i],
                actions[i], returns[i], update,
            )
        return int(candidates.size)

    def evict_older_than(self, update: int, max_age_updates: int) -> None:
        """Drops entries older than max_age_updates."""
        keep = [i for i, t in enumerate(self.inserted_at) if update - t <= max_age_updates]
        if len(keep) == len(self.inserted_at):
            return
        self.bool_planes = [self.bool_planes[i] for i in keep]
        self.owner_grid = [self.owner_grid[i] for i in keep]
        self.ratio_table = [self.ratio_table[i] for i in keep]
        self.scalar = [self.scalar[i] for i in keep]
        self.entities = [self.entities[i] for i in keep]
        self.entity_mask = [self.entity_mask[i] for i in keep]
        self.action = [self.action[i] for i in keep]
        self.ret = [self.ret[i] for i in keep]
        self.inserted_at = [self.inserted_at[i] for i in keep]
        self._next_slot = len(keep)

    def sample(self, batch_size: int) -> dict | None:
        """A random minibatch as stacked numpy arrays, or None if the buffer is empty."""
        n = len(self)
        if n == 0:
            return None
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return dict(
            bool_planes=np.stack([self.bool_planes[i] for i in idx]),
            owner_grid=np.stack([self.owner_grid[i] for i in idx]),
            ratio_table=np.stack([self.ratio_table[i] for i in idx]),
            scalar=np.stack([self.scalar[i] for i in idx]),
            entities=np.stack([self.entities[i] for i in idx]),
            entity_mask=np.stack([self.entity_mask[i] for i in idx]),
            action=np.array([self.action[i] for i in idx]),
            ret=np.array([self.ret[i] for i in idx], dtype=np.float32),
        )
