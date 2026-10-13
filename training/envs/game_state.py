"""Decodes raw bridge updates into accumulated game state.

No display concerns here (no cv2) - this is shared by both visualization
(map_view.py) and the observation encoder, so it can't depend on either.

Bit layout below is taken directly from the packing logic in
engine/OpenFrontIO/src/core/game/GameImpl.ts (recordTileUpdate) and
GameMap.ts (GameMapImpl's PLAYER_ID_MASK/IS_LAND_BIT etc):

  packed = (tileState & 0xffff) | (terrainByte << 16)

  tileState (low 16 bits):
    bits 0-11  owner smallID (0 = unowned)
    bit 13     fallout
    bit 14     defense bonus

  terrainByte (bits 16-23 of packed, so absolute bit 16+N):
    bit 23 (terrainByte bit 7)  is land
"""

from dataclasses import dataclass

import numpy as np

from training.envs.observation import encode_observation
from training.envs.action_space import build_entities

PLAYER_ID_MASK = 0xFFF
IS_LAND_BIT = 23  # terrainByte's IS_LAND_BIT (7) shifted by 16


@dataclass
class EpisodeState:
    """One in-flight episode's accumulated state - shared shape between train.py's
    vectorized rollout collection and watch.py's single-game playback, so the two
    can't silently diverge on what an "episode" carries (see apply_step/apply_reset)."""
    width: int
    height: int
    owner_grid: np.ndarray
    land_mask: np.ndarray
    impassable_mask: np.ndarray
    is_mine: np.ndarray
    is_enemy: np.ndarray
    ratio_table: np.ndarray
    player_stats: dict
    agent_client_id: str
    agent_small_id: int
    agent_name: str
    player_types: dict
    spatial: np.ndarray
    scalar: np.ndarray
    entities: np.ndarray
    entity_mask: np.ndarray
    targets: list
    ticks: int = 0


def apply_tile_updates(
    player_stats: dict,
    owner_grid: np.ndarray,
    land_mask: np.ndarray,
    width: int,
    tile_updates: np.ndarray | None,
    agent_small_id: int,
    is_mine: np.ndarray,
    is_enemy: np.ndarray,
    troop_ratio_plane: np.ndarray | None = None,
    troop_advantage_plane: np.ndarray | None = None
) -> None:
    """Updates is_mine/is_enemy in place at the changed tiles only."""
    if tile_updates is None or len(tile_updates) == 0:
        return
    refs = tile_updates[0::2]
    states = tile_updates[1::2]
    xs = refs % width
    ys = refs // width
    owners = states & PLAYER_ID_MASK
    is_land = (states >> IS_LAND_BIT) & 1 == 1
    owner_grid[ys, xs] = owners
    land_mask[ys, xs] = is_land

    is_mine[ys, xs] = is_land & (owners == agent_small_id)
    is_enemy[ys, xs] = is_land & (owners != 0) & (owners != agent_small_id)





def apply_land_bitmap(land_mask: np.ndarray, width: int, bitmap: np.ndarray) -> None:
    """One-time full terrain dump at reset (server.ts's landBitmap) - unlike
    tile_updates deltas, which only ever cover a tile once its owner/water/fallout
    state changes at least once, this covers every tile so land_mask starts
    complete instead of growing in from near-empty over the course of an episode."""
    land_mask[:] = bitmap.reshape(land_mask.shape).astype(bool)


def apply_impassable_bitmap(impassable_mask: np.ndarray, width: int, bitmap: np.ndarray) -> None:
    """One-time full dump at reset (server.ts's impassableBitmap) - unlike land_mask,
    impassability can never change mid-episode (GameMap.ts's setWater() refuses to
    touch impassable tiles), so this never needs tile_updates upkeep."""
    impassable_mask[:] = bitmap.reshape(impassable_mask.shape).astype(bool)


def apply_unit_updates(units: dict, unit_updates: list | None) -> None:
    if not unit_updates:
        return
    for u in unit_updates:
        if not u["isActive"] or u["markedForDeletion"] is not False:
            units.pop(u["id"], None)
        else:
            units[u["id"]] = u


def apply_player_updates(player_stats: dict, player_updates) -> None:
    if player_updates is None:
        return
    # quint since upstream added goldEarned (PlayerImpl.toUpdate) - unused here.
    for i in range(0, len(player_updates), 5):
        small_id, tiles, gold, troops, _gold_earned = player_updates[i:i + 5]
        player_stats[int(small_id)] = {"tilesOwned": tiles, "gold": gold, "troops": troops}


def apply_max_troops(player_stats: dict, max_troops: dict) -> None:
    """maxTroops arrives as a plain JSON side-channel (server.ts), not the packed
    delta stream - full every tick, for every player, unlike packedPlayerUpdates
    which only carries players whose stats changed. Merged in rather than replacing,
    since a player's tilesOwned/gold/troops entry may not have arrived yet."""
    for sid, cap in max_troops.items():
        player_stats.setdefault(int(sid), {})["maxTroops"] = cap


def new_masks(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fresh (owner_grid, land_mask, impassable_mask, is_mine, is_enemy) for one episode."""
    return (
        np.zeros((height, width), dtype=np.uint32),
        np.zeros((height, width), dtype=bool),
        np.zeros((height, width), dtype=bool),
        np.zeros((height, width), dtype=bool),
        np.zeros((height, width), dtype=bool),
    )


def apply_reset(
    owner_grid: np.ndarray, land_mask: np.ndarray, impassable_mask: np.ndarray,
    is_mine: np.ndarray, is_enemy: np.ndarray, width: int, setup: dict,
) -> tuple[dict, int, dict]:
    """Applies bridge.reset()'s full-dump payload to the given masks in place.
    Returns (player_stats, agent_small_id, player_types) - callers still own total_land
    (setup["totalLand"]) and any gold-stats handling, since train.py/watch.py disagree on
    whether the running gold stats get updated here or stay frozen (see apply_step)."""
    player_stats: dict = {}
    agent_small_id = setup["agentSmallID"]
    apply_tile_updates(
        player_stats, owner_grid, land_mask, width, setup["packedTileUpdates"], agent_small_id, is_mine, is_enemy,
    )
    apply_land_bitmap(land_mask, width, setup["landBitmap"])
    apply_impassable_bitmap(impassable_mask, width, setup["impassableBitmap"])
    apply_player_updates(player_stats, setup["packedPlayerUpdates"])
    apply_max_troops(player_stats, setup["maxTroops"])
    player_types = {int(sid): t for sid, t in setup["playerTypes"].items()}
    return player_stats, agent_small_id, player_types


def apply_step(episode: EpisodeState, step_result: dict) -> tuple[np.ndarray, dict, dict, bool, bool]:
    """Applies one tick's bridge.step() result to episode's masks/player_stats/spatial/
    scalar/targets in place. episode.entities/entity_mask are NOT touched - gold
    normalization has to run on the entity_features returned here before pad_entities(),
    and train.py updates its running gold stats from every tick's real entities while
    watch.py normalizes with a frozen checkpoint snapshot instead (see CLAUDE.md's gold-
    normalization note) - folding that in here would force one script's semantics onto
    the other. Callers apply their own gold treatment + pad_entities(), then assign
    episode.entities/entity_mask themselves.

    Returns (entity_features, prev_stats, curr_stats, done, won). prev_stats is a
    snapshot from before this tick's updates were applied, for reward/diagnostic deltas.
    """
    prev_stats = episode.player_stats.get(episode.agent_small_id, {})

    apply_player_updates(episode.player_stats, step_result["packedPlayerUpdates"])
    apply_max_troops(episode.player_stats, step_result["maxTroops"])
    apply_tile_updates(
        episode.player_stats, episode.owner_grid, episode.land_mask, episode.width,
        step_result["packedTileUpdates"], episode.agent_small_id, episode.is_mine, episode.is_enemy,
    )

    curr_stats = episode.player_stats.get(episode.agent_small_id, {})
    # elimination is its own loss - the wider FFA runs long after we're wiped out
    eliminated = curr_stats.get("tilesOwned", 0.0) == 0.0
    done = step_result["done"] or eliminated
    won = (not eliminated) and step_result["winner"] == episode.agent_name

    total_land = step_result["totalLand"]
    episode.spatial, episode.scalar, episode.ratio_table = encode_observation(
        episode.land_mask, episode.impassable_mask, episode.is_mine, episode.is_enemy,
        episode.player_stats, episode.agent_small_id, total_land, episode.owner_grid,
        step_result["outgoingTroops"], step_result["incomingTroops"],
    )
    entity_features, episode.targets = build_entities(
        step_result["attackMask"], episode.player_stats, episode.player_types, total_land
    )

    return entity_features, prev_stats, curr_stats, done, won
