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

import numpy as np

PLAYER_ID_MASK = 0xFFF
IS_LAND_BIT = 23  # terrainByte's IS_LAND_BIT (7) shifted by 16


def apply_tile_updates(
    owner_grid: np.ndarray,
    land_mask: np.ndarray,
    width: int,
    tile_updates: np.ndarray | None,
    agent_small_id: int | None = None,
    is_mine: np.ndarray | None = None,
    is_enemy: np.ndarray | None = None,
) -> None:
    """agent_small_id/is_mine/is_enemy are optional: when given, updates those masks in place at the changed tiles only."""
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

    if agent_small_id is not None:
        is_mine[ys, xs] = is_land & (owners == agent_small_id)
        is_enemy[ys, xs] = is_land & (owners != 0) & (owners != agent_small_id)


def apply_land_bitmap(land_mask: np.ndarray, width: int, bitmap: np.ndarray) -> None:
    """One-time full terrain dump at reset (server.ts's landBitmap) - unlike
    tile_updates deltas, which only ever cover a tile once its owner/water/fallout
    state changes at least once, this covers every tile so land_mask starts
    complete instead of growing in from near-empty over the course of an episode."""
    land_mask[:] = bitmap.reshape(land_mask.shape).astype(bool)


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
    for i in range(0, len(player_updates), 4):
        small_id, tiles, gold, troops = player_updates[i:i + 4]
        player_stats[int(small_id)] = {"tilesOwned": tiles, "gold": gold, "troops": troops}


def apply_max_troops(player_stats: dict, max_troops: dict) -> None:
    """maxTroops arrives as a plain JSON side-channel (server.ts), not the packed
    delta stream - full every tick, for every player, unlike packedPlayerUpdates
    which only carries players whose stats changed. Merged in rather than replacing,
    since a player's tilesOwned/gold/troops entry may not have arrived yet."""
    for sid, cap in max_troops.items():
        player_stats.setdefault(int(sid), {})["maxTroops"] = cap
