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
) -> None:
    if tile_updates is None or len(tile_updates) == 0:
        return
    refs = tile_updates[0::2]
    states = tile_updates[1::2]
    xs = refs % width
    ys = refs // width
    owner_grid[ys, xs] = states & PLAYER_ID_MASK
    land_mask[ys, xs] = (states >> IS_LAND_BIT) & 1 == 1


def apply_unit_updates(units: dict, unit_updates: list | None) -> None:
    if not unit_updates:
        return
    for u in unit_updates:
        if not u["isActive"] or u["markedForDeletion"] is not False:
            units.pop(u["id"], None)
        else:
            units[u["id"]] = u
