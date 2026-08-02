"""Quick visual check that game state is being tracked correctly.

Renders tile ownership as it comes in over the bridge - land tiles colored
by owner, water tiles a fixed color, unowned land another fixed color.

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

import cv2
import numpy as np

from raster import MaskedRaster

PLAYER_ID_MASK = 0xFFF
IS_LAND_BIT = 23  # terrainByte's IS_LAND_BIT (7) shifted by 16

WATER = (0.55, 0.75, 0.95)
UNOWNED_LAND = (0.85, 0.83, 0.75)
AGENT_COLOR = (1.0, 0.05, 0.05)  # bright red - not part of tab20, stands out


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


def render(
    owner_grid: np.ndarray,
    land_mask: np.ndarray,
    agent_small_id: int | None = None,
) -> MaskedRaster:
    h, w = owner_grid.shape
    owned = land_mask & (owner_grid != 0)
    canvas = (
        MaskedRaster(h, w, background=WATER)
        .fill(land_mask & (owner_grid == 0), UNOWNED_LAND)
        .fill_by_colormap(owned, owner_grid)
    )
    if agent_small_id is not None:
        canvas.fill(owner_grid == agent_small_id, AGENT_COLOR)
    return canvas


def outline_agent_territory(
    img: np.ndarray,
    owner_grid: np.ndarray,
    agent_small_id: int | None,
    scale: int = 1,
    color: tuple[int, int, int] = (0, 255, 255),  # bright yellow, BGR
    thickness: int = 2,
) -> np.ndarray:
    if agent_small_id is None:
        return img
    mask = (owner_grid == agent_small_id).astype(np.uint8) * 255
    if scale != 1:
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def apply_unit_updates(units: dict, unit_updates: list | None) -> None:
    if not unit_updates:
        return
    for u in unit_updates:
        if not u["isActive"] or u["markedForDeletion"] is not False:
            units.pop(u["id"], None)
        else:
            units[u["id"]] = u


UNIT_LABELS = {
    "City": "C",
    "Factory": "F",
    "Port": "P",
    "Missile Silo": "MS",
    "Defense Post": "D",
    "SAM Launcher": "S",
    "Train": "T",
}


def draw_units(img: np.ndarray, units: dict, width: int, scale: int = 1) -> np.ndarray:
    # Fixed black-on-white marker, not owner-colored - an owner-colored marker
    # can blend into same-colored territory underneath it and disappear.
    radius = max(4, scale)
    for u in units.values():
        x, y = (u["pos"] % width) * scale, (u["pos"] // width) * scale
        cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
        cv2.circle(img, (x, y), radius, (0, 0, 0), 2)
        label = UNIT_LABELS.get(u["unitType"], "?")
        cv2.putText(
            img, label, (x - radius, y - radius - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return img
