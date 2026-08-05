"""Renders accumulated game state (game_state.py) for a quick visual check."""

import cv2
import numpy as np

from training.envs.raster import MaskedRaster

WATER = (0.55, 0.75, 0.95)
UNOWNED_LAND = (0.85, 0.83, 0.75)
AGENT_COLOR = (1.0, 0.05, 0.05)  # bright red - not part of tab20, stands out
BOT_COLOR = (0.5, 0.5, 0.5)  # equal RGB - flat gray, no colormap tint


def render(
    owner_grid: np.ndarray,
    land_mask: np.ndarray,
    agent_small_id: int | None = None,
    player_types: dict[int, str] | None = None,
) -> MaskedRaster:
    h, w = owner_grid.shape
    owned = land_mask & (owner_grid != 0)
    canvas = (
        MaskedRaster(h, w, background=WATER)
        .fill(land_mask & (owner_grid == 0), UNOWNED_LAND)
        .fill_by_colormap(owned, owner_grid)
    )
    if player_types:
        bot_ids = [sid for sid, t in player_types.items() if t == "BOT"]
        if bot_ids:
            canvas.fill(owned & np.isin(owner_grid, bot_ids), BOT_COLOR)
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
