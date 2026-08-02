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
from matplotlib import colormaps

from bridge import OpenFrontBridge

PLAYER_ID_MASK = 0xFFF
IS_LAND_BIT = 23  # terrainByte's IS_LAND_BIT (7) shifted by 16


def apply_tile_updates(owner_grid, land_mask, width, tile_updates):
    if tile_updates is None or len(tile_updates) == 0:
        return
    refs = tile_updates[0::2]
    states = tile_updates[1::2]
    xs = refs % width
    ys = refs // width
    owner_grid[ys, xs] = states & PLAYER_ID_MASK
    land_mask[ys, xs] = (states >> IS_LAND_BIT) & 1 == 1


def render(owner_grid, land_mask):
    h, w = owner_grid.shape
    img = np.empty((h, w, 3))
    img[:] = (0.55, 0.75, 0.95)  # water
    img[land_mask & (owner_grid == 0)] = (0.85, 0.83, 0.75)  # unowned land

    owned = land_mask & (owner_grid != 0)
    cmap = colormaps["tab20"]
    img[owned] = cmap((owner_grid[owned].astype(float) % 20) / 20)[:, :3]
    return img


def to_bgr_uint8(img):
    # cv2 expects BGR channel order and uint8 0-255, not RGB float 0-1.
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main(ticks=2000, redraw_every=10, seed="seedseed"):
    with OpenFrontBridge() as bridge:
        result = bridge.reset(seed=seed)
        width, height = result["width"], result["height"]
        owner_grid = np.zeros((height, width), dtype=np.uint32)
        land_mask = np.zeros((height, width), dtype=bool)
        apply_tile_updates(owner_grid, land_mask, width, result["packedTileUpdates"])

        cv2.imshow("map", to_bgr_uint8(render(owner_grid, land_mask)))
        cv2.waitKey(1)

        try:
            for i in range(ticks):
                result = bridge.step([])
                apply_tile_updates(owner_grid, land_mask, width, result["packedTileUpdates"])
                if i % redraw_every == 0:
                    cv2.imshow("map", to_bgr_uint8(render(owner_grid, land_mask)))
                    # waitKey also pumps the window's event loop - required for
                    # it to actually repaint, not just a "wait for keypress".
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if result["done"]:
                    print("game over, winner:", result["winner"])
                    break
        finally:
            cv2.imshow("map", to_bgr_uint8(render(owner_grid, land_mask)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main(ticks=int(10e100), seed="brad has a big booty")
