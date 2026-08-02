"""Compose (H, W, C) float arrays from named masks, layer by layer.

Channel-count-agnostic - same class works for a 3-channel RGB preview image
and an N-channel stack of CNN input planes (is-mine, is-enemy, terrain, ...).
"""

from collections.abc import Sequence

import cv2
import numpy as np
from matplotlib import colormaps


class MaskedRaster:
    def __init__(
        self,
        height: int,
        width: int,
        channels: int = 3,
        background: float | Sequence[float] = 0.0,
    ) -> None:
        self.image: np.ndarray = np.empty((height, width, channels))
        self.image[:] = background

    def fill(self, mask: np.ndarray, value: float | Sequence[float]) -> "MaskedRaster":
        self.image[mask] = value
        return self

    def fill_by_colormap(
        self,
        mask: np.ndarray,
        values: np.ndarray,
        cmap_name: str = "tab20",
        modulo: int = 20,
    ) -> "MaskedRaster":
        # Display-only: maps values to visually distinct RGB colors. Not
        # meaningful for network input planes - use fill() for those.
        cmap = colormaps[cmap_name]
        self.image[mask] = cmap((values[mask].astype(float) % modulo) / modulo)[:, :3]
        return self

    def to_bgr_uint8(self, scale: int = 1) -> np.ndarray:
        # Display-only: cv2 expects BGR uint8 0-255, not RGB float 0-1.
        bgr = cv2.cvtColor((self.image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if scale != 1:
            # Nearest-neighbor: this is a tile grid, not a photo - keep edges
            # crisp instead of blurring them with the default interpolation.
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return bgr
