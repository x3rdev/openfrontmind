import numpy as np


def encode_observation(owner_grid: np.ndarray, land_mask: np.ndarray, player_stats: dict, agent_small_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn accumulated game state into (spatial, scalar) arrays for the network.

    Args:
        owner_grid: uint32, shape (H, W). Each cell is a player's smallID, or
            0 if unowned. Axis order is (row, col) = (y, x) - built by
            game_state.apply_tile_updates.
        land_mask: bool, shape (H, W), same axis order as owner_grid. True
            where the tile is land.
        player_stats: {smallID: {"tilesOwned": float, "gold": float,
            "troops": float}}, built by game_state.apply_player_updates.
        agent_small_id: the agent's own smallID (result["agentSmallID"]).

    Returns:
        spatial: float32, shape (3, H, W) - channels-first (PyTorch Conv2d
            convention), planes are [is_land, is_mine, is_enemy].
        scalar: float32, shape (3,) - [gold, troops, tilesOwned] for the
            agent, normalized.
    """

    is_land = land_mask
    is_mine = land_mask & (owner_grid == agent_small_id)
    is_enemy = land_mask & ((owner_grid != 0) & (owner_grid != agent_small_id))
    spatial = np.stack([is_land, is_mine, is_enemy], axis=0, dtype=np.float32)

    agent_stats = player_stats.get(agent_small_id, {})
    total_land = land_mask.sum()
    tiles_fraction = agent_stats.get("tilesOwned", 0.0) / max(total_land, 1)
    scalar = np.array([
        np.log1p(agent_stats.get("gold", 0.0)),
        np.log1p(agent_stats.get("troops", 0.0)),
        tiles_fraction,
    ], dtype=np.float32)

    return spatial, scalar