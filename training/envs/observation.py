import numpy as np


def encode_observation(land_mask: np.ndarray, is_mine: np.ndarray, is_enemy: np.ndarray, player_stats: dict, agent_small_id: int, total_land: float) -> tuple[np.ndarray, np.ndarray]:
    """Turn accumulated game state into (spatial, scalar) arrays for the network.

    Args:
        land_mask: bool, shape (H, W). Axis order (row, col) = (y, x). True
            where the tile is land - built by game_state.apply_tile_updates.
        is_mine: bool, shape (H, W), same axis order. True where the agent
            owns the tile - maintained incrementally by apply_tile_updates.
        is_enemy: bool, shape (H, W), same axis order. True where an enemy
            owns the tile - see is_mine.
        player_stats: {smallID: {"tilesOwned": float, "gold": float,
            "troops": float}}, built by game_state.apply_player_updates.
        agent_small_id: the agent's own smallID (result["agentSmallID"]).
        total_land: land_mask.sum(), passed in to avoid recomputing it per call.

    Returns:
        spatial: float32, shape (3, H, W) - channels-first (PyTorch Conv2d
            convention), planes are [is_land, is_mine, is_enemy].
        scalar: float32, shape (3,) - [gold, troop_ratio, tilesOwned] for the
            agent, normalized.
    """

    spatial = np.stack([land_mask, is_mine, is_enemy], axis=0, dtype=np.float32)

    agent_stats = player_stats.get(agent_small_id, {})
    tiles_fraction = agent_stats.get("tilesOwned", 0.0) / max(total_land, 1)
    # ratio of current to max troops, not raw count - raw troops grows with elapsed
    # game time/territory regardless of strategy, which a value function can latch
    # onto as a shortcut for "how far into the game are we" instead of learning
    # anything state-dependent. Ratio is what the engine's own scripted AI gates on
    # (AiAttackBehavior.ts's hasReserveRatioTroops), not raw troop count.
    troop_ratio = agent_stats.get("troops", 0.0) / max(agent_stats.get("maxTroops", 1.0), 1)
    scalar = np.array([
        np.log1p(agent_stats.get("gold", 0.0)),
        troop_ratio,
        tiles_fraction,
    ], dtype=np.float32)

    return spatial, scalar