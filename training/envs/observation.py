import numpy as np
import torch

MAX_SMALL_ID = 4096  # PLAYER_ID_MASK + 1 (game_state.py) - owner smallID is a 12-bit field


def build_ratio_table(player_stats: dict) -> np.ndarray:
    """troops/maxTroops per owner smallID, dense over the full smallID space - 0 for
    unowned/absent ids. This is the compact form that actually gets persisted (see
    EpisodeState.ratio_table/train.py's rollout buffer): owner_grid + this table is a
    lossless dictionary encoding of the same information build_troop_ratio_plane's
    (H, W) gather materializes, at a tiny fraction of the storage."""
    ratio_by_id = np.zeros(MAX_SMALL_ID, dtype=np.float32)
    for sid, stats in player_stats.items():
        max_troops = stats.get("maxTroops", 0.0)
        if max_troops > 0:
            ratio_by_id[sid] = stats.get("troops", 0.0) / max_troops
    return ratio_by_id


def build_troop_ratio_plane(ratio_table: np.ndarray, owner_grid: np.ndarray) -> np.ndarray:
    """troops/maxTroops of whoever owns each tile, scattered onto the grid via a
    lookup-array gather - one vectorized op, not a per-tile loop. 0 on unowned tiles
    (ratio_table[0] is never set) - ambiguous with "owned by someone at 0 troops" in
    isolation, but this only ever gets interpreted alongside is_mine/is_enemy."""
    return ratio_table[owner_grid]


def build_troop_advantage_plane(agent_small_id: int, ratio_table: np.ndarray, troop_ratio_plane: np.ndarray) -> np.ndarray:
    """agent's troop_ratio minus whoever owns each tile's troop_ratio - bounded to
    roughly [-1, 1] and centered at 0 for parity, unlike a raw troop-count ratio
    (unbounded, and reintroduces the "raw count grows with game time" problem
    troop_ratio itself was built to avoid - see encode_observation's troop_ratio
    comment below). Derived from ratio_table instead of re-scanning player_stats, so
    the two planes can't drift out of sync with each other."""
    agent_ratio = ratio_table[agent_small_id]
    return agent_ratio - troop_ratio_plane


def reconstruct_spatial_torch(
    bool_planes: torch.Tensor, owner_grid: torch.Tensor, ratio_table: torch.Tensor, agent_ratio: torch.Tensor,
    static_planes: torch.Tensor,
) -> torch.Tensor:
    """Rebuilds the (B, 6, H, W) float spatial tensor the network trains against from
    the compressed rollout-buffer representation - bool_planes (B, 2, H, W) bool
    [is_mine, is_enemy], static_planes (2, H, W) bool [is_land, is_impassable] shared
    by the whole batch (assumes land/impassable never change mid-episode and the map
    is constant - not true with water nukes), owner_grid (B, H, W) long, ratio_table (B, MAX_SMALL_ID) float, agent_ratio (B,)
    float (the agent's own troop_ratio, scalar index 1 - see encode_observation).
    Exact, not an approximation: owner_grid -> ratio_table is the same dictionary
    lookup build_troop_ratio_plane does at collection time, just deferred to just
    before the forward pass instead of being materialized as an (H, W) plane for
    every one of STEPS ticks. Only used for buffered (PPO/SIL) minibatches - the live
    per-tick action decision still uses encode_observation's fully materialized
    spatial directly, since that's never accumulated across STEPS."""
    b, h, w = owner_grid.shape
    troop_ratio_plane = torch.gather(ratio_table, 1, owner_grid.reshape(b, h * w)).reshape(b, h, w)
    troop_advantage_plane = agent_ratio.view(b, 1, 1) - troop_ratio_plane
    static = static_planes.float().unsqueeze(0).expand(b, -1, -1, -1)
    return torch.cat([
        static, bool_planes.float(), troop_ratio_plane.unsqueeze(1), troop_advantage_plane.unsqueeze(1),
    ], dim=1)


def encode_observation(
    land_mask: np.ndarray, impassable_mask: np.ndarray, is_mine: np.ndarray, is_enemy: np.ndarray,
    player_stats: dict, agent_small_id: int, total_land: float, owner_grid: np.ndarray,
    outgoing_troops: float = 0.0, incoming_troops: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Turn accumulated game state into (spatial, scalar) arrays for the network.

    Args:
        land_mask: bool, shape (H, W). Axis order (row, col) = (y, x). True
            where the tile is land - built by game_state.apply_tile_updates.
        impassable_mask: bool, shape (H, W), same axis order. True where the
            tile is permanently unownable terrain (mountains/walls) - unlike
            land_mask this never changes mid-episode, see
            game_state.apply_impassable_bitmap. Without this the CNN can't
            tell a real neutral-land expansion target from a dead zone that
            will always look unclaimed but can never be attacked into.
        is_mine: bool, shape (H, W), same axis order. True where the agent
            owns the tile - maintained incrementally by apply_tile_updates.
        is_enemy: bool, shape (H, W), same axis order. True where an enemy
            owns the tile - see is_mine.
        player_stats: {smallID: {"tilesOwned": float, "gold": float,
            "troops": float}}, built by game_state.apply_player_updates.
        agent_small_id: the agent's own smallID (result["agentSmallID"]).
        total_land: land_mask.sum(), passed in to avoid recomputing it per call.
        owner_grid: uint32, shape (H, W), same axis order. Per-tile owner smallID
            (0 = unowned) - used to scatter per-player troop_ratio onto the grid
            (see build_troop_ratio_plane/build_troop_advantage_plane). Unlike
            is_mine/is_enemy this can't be maintained incrementally: troop_ratio
            drifts every tick for every player regardless of whether any tile
            changed hands, so it's rebuilt fresh from the full grid every call.
        outgoing_troops: troops the agent currently has committed to its own
            active attacks (harness.ts's StepResult.outgoingTroops) - 0.0 at
            episode start, before any attack has been issued.
        incoming_troops: total enemy troops currently attacking the agent
            (StepResult.incomingTroops) - 0.0 at episode start.

    Returns:
        spatial: float32, shape (6, H, W) - channels-first (PyTorch Conv2d
            convention), planes are [is_land, is_impassable, is_mine, is_enemy,
            troop_ratio, troop_advantage].
        scalar: float32, shape (5,) - [gold, troop_ratio, tilesOwned,
            outgoing_ratio, incoming_ratio] for the agent, normalized.
        ratio_table: float32, shape (MAX_SMALL_ID,) - see build_ratio_table. Callers
            store this on EpisodeState so train.py's rollout buffer can persist the
            compact (owner_grid, ratio_table) form instead of the materialized planes.
    """

    ratio_table = build_ratio_table(player_stats)
    troop_ratio_plane = build_troop_ratio_plane(ratio_table, owner_grid)
    troop_advantage_plane = build_troop_advantage_plane(agent_small_id, ratio_table, troop_ratio_plane)

    spatial = np.stack([
        land_mask, impassable_mask, is_mine, is_enemy, troop_ratio_plane, troop_advantage_plane,
    ], axis=0, dtype=np.float32)

    agent_stats = player_stats.get(agent_small_id, {})
    tiles_fraction = agent_stats.get("tilesOwned", 0.0) / max(total_land, 1)
    max_troops = max(agent_stats.get("maxTroops", 1.0), 1)
    # ratio of current to max troops, not raw count - raw troops grows with elapsed
    # game time/territory regardless of strategy, which a value function can latch
    # onto as a shortcut for "how far into the game are we" instead of learning
    # anything state-dependent. Ratio is what the engine's own scripted AI gates on
    # (AiAttackBehavior.ts's hasReserveRatioTroops), not raw troop count.
    troop_ratio = agent_stats.get("troops", 0.0) / max_troops
    # same ratio treatment for troop commitment the agent otherwise has no way to
    # perceive: how much of its own army is already locked into an attack, and how
    # much enemy troop count is currently attacking it.
    outgoing_ratio = outgoing_troops / max_troops
    incoming_ratio = incoming_troops / max_troops
    scalar = np.array([
        np.log1p(agent_stats.get("gold", 0.0)),
        troop_ratio,
        tiles_fraction,
        outgoing_ratio,
        incoming_ratio,
    ], dtype=np.float32)

    return spatial, scalar, ratio_table
