import numpy as np
import torch

MAX_SMALL_ID = 4096  # owner smallID is a 12-bit field


def build_ratio_table(player_stats: dict) -> np.ndarray:
    """troops/maxTroops per owner smallID (0 for unowned or absent ids).
    Together with owner_grid this is a compact encoding of the troop-ratio plane."""
    ratio_by_id = np.zeros(MAX_SMALL_ID, dtype=np.float32)
    for sid, stats in player_stats.items():
        max_troops = stats.get("maxTroops", 0.0)
        if max_troops > 0:
            ratio_by_id[sid] = stats.get("troops", 0.0) / max_troops
    return ratio_by_id


def build_troop_ratio_plane(ratio_table: np.ndarray, owner_grid: np.ndarray) -> np.ndarray:
    """Each tile's owner's troop ratio (0 on unowned tiles)."""
    return ratio_table[owner_grid]


def build_troop_advantage_plane(agent_small_id: int, ratio_table: np.ndarray, troop_ratio_plane: np.ndarray) -> np.ndarray:
    """The agent's troop ratio minus each tile owner's, roughly in [-1, 1]."""
    return ratio_table[agent_small_id] - troop_ratio_plane


def reconstruct_spatial_torch(
    bool_planes: torch.Tensor, owner_grid: torch.Tensor, ratio_table: torch.Tensor, agent_ratio: torch.Tensor,
    static_planes: torch.Tensor,
) -> torch.Tensor:
    """Rebuilds the (B, 6, H, W) spatial tensor from the compressed rollout buffers.

    bool_planes (B, 2, H, W) is [is_mine, is_enemy]; static_planes (2, H, W) is
    [is_land, is_impassable], shared by the batch (the map never changes mid-episode);
    owner_grid (B, H, W) long; ratio_table (B, MAX_SMALL_ID); agent_ratio (B,)."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the network inputs from the accumulated game state.

    Returns (spatial, scalar, ratio_table):
      spatial: (6, H, W) planes [is_land, is_impassable, is_mine, is_enemy, troop_ratio, troop_advantage]
      scalar: (5,) agent features [log gold, troop_ratio, tile fraction, outgoing_ratio, incoming_ratio]
      ratio_table: see build_ratio_table
    outgoing/incoming_troops are the agent's committed attack troops and the enemy troops attacking it.
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
    # a ratio rather than a raw count, since raw troops just track elapsed time
    troop_ratio = agent_stats.get("troops", 0.0) / max_troops
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
