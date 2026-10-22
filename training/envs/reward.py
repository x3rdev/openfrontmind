import os

WIN_REWARD = 150.0
LOSS_REWARD = -150.0
TILE_SCALE = 100.0
TIME_COST_PER_TICK = float(os.environ.get("OFM_TIME_COST", "0.01"))

# Potential-based troop shaping: the potential is the share of peak troop production
# forfeited, so spending troops costs the production it gives up.
PROD_COEF = 20.0
PROD_PEAK_RATIO = 0.73 / 1.73  # troops/max where Config.troopIncreaseRate peaks


def _regen(troops: float, max_troops: float) -> float:
    """Config.troopIncreaseRate."""
    return (10.0 + max(troops, 0.0) ** 0.73 / 4.0) * max(1.0 - troops / max_troops, 0.0)


def _prod_potential(stats: dict) -> float:
    max_troops = max(stats.get("maxTroops", 1.0), 1.0)
    peak = _regen(PROD_PEAK_RATIO * max_troops, max_troops)
    return -PROD_COEF * (1.0 - _regen(stats.get("troops", 0.0), max_troops) / peak)


def compute_reward(prev_stats: dict, curr_stats: dict, done: bool, won: bool, total_land: float, gamma: float) -> float:
    if done:
        return WIN_REWARD if won else LOSS_REWARD
    tile_gain = (curr_stats.get("tilesOwned", 0.0) - prev_stats.get("tilesOwned", 0.0)) / max(total_land, 1)
    shaping = gamma * _prod_potential(curr_stats) - _prod_potential(prev_stats)
    return TILE_SCALE * tile_gain + shaping - TIME_COST_PER_TICK
