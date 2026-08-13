WIN_REWARD = 1000.0
LOSS_REWARD = -1000.0
SCALE = 100.0

# Peak of the engine's troop production formula (Config.ts troopIncreaseRate()).
TROOP_RATIO_TARGET = 0.422
SHAPING_COEF = 0.3  # was 1.0, overshot into troop hoarding


def _troop_potential(ratio: float) -> float:
    return -((ratio - TROOP_RATIO_TARGET) ** 2)


def compute_reward(
    prev_stats: dict, curr_stats: dict, done: bool, won: bool, total_land: float, gamma: float
) -> float:
    if done:
        return WIN_REWARD if won else LOSS_REWARD
    prev_tiles = prev_stats.get("tilesOwned", 0.0)
    curr_tiles = curr_stats.get("tilesOwned", 0.0)

    prev_fraction = prev_tiles / max(total_land, 1)
    curr_fraction = curr_tiles / max(total_land, 1)

    # Potential-based shaping (Ng, Harada & Russell 1999) - telescopes to zero over an episode.
    prev_ratio = prev_stats.get("troops", 0.0) / max(prev_stats.get("maxTroops", 1.0), 1)
    curr_ratio = curr_stats.get("troops", 0.0) / max(curr_stats.get("maxTroops", 1.0), 1)
    shaping = SHAPING_COEF * (gamma * _troop_potential(curr_ratio) - _troop_potential(prev_ratio))

    return SCALE*(curr_fraction-prev_fraction) + shaping
