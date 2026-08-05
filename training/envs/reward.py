WIN_REWARD = 100.0
LOSS_REWARD = -100.0
SCALE = 100.0


def compute_reward(prev_stats: dict, curr_stats: dict, done: bool, won: bool, total_land: float) -> float:
    if done:
        return WIN_REWARD if won else LOSS_REWARD
    prev_tiles = prev_stats.get("tilesOwned", 0.0)
    curr_tiles = curr_stats.get("tilesOwned", 0.0)
    prev_gold = prev_stats.get("gold", 0.0)
    curr_gold = curr_stats.get("gold", 0.0)

    prev_fraction = prev_tiles / max(total_land, 1)
    curr_fraction = curr_tiles / max(total_land, 1)
    delta_gold = curr_gold-prev_gold

    return SCALE*(curr_fraction-prev_fraction)
