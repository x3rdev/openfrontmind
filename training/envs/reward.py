WIN_REWARD = 10000.0
LOSS_REWARD = -10000.0


def apply_player_updates(player_stats: dict, player_updates) -> None:
    if player_updates is None:
        return
    for i in range(0, len(player_updates), 4):
        small_id, tiles, gold, troops = player_updates[i:i+4]
        player_stats[int(small_id)] = {"tilesOwned": tiles, "gold": gold, "troops": troops}

def compute_reward(prev_stats: dict, curr_stats: dict, done: bool, won: bool) -> float:
    if done:
        return WIN_REWARD if won else LOSS_REWARD
    prev_tiles = prev_stats.get("tilesOwned", 0.0)
    curr_tiles = curr_stats.get("tilesOwned", 0.0)
    prev_gold = prev_stats.get("gold", 0.0)
    curr_gold = curr_stats.get("gold", 0.0)

    delta_tiles = curr_tiles-prev_tiles
    delta_gold = curr_gold-prev_gold

    return 1.0*delta_tiles
