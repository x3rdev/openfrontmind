"""Tests whether the reward+discounting setup itself makes remaining time a better
predictor of realized return than actual board state. Uses a patience heuristic
swept across several troop_ratio thresholds (not the current trained policy), so
outcome/behavior diversity isn't confounded by the current policy's narrow, mostly-
losing behavior - see CLAUDE.md.
"""
import argparse

import numpy as np

from training.envs.bridge import OpenFrontBridge
from training.envs.game_state import apply_player_updates, apply_max_troops
from training.envs.action_space import DO_NOTHING, action_to_intents
from training.envs.reward import compute_reward
from training.models.train import BOTS, NATIONS, DIFFICULTY, GAMMA
from training.models.patience_heuristic import weakest_nation_target


def run_episode(seed: str, threshold: float, bots: int, nations: int, difficulty: str, max_ticks: int):
    """Returns a list of (troop_ratio, tile_fraction, reward) for every tick."""
    with OpenFrontBridge() as bridge:
        setup = bridge.reset(seed=seed, map="mid_plains", bots=bots, nations=nations, difficulty=difficulty)
        agent_client_id = setup["agentClientID"]
        agent_small_id = setup["agentSmallID"]
        agent_name = setup["agentName"]
        player_types = {int(sid): t for sid, t in setup["playerTypes"].items()}

        player_stats: dict = {}
        apply_player_updates(player_stats, setup["packedPlayerUpdates"])
        apply_max_troops(player_stats, setup["maxTroops"])

        attack_mask: list = []
        trace = []

        for i in range(max_ticks):
            prev_stats = dict(player_stats.get(agent_small_id, {}))
            max_troops = max(prev_stats.get("maxTroops", 1.0), 1)
            troop_ratio = prev_stats.get("troops", 0.0) / max_troops

            target = DO_NOTHING
            if troop_ratio >= threshold:
                target = weakest_nation_target(attack_mask, player_stats, player_types)

            intents = action_to_intents(target=target, agent_client_id=agent_client_id)
            step_result = bridge.step(intents)
            attack_mask = step_result["attackMask"]

            apply_player_updates(player_stats, step_result["packedPlayerUpdates"])
            apply_max_troops(player_stats, step_result["maxTroops"])

            curr_stats = player_stats.get(agent_small_id, {})
            eliminated = curr_stats.get("tilesOwned", 0.0) == 0.0
            done = step_result["done"] or eliminated
            won = (not eliminated) and step_result["winner"] == agent_name
            total_land = step_result["totalLand"]

            reward = compute_reward(prev_stats, curr_stats, done, won, total_land, GAMMA)
            tile_fraction = curr_stats.get("tilesOwned", 0.0) / max(total_land, 1)

            trace.append((troop_ratio, tile_fraction, reward))

            if done:
                return trace, won

        return trace, None


def realized_returns(rewards: list) -> np.ndarray:
    """Monte Carlo return-to-go at every tick, using the real training GAMMA."""
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + GAMMA * running
        returns[t] = running
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.422,0.6,0.8")
    parser.add_argument("--seeds-per-threshold", type=int, default=4)
    parser.add_argument("--bots", type=int, default=BOTS)
    parser.add_argument("--nations", type=int, default=NATIONS)
    parser.add_argument("--difficulty", default=DIFFICULTY)
    parser.add_argument("--max-ticks", type=int, default=6000)
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    all_returns, all_ticks_remaining, all_troop_ratio, all_tile_fraction = [], [], [], []
    wins, losses, timeouts = 0, 0, 0

    seed_counter = 0
    for threshold in thresholds:
        for _ in range(args.seeds_per_threshold):
            seed_counter += 1
            seed = str(1000 * int(threshold * 1000) + seed_counter)
            trace, won = run_episode(seed, threshold, args.bots, args.nations, args.difficulty, args.max_ticks)
            if won is True:
                wins += 1
            elif won is False:
                losses += 1
            else:
                timeouts += 1

            troop_ratios = np.array([t[0] for t in trace])
            tile_fractions = np.array([t[1] for t in trace])
            rewards = [t[2] for t in trace]
            returns = realized_returns(rewards)
            ticks_remaining = np.arange(len(trace) - 1, -1, -1)

            print(
                f"threshold={threshold:.3f} seed={seed} ticks={len(trace)} "
                f"won={won} final_troop_ratio={troop_ratios[-1]:.3f}"
            )

            all_returns.append(returns)
            all_ticks_remaining.append(ticks_remaining)
            all_troop_ratio.append(troop_ratios)
            all_tile_fraction.append(tile_fractions)

    returns = np.concatenate(all_returns)
    ticks_remaining = np.concatenate(all_ticks_remaining).astype(np.float64)
    troop_ratio = np.concatenate(all_troop_ratio)
    tile_fraction = np.concatenate(all_tile_fraction)

    print(f"\n{wins} wins, {losses} losses, {timeouts} timeouts, {len(returns)} total ticks\n")

    print("correlations with realized return:")
    print(f"  ticks_remaining: {np.corrcoef(ticks_remaining, returns)[0, 1]:.3f}")
    print(f"  troop_ratio:     {np.corrcoef(troop_ratio, returns)[0, 1]:.3f}")
    print(f"  tile_fraction:   {np.corrcoef(tile_fraction, returns)[0, 1]:.3f}")

    def r2(X: np.ndarray, y: np.ndarray) -> float:
        X = np.column_stack([X, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

    r2_ticks_only = r2(ticks_remaining.reshape(-1, 1), returns)
    r2_ticks_and_state = r2(np.column_stack([ticks_remaining, troop_ratio, tile_fraction]), returns)

    print(f"\nR^2 predicting realized return from ticks_remaining alone: {r2_ticks_only:.3f}")
    print(f"R^2 adding troop_ratio + tile_fraction on top:              {r2_ticks_and_state:.3f}")
    print(f"state's marginal contribution:                              {r2_ticks_and_state - r2_ticks_only:.3f}")
