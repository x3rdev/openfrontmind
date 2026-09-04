"""Zero-learning heuristic: wait until troop_ratio crosses TROOP_RATIO_TARGET, then
attack the weakest (fewest raw troops) bordering nation. Tests whether patience is
even a viable strategy at a given bots/nations lobby size, isolated from any RL
training dynamics - same method that proved patience beats greedy at nations=1/0.
"""
import argparse

from training.envs.bridge import OpenFrontBridge
from training.envs.game_state import apply_player_updates, apply_max_troops
from training.envs.action_space import DO_NOTHING, action_to_intents
from training.models.train import BOTS, NATIONS, DIFFICULTY

TROOP_RATIO_TARGET = 0.422


def weakest_nation_target(attack_mask: list, player_stats: dict, player_types: dict):
    """Bordering nation with the fewest raw troops (matches AiAttackBehavior.ts's
    'weakest'), falling back to wilderness (targetID=None) - territory only grows
    via attacks, so with no bordering nation yet this expands toward one instead
    of sitting frozen at spawn size forever. Returns DO_NOTHING if no candidate at
    all exists this tick."""
    best_nation, best_troops, has_wilderness = None, float("inf"), False
    for c in attack_mask:
        if c["alreadyAttacking"]:
            continue
        if c["targetID"] is None:
            has_wilderness = True
            continue
        if player_types.get(c["smallID"]) != "NATION":
            continue
        troops = player_stats.get(c["smallID"], {}).get("troops", 0.0)
        if troops < best_troops:
            best_nation, best_troops = c["targetID"], troops
    if best_nation is not None:
        return best_nation
    if has_wilderness:
        return None
    return DO_NOTHING


def run(seed: str, bots: int, nations: int, difficulty: str, max_ticks: int | None) -> dict:
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
        peak_tile_fraction, peak_tick = 0.0, 0
        first_attack_tick, first_attack_troop_ratio = None, None

        for i in range(int(1.0e100)):
            stats = player_stats.get(agent_small_id, {})
            max_troops = max(stats.get("maxTroops", 1.0), 1)
            troop_ratio = stats.get("troops", 0.0) / max_troops

            target = DO_NOTHING
            if troop_ratio >= TROOP_RATIO_TARGET:
                target = weakest_nation_target(attack_mask, player_stats, player_types)
                if target is not DO_NOTHING:
                    if first_attack_tick is None:
                        first_attack_tick = i
                        first_attack_troop_ratio = troop_ratio

            intents = action_to_intents(target=target, agent_client_id=agent_client_id)
            step_result = bridge.step(intents)
            attack_mask = step_result["attackMask"]

            apply_player_updates(player_stats, step_result["packedPlayerUpdates"])
            apply_max_troops(player_stats, step_result["maxTroops"])

            eliminated = player_stats.get(agent_small_id, {}).get("tilesOwned", 0.0) == 0.0
            total_land = step_result["totalLand"]
            tile_fraction = player_stats.get(agent_small_id, {}).get("tilesOwned", 0.0) / max(total_land, 1)
            if tile_fraction > peak_tile_fraction:
                peak_tile_fraction = tile_fraction
                peak_tick = i

            if step_result["done"] or eliminated:
                winner = "eliminated" if eliminated and not step_result["done"] else step_result["winner"]
                return {
                    "seed": seed, "won": winner == agent_name, "winner": winner, "ticks": i,
                    "peak_tile_fraction": peak_tile_fraction, "peak_tick": peak_tick,
                    "first_attack_tick": first_attack_tick,
                    "first_attack_troop_ratio": first_attack_troop_ratio,
                }

            if max_ticks is not None and i >= max_ticks:
                return {
                    "seed": seed, "won": None, "winner": None, "ticks": i,
                    "peak_tile_fraction": peak_tile_fraction, "peak_tick": peak_tick,
                    "first_attack_tick": first_attack_tick,
                    "first_attack_troop_ratio": first_attack_troop_ratio,
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="1", help="comma-separated for multiple runs")
    parser.add_argument("--bots", type=int, default=BOTS)
    parser.add_argument("--nations", type=int, default=NATIONS)
    parser.add_argument("--difficulty", default=DIFFICULTY)
    parser.add_argument("--max-ticks", type=int, default=6000)
    args = parser.parse_args()

    results = []
    for seed in args.seed.split(","):
        r = run(seed, args.bots, args.nations, args.difficulty, args.max_ticks)
        results.append(r)
        print(
            f"[patience-heuristic] seed={r['seed']} won={r['won']} winner={r['winner']} "
            f"ticks={r['ticks']} peak_tile_fraction={r['peak_tile_fraction']:.3f}@{r['peak_tick']} "
            f"first_attack_tick={r['first_attack_tick']} "
            f"first_attack_troop_ratio={r['first_attack_troop_ratio']}"
        )

    wins = sum(1 for r in results if r["won"])
    print(f"\n{wins}/{len(results)} wins")
