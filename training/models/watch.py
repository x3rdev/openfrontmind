import argparse
from pathlib import Path

import numpy as np
import torch

from training.envs.bridge import OpenFrontBridge
from training.envs.game_state import apply_player_updates, apply_tile_updates, apply_unit_updates, apply_max_troops, apply_land_bitmap
from training.envs.map_view import draw_troop_counts, draw_units, outline_agent_territory, render
from training.envs.observation import encode_observation
from training.envs.action_space import build_entities, pad_entities, action_to_intents, DO_NOTHING
from training.envs.running_stats import RunningMeanStd
from training.models.policy import Agent
from training.models.train import latest_checkpoint, BOTS, NATIONS, DIFFICULTY


def main(
    checkpoint_path: str | Path, seed: str, scale: int, headless: bool = False, diagnostic: bool = False,
    log_probs: bool = False, prob_log_interval: int = 1,
    log_value: bool = False, value_log_interval: int = 1,
    bots: int = BOTS, nations: int = NATIONS, difficulty: str = DIFFICULTY,
    max_ticks: int | None = None,
) -> dict | None:
    print(f"using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    # frozen, not updated - inference should use exactly the normalization the
    # weights were trained against, not one that keeps adapting mid-game
    gold_stats = RunningMeanStd()
    gold_stats.mean = checkpoint["gold_mean"]
    gold_stats.var = checkpoint["gold_var"]
    gold_stats.count = checkpoint["gold_count"]

    with OpenFrontBridge() as bridge:
        setup = bridge.reset(seed=seed, map="mid_plains", bots=bots, nations=nations, difficulty=difficulty)
        agent_client_id = setup["agentClientID"]
        agent_small_id = setup["agentSmallID"]
        agent_name = setup["agentName"]
        player_types = {int(sid): t for sid, t in setup["playerTypes"].items()}
        width, height = setup["width"], setup["height"]

        owner_grid = np.zeros((height, width), dtype=np.uint32)
        land_mask = np.zeros((height, width), dtype=bool)
        is_mine = np.zeros((height, width), dtype=bool)
        is_enemy = np.zeros((height, width), dtype=bool)
        player_stats: dict = {}
        units: dict = {}
        apply_tile_updates(owner_grid, land_mask, width, setup["packedTileUpdates"], agent_small_id, is_mine, is_enemy)
        apply_land_bitmap(land_mask, width, setup["landBitmap"])
        apply_player_updates(player_stats, setup["packedPlayerUpdates"])
        apply_max_troops(player_stats, setup["maxTroops"])
        apply_unit_updates(units, setup["unitUpdates"])

        total_land = setup["totalLand"]
        spatial, scalar = encode_observation(land_mask, is_mine, is_enemy, player_stats, agent_small_id, total_land)
        entity_features, targets = build_entities([], player_stats, player_types, total_land)
        scalar[0] = gold_stats.normalize(scalar[0])
        entity_features[:, 5] = gold_stats.normalize(entity_features[:, 5])
        entities, entity_mask = pad_entities(entity_features)

        agent = Agent.from_observation(spatial, scalar, entities)
        agent.load_state_dict(checkpoint["model"])
        agent.eval()

        if not headless:
            import cv2

            cv2.namedWindow("map")
            img = render(owner_grid, land_mask, agent_small_id, player_types).to_bgr_uint8(scale=scale)
            draw_units(img, units, width, scale)
            draw_troop_counts(img, owner_grid, land_mask, player_stats, agent_small_id, scale)
            cv2.imshow("map", outline_agent_territory(img, owner_grid, agent_small_id, scale))
            cv2.waitKey(1)

        # decision-time troop_ratio/tile_fraction, tracked before stepping - lets us
        # see whether attacks cluster at high troop_ratio (patience) or fire immediately
        # regardless of ratio (the greedy failure mode from the CLAUDE.md diagnostic).
        do_nothing_ticks = 0
        attack_ticks = 0
        attack_troop_ratios = []
        first_attack_tick = None
        first_attack_troop_ratio = None
        peak_tile_fraction = 0.0
        peak_tick = 0

        # (troop_ratio, p_do_nothing) pairs, for the end-of-game bucketed summary -
        # answers "does p(wait) actually track troop_ratio, or is it flat/dilution-bound".
        prob_samples = []

        # (tick, value, troop_ratio, tile_fraction, incoming_ratio) - answers "is V(s)
        # state-aware, or is it just fitting a smooth curve against elapsed tick number"
        # (see CLAUDE.md's earlier value-trace finding, from before troop_ratio was in
        # the observation). incoming_ratio added alongside outgoing/incoming troops.
        value_samples = []

        for i in range(int(1.0e100)):
            with torch.no_grad():
                result = agent.get_action_and_value(
                    spatial=torch.from_numpy(spatial).unsqueeze(0),
                    scalar=torch.from_numpy(scalar).unsqueeze(0),
                    entities=torch.from_numpy(entities).unsqueeze(0),
                    entity_mask=torch.from_numpy(entity_mask).unsqueeze(0),
                    action=None,
                    return_probs=log_probs,
                )
            action, log_prob, entropy, value = result[:4]
            action_idx = action.item()
            target = targets[action_idx] if action_idx < len(targets) else DO_NOTHING

            if log_probs:
                probs = result[4][0]  # (MAX_ENTITIES,)
                # do_nothing is always the last row build_entities appends - unless
                # truncated away by pad_entities when >MAX_ENTITIES candidates existed,
                # in which case it wasn't a selectable action this tick at all.
                do_nothing_rows = np.where(entity_features[:, 3] == 1.0)[0]
                do_nothing_idx = int(do_nothing_rows[0]) if len(do_nothing_rows) and do_nothing_rows[0] < len(probs) else None
                p_do_nothing = probs[do_nothing_idx].item() if do_nothing_idx is not None else None
                n_candidates = len(targets)
                troop_ratio_now = float(scalar[1])

                if p_do_nothing is not None:
                    prob_samples.append((troop_ratio_now, p_do_nothing))

                if i % prob_log_interval == 0:
                    chosen = "do_nothing" if target is DO_NOTHING else "attack"
                    p_str = "n/a(truncated)" if p_do_nothing is None else f"{p_do_nothing:.3f}"
                    # what's actually being attacked - is_tribe/is_nation/is_wilderness,
                    # tile_fraction, troop_ratio (see ENTITY_FEATURES order in action_space.py)
                    target_str = ""
                    if target is not DO_NOTHING and action_idx < len(entity_features):
                        row = entity_features[action_idx]
                        kind = "tribe" if row[0] else "nation" if row[1] else "wilderness" if row[2] else "?"
                        target_str = (
                            f" target_kind={kind} target_tile_fraction={row[4]:.3f} "
                            f"target_troop_ratio={row[6]:.3f} target_attacking_me_ratio={row[7]:.3f}"
                        )
                    print(
                        f"[prob] tick={i} troop_ratio={troop_ratio_now:.3f} n_candidates={n_candidates} "
                        f"p_do_nothing={p_str} p_attack={'n/a' if p_do_nothing is None else f'{1 - p_do_nothing:.3f}'} "
                        f"chosen={chosen}{target_str}"
                    )

            if log_value:
                value_now = value.item()
                troop_ratio_now = float(scalar[1])
                tile_fraction_now = float(scalar[2])
                incoming_ratio_now = float(scalar[4])
                value_samples.append((i, value_now, troop_ratio_now, tile_fraction_now, incoming_ratio_now))
                if i % value_log_interval == 0:
                    print(
                        f"[value] tick={i} V={value_now:.3f} troop_ratio={troop_ratio_now:.3f} "
                        f"tile_fraction={tile_fraction_now:.3f} incoming_ratio={incoming_ratio_now:.3f}"
                    )

            if diagnostic:
                troop_ratio = float(scalar[1])
                tile_fraction = float(scalar[2])
                if target is DO_NOTHING:
                    do_nothing_ticks += 1
                else:
                    attack_ticks += 1
                    attack_troop_ratios.append(troop_ratio)
                    if first_attack_tick is None:
                        first_attack_tick = i
                        first_attack_troop_ratio = troop_ratio
                if tile_fraction > peak_tile_fraction:
                    peak_tile_fraction = tile_fraction
                    peak_tick = i

            intents = action_to_intents(target=target, agent_client_id=agent_client_id)
            step_result = bridge.step(intents)
            attack_mask = step_result["attackMask"]

            apply_tile_updates(
                owner_grid, land_mask, width, step_result["packedTileUpdates"],
                agent_small_id, is_mine, is_enemy
            )
            apply_player_updates(player_stats, step_result["packedPlayerUpdates"])
            apply_max_troops(player_stats, step_result["maxTroops"])
            apply_unit_updates(units, step_result["unitUpdates"])

            # the wider FFA game runs long after our agent is wiped out - train.py
            # already treats tilesOwned==0 as terminal for this reason (see its
            # eliminated check); watch.py needs the same or it keeps "observing" a
            # dead agent whose troops/maxTroops go stale/degenerate.
            eliminated = player_stats.get(agent_small_id, {}).get("tilesOwned", 0.0) == 0.0

            total_land = step_result["totalLand"]
            spatial, scalar = encode_observation(
                land_mask, is_mine, is_enemy, player_stats, agent_small_id, total_land,
                step_result["outgoingTroops"], step_result["incomingTroops"],
            )
            entity_features, targets = build_entities(attack_mask, player_stats, player_types, total_land)
            scalar[0] = gold_stats.normalize(scalar[0])
            entity_features[:, 5] = gold_stats.normalize(entity_features[:, 5])
            entities, entity_mask = pad_entities(entity_features)

            if not headless and i % 10 == 0:
                img = render(owner_grid, land_mask, agent_small_id, player_types).to_bgr_uint8(scale=scale)
                draw_units(img, units, width, scale)
                draw_troop_counts(img, owner_grid, land_mask, player_stats, agent_small_id, scale)
                cv2.imshow("map", outline_agent_territory(img, owner_grid, agent_small_id, scale))
                # waitKey also pumps the window's event loop - required for it to
                # actually repaint, not just a keypress wait.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if step_result["done"] or eliminated:
                if eliminated and not step_result["done"]:
                    step_result["winner"] = "eliminated"
                print("game over, winner:", step_result["winner"])
                break

            if max_ticks is not None and i >= max_ticks:
                print(f"stopped early at max_ticks={max_ticks}")
                step_result["winner"] = None
                break

        if not headless:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if log_value and len(value_samples) > 1:
            ticks, values, ratios, tile_fracs, incoming_ratios = (np.array(x) for x in zip(*value_samples))
            corr_tick = np.corrcoef(ticks, values)[0, 1]
            corr_ratio = np.corrcoef(ratios, values)[0, 1]
            corr_tile = np.corrcoef(tile_fracs, values)[0, 1]
            corr_incoming = np.corrcoef(incoming_ratios, values)[0, 1] if incoming_ratios.std() > 0 else float("nan")
            print(
                f"[value-summary] seed={seed} n={len(value_samples)} "
                f"corr(V,tick)={corr_tick:.3f} corr(V,troop_ratio)={corr_ratio:.3f} "
                f"corr(V,tile_fraction)={corr_tile:.3f} corr(V,incoming_ratio)={corr_incoming:.3f}"
            )

        if log_probs and prob_samples:
            # bucket by troop_ratio decile - a real "wait when low" policy should show
            # p_do_nothing falling as the ratio climbs; a dilution-bound one stays flat.
            buckets = [[] for _ in range(10)]
            for ratio, p in prob_samples:
                idx = min(int(ratio * 10), 9)
                buckets[idx].append(p)
            print(f"[prob-summary] seed={seed}")
            for b, ps in enumerate(buckets):
                if ps:
                    print(f"  troop_ratio [{b/10:.1f},{(b+1)/10:.1f}): n={len(ps):5d} mean_p_do_nothing={sum(ps)/len(ps):.3f}")

        if diagnostic:
            mean_attack_troop_ratio = sum(attack_troop_ratios) / len(attack_troop_ratios) if attack_troop_ratios else None
            summary = {
                "seed": seed,
                "final_tick": i,
                "winner": step_result["winner"],
                "won": step_result["winner"] == agent_name,
                "do_nothing_ticks": do_nothing_ticks,
                "attack_ticks": attack_ticks,
                "first_attack_tick": first_attack_tick,
                "first_attack_troop_ratio": first_attack_troop_ratio,
                "mean_attack_troop_ratio": mean_attack_troop_ratio,
                "peak_tile_fraction": peak_tile_fraction,
                "peak_tick": peak_tick,
            }
            print(
                f"[diagnostic] seed={seed} winner={step_result['winner']} won={summary['won']} "
                f"ticks={i} do_nothing={do_nothing_ticks} attack={attack_ticks} "
                f"first_attack_tick={first_attack_tick} first_attack_troop_ratio="
                f"{first_attack_troop_ratio if first_attack_troop_ratio is None else f'{first_attack_troop_ratio:.3f}'} "
                f"mean_attack_troop_ratio="
                f"{mean_attack_troop_ratio if mean_attack_troop_ratio is None else f'{mean_attack_troop_ratio:.3f}'} "
                f"peak_tile_fraction={peak_tile_fraction:.3f}@{peak_tick}"
            )
            return summary

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="defaults to the most recent checkpoint")
    parser.add_argument("--seed", default="seedseed", help="comma-separated for multiple runs")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--headless", action="store_true", help="skip the cv2 window")
    parser.add_argument("--diagnostic", action="store_true", help="log troop_ratio/action stats and print an end-of-game summary")
    parser.add_argument("--log-probs", action="store_true", help="log p(do_nothing) vs p(attack) per tick plus a troop_ratio-bucketed summary")
    parser.add_argument("--prob-log-interval", type=int, default=1, help="print a [prob] line every N ticks (only with --log-probs)")
    parser.add_argument("--log-value", action="store_true", help="log V(s) per tick plus corr(V,tick)/corr(V,troop_ratio)/corr(V,tile_fraction)")
    parser.add_argument("--value-log-interval", type=int, default=1, help="print a [value] line every N ticks (only with --log-value)")
    parser.add_argument("--bots", type=int, default=BOTS, help=f"defaults to train.py's current BOTS ({BOTS}) - keep in sync with what the checkpoint was actually trained against")
    parser.add_argument("--nations", type=int, default=NATIONS, help=f"defaults to train.py's current NATIONS ({NATIONS})")
    parser.add_argument("--difficulty", default=DIFFICULTY, help=f"defaults to train.py's current DIFFICULTY ({DIFFICULTY})")
    parser.add_argument("--max-ticks", type=int, default=None, help="stop early after N ticks - for fast cross-checkpoint probing instead of playing out a full game")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or latest_checkpoint()
    if checkpoint_path is None:
        raise SystemExit("no checkpoint found in training/models/checkpoints/ - train first")

    seeds = args.seed.split(",")
    summaries = []
    for seed in seeds:
        summaries.append(main(
            checkpoint_path=checkpoint_path, seed=seed, scale=args.scale,
            headless=args.headless, diagnostic=args.diagnostic,
            log_probs=args.log_probs, prob_log_interval=args.prob_log_interval,
            log_value=args.log_value, value_log_interval=args.value_log_interval,
            bots=args.bots, nations=args.nations, difficulty=args.difficulty,
            max_ticks=args.max_ticks,
        ))

    if args.diagnostic and len(seeds) > 1:
        summaries = [s for s in summaries if s is not None]
        wins = sum(1 for s in summaries if s["won"])
        mean_peak = sum(s["peak_tile_fraction"] for s in summaries) / len(summaries)
        ratios = [s["first_attack_troop_ratio"] for s in summaries if s["first_attack_troop_ratio"] is not None]
        mean_first_ratio = sum(ratios) / len(ratios) if ratios else None
        print(
            f"[aggregate] n={len(summaries)} wins={wins} mean_peak_tile_fraction={mean_peak:.3f} "
            f"mean_first_attack_troop_ratio="
            f"{mean_first_ratio if mean_first_ratio is None else f'{mean_first_ratio:.3f}'}"
        )
