import argparse
import zlib
from pathlib import Path

import numpy as np
import torch

from training.envs.bridge import OpenFrontBridge
from training.envs.game_state import EpisodeState, new_masks, apply_reset, apply_step, apply_unit_updates
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
    log_entropy: bool = False, entropy_log_interval: int = 1,
    bots: int = BOTS, nations: int = NATIONS, difficulty: str = DIFFICULTY,
    max_ticks: int | None = None, map_name: str = "mid_plains",
) -> dict | None:
    print(f"using checkpoint: {checkpoint_path}")
    # env determinism alone isn't enough - get_action_and_value samples from the
    # global torch RNG, so without this the same seed still gives a different
    # trajectory (and potentially outcome) on every separate run of this script.
    torch.manual_seed(zlib.crc32(seed.encode()))
    checkpoint = torch.load(checkpoint_path)
    # frozen, not updated - inference should use exactly the normalization the
    # weights were trained against, not one that keeps adapting mid-game
    gold_stats = RunningMeanStd()
    gold_stats.mean = checkpoint["gold_mean"]
    gold_stats.var = checkpoint["gold_var"]
    gold_stats.count = checkpoint["gold_count"]

    with OpenFrontBridge() as bridge:
        setup = bridge.reset(seed=seed, map=map_name, bots=bots, nations=nations, difficulty=difficulty)
        width, height = setup["width"], setup["height"]

        owner_grid, land_mask, impassable_mask, is_mine, is_enemy = new_masks(width, height)
        units: dict = {}
        player_stats, agent_small_id, player_types = apply_reset(
            owner_grid, land_mask, impassable_mask, is_mine, is_enemy, width, setup,
        )
        apply_unit_updates(units, setup["unitUpdates"])

        total_land = setup["totalLand"]
        spatial, scalar, ratio_table = encode_observation(land_mask, impassable_mask, is_mine, is_enemy, player_stats, agent_small_id, total_land, owner_grid)
        entity_features, targets = build_entities([], player_stats, player_types, total_land)
        scalar[0] = gold_stats.normalize(scalar[0])
        entity_features[:, 5] = gold_stats.normalize(entity_features[:, 5])
        entities, entity_mask = pad_entities(entity_features)

        episode = EpisodeState(
            width=width, height=height,
            owner_grid=owner_grid, land_mask=land_mask, impassable_mask=impassable_mask,
            is_mine=is_mine, is_enemy=is_enemy, ratio_table=ratio_table, player_stats=player_stats,
            agent_client_id=setup["agentClientID"], agent_small_id=agent_small_id, agent_name=setup["agentName"],
            player_types=player_types,
            spatial=spatial, scalar=scalar, entities=entities, entity_mask=entity_mask, targets=targets,
        )

        agent = Agent.from_observation(spatial, scalar, entities)
        # shape mismatches (e.g. entity_dim growing) still raise under strict=False - drop
        # those keys so they init fresh instead of failing the load, same as train.py
        own_state = agent.state_dict()
        saved_state = checkpoint["model"]
        for k in [k for k, v in saved_state.items() if k in own_state and v.shape != own_state[k].shape]:
            del saved_state[k]
        agent.load_state_dict(saved_state, strict=False)
        agent.eval()

        if not headless:
            import cv2

            cv2.namedWindow("map")
            img = render(episode.owner_grid, episode.land_mask, episode.agent_small_id, episode.player_types).to_bgr_uint8(scale=scale)
            draw_units(img, units, width, scale)
            draw_troop_counts(img, episode.owner_grid, episode.land_mask, episode.player_stats, episode.agent_small_id, scale)
            cv2.imshow("map", outline_agent_territory(img, episode.owner_grid, episode.agent_small_id, scale))
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

        # (num_candidates, gate_entropy, target_entropy) - disentangles a less-confident
        # gate (behavioral) from a target_dist with more candidates to spread probability
        # over (structural, grows with nations count) - see CLAUDE.md's entropy-scaling
        # hypothesis. A fixed checkpoint run across nations=1/2/3 isolates the structural
        # effect: if candidate_count and target_entropy rise together while gate_entropy
        # stays flat, the hypothesis is confirmed.
        entropy_samples = []

        for i in range(int(1.0e100)):
            with torch.no_grad():
                result = agent.get_action_and_value(
                    spatial=torch.from_numpy(spatial).unsqueeze(0),
                    scalar=torch.from_numpy(scalar).unsqueeze(0),
                    entities=torch.from_numpy(entities).unsqueeze(0),
                    entity_mask=torch.from_numpy(entity_mask).unsqueeze(0),
                    action=None,
                    return_probs=log_probs,
                    return_entropy_components=log_entropy,
                )
            action, log_prob, entropy, value = result[:4]
            action_idx = action.item()
            target = targets[action_idx] if action_idx < len(targets) else DO_NOTHING

            # positional, not fixed indices - matches policy.py's append order (probs
            # first if requested, then entropy components), so either flag works alone.
            extra_idx = 4
            if log_probs:
                probs = result[extra_idx][0]  # (MAX_ENTITIES,)
                extra_idx += 1
            if log_entropy:
                gate_entropy, target_entropy, num_candidates = (x.item() for x in result[extra_idx:extra_idx + 3])
                extra_idx += 3

            if log_probs:
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

            if log_entropy:
                entropy_samples.append((num_candidates, gate_entropy, target_entropy))
                if i % entropy_log_interval == 0:
                    # ceiling = log(num_candidates) - target_entropy's max possible value
                    # if the distribution were perfectly uniform over that many candidates.
                    # target_entropy/ceiling near 1.0 across every nations count means the
                    # rise is purely "more candidates to spread over," not less confidence.
                    ceiling = np.log(num_candidates) if num_candidates > 1 else float("nan")
                    ratio_str = "n/a" if np.isnan(ceiling) or ceiling == 0 else f"{target_entropy / ceiling:.3f}"
                    print(
                        f"[entropy] tick={i} num_candidates={num_candidates:.0f} "
                        f"gate_entropy={gate_entropy:.4f} target_entropy={target_entropy:.4f} "
                        f"target_entropy/ceiling={ratio_str}"
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

            intents = action_to_intents(target=target, agent_client_id=episode.agent_client_id)
            step_result = bridge.step(intents)

            # apply_step's own eliminated/done check (mirrors train.py's - see its
            # docstring) covers the same "don't keep observing a dead agent" case this
            # used to compute locally; done here folds in eliminated the same way.
            # Mutates episode.spatial/scalar/targets in place - re-read them below.
            entity_features, _prev_stats, curr_stats, done, _won = apply_step(episode, step_result)
            spatial, scalar, targets = episode.spatial, episode.scalar, episode.targets
            apply_unit_updates(units, step_result["unitUpdates"])
            eliminated = curr_stats.get("tilesOwned", 0.0) == 0.0

            scalar[0] = gold_stats.normalize(scalar[0])
            entity_features[:, 5] = gold_stats.normalize(entity_features[:, 5])
            episode.entities, episode.entity_mask = pad_entities(entity_features)
            entities, entity_mask = episode.entities, episode.entity_mask

            if not headless and i % 10 == 0:
                img = render(episode.owner_grid, episode.land_mask, episode.agent_small_id, episode.player_types).to_bgr_uint8(scale=scale)
                draw_units(img, units, width, scale)
                draw_troop_counts(img, episode.owner_grid, episode.land_mask, episode.player_stats, episode.agent_small_id, scale)
                cv2.imshow("map", outline_agent_territory(img, episode.owner_grid, episode.agent_small_id, scale))
                # waitKey also pumps the window's event loop - required for it to
                # actually repaint, not just a keypress wait.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if done:
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

        if log_entropy and entropy_samples:
            candidates, gate_ents, target_ents = (np.array(x) for x in zip(*entropy_samples))
            ceilings = np.log(np.maximum(candidates, 1))
            valid = candidates > 1
            mean_ratio = float((target_ents[valid] / ceilings[valid]).mean()) if valid.any() else float("nan")
            print(
                f"[entropy-summary] seed={seed} n={len(entropy_samples)} "
                f"mean_num_candidates={candidates.mean():.2f} mean_gate_entropy={gate_ents.mean():.4f} "
                f"mean_target_entropy={target_ents.mean():.4f} mean_target_entropy/ceiling={mean_ratio:.3f}"
            )

        if diagnostic:
            mean_attack_troop_ratio = sum(attack_troop_ratios) / len(attack_troop_ratios) if attack_troop_ratios else None
            summary = {
                "seed": seed,
                "final_tick": i,
                "winner": step_result["winner"],
                "won": step_result["winner"] == episode.agent_name,
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
    parser.add_argument("--log-entropy", action="store_true", help="log gate_entropy vs target_entropy vs num_candidates per tick - disentangles a less-confident gate from a target_dist that just has more candidates to spread over (structural, scales with nations count)")
    parser.add_argument("--entropy-log-interval", type=int, default=1, help="print an [entropy] line every N ticks (only with --log-entropy)")
    parser.add_argument("--bots", type=int, default=BOTS, help=f"defaults to train.py's current BOTS ({BOTS}) - keep in sync with what the checkpoint was actually trained against")
    parser.add_argument("--nations", type=int, default=NATIONS, help=f"defaults to train.py's current NATIONS ({NATIONS})")
    parser.add_argument("--difficulty", default=DIFFICULTY, help=f"defaults to train.py's current DIFFICULTY ({DIFFICULTY})")
    parser.add_argument("--max-ticks", type=int, default=None, help="stop early after N ticks - for fast cross-checkpoint probing instead of playing out a full game")
    parser.add_argument("--map", dest="map_name", default="mid_plains", help="defaults to train.py's current map (mid_plains) - checkpoint wasn't trained on other maps, so results are exploratory")
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
            log_entropy=args.log_entropy, entropy_log_interval=args.entropy_log_interval,
            bots=args.bots, nations=args.nations, difficulty=args.difficulty,
            max_ticks=args.max_ticks, map_name=args.map_name,
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
