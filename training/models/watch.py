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
from training.models.train import latest_checkpoint, NATIONS, DIFFICULTY


def main(
    checkpoint_path: str | Path, seed: str, scale: int, headless: bool = False, diagnostic: bool = False,
    nations: int = NATIONS, difficulty: str = DIFFICULTY,
    max_ticks: int | None = None, map_name: str = "mid_plains",
) -> dict | None:
    print(f"using checkpoint: {checkpoint_path}")
    torch.manual_seed(zlib.crc32(seed.encode()))  # actions are sampled from the torch RNG
    checkpoint = torch.load(checkpoint_path)
    # normalization stays frozen at what the weights were trained with
    gold_stats = RunningMeanStd()
    gold_stats.mean = checkpoint["gold_mean"]
    gold_stats.var = checkpoint["gold_var"]
    gold_stats.count = checkpoint["gold_count"]

    with OpenFrontBridge() as bridge:
        setup = bridge.reset(seed=seed, map=map_name, bots=0, nations=nations, difficulty=difficulty)
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
        agent.load_state_dict(checkpoint["model"])
        agent.eval()

        if not headless:
            import cv2

            cv2.namedWindow("map")
            img = render(episode.owner_grid, episode.land_mask, episode.agent_small_id, episode.player_types).to_bgr_uint8(scale=scale)
            draw_units(img, units, width, scale)
            draw_troop_counts(img, episode.owner_grid, episode.land_mask, episode.player_stats, episode.agent_small_id, scale)
            cv2.imshow("map", outline_agent_territory(img, episode.owner_grid, episode.agent_small_id, scale))
            cv2.waitKey(1)

        do_nothing_ticks = 0
        attack_ticks = 0
        attack_troop_ratios = []
        first_attack_tick = None
        first_attack_troop_ratio = None
        peak_tile_fraction = 0.0
        peak_tick = 0

        for i in range(int(1.0e100)):
            with torch.no_grad():
                action = agent.get_action_and_value(
                    spatial=torch.from_numpy(spatial).unsqueeze(0),
                    scalar=torch.from_numpy(scalar).unsqueeze(0),
                    entities=torch.from_numpy(entities).unsqueeze(0),
                    entity_mask=torch.from_numpy(entity_mask).unsqueeze(0),
                    action=None,
                )[0]
            action_idx = int(np.floor(action.item()))
            size_frac = action.item() - action_idx
            target = targets[action_idx] if action_idx < len(targets) else DO_NOTHING

            troop_ratio_now = float(scalar[1])
            tile_fraction_now = float(scalar[2])

            if diagnostic:
                if target is DO_NOTHING:
                    do_nothing_ticks += 1
                else:
                    attack_ticks += 1
                    attack_troop_ratios.append(troop_ratio_now)
                    if first_attack_tick is None:
                        first_attack_tick = i
                        first_attack_troop_ratio = troop_ratio_now
                if tile_fraction_now > peak_tile_fraction:
                    peak_tile_fraction = tile_fraction_now
                    peak_tick = i

            troops = size_frac * episode.player_stats.get(episode.agent_small_id, {}).get("troops", 0.0)
            intents = action_to_intents(target=target, agent_client_id=episode.agent_client_id, troops=troops)
            step_result = bridge.step(intents)

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
                if cv2.waitKey(1) & 0xFF == ord("q"):  # waitKey also repaints the window
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
    parser.add_argument("--diagnostic", action="store_true", help="print per-game action stats and, over several seeds, an aggregate")
    parser.add_argument("--nations", type=int, default=NATIONS)
    parser.add_argument("--difficulty", default=DIFFICULTY)
    parser.add_argument("--max-ticks", type=int, default=None, help="stop early after N ticks")
    parser.add_argument("--map", dest="map_name", default="mid_plains")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or latest_checkpoint()
    if checkpoint_path is None:
        raise SystemExit("no checkpoint found - train first")

    seeds = args.seed.split(",")
    summaries = []
    for seed in seeds:
        summaries.append(main(
            checkpoint_path=checkpoint_path, seed=seed, scale=args.scale,
            headless=args.headless, diagnostic=args.diagnostic,
            nations=args.nations, difficulty=args.difficulty,
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
