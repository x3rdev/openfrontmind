import argparse
import zlib
from pathlib import Path

import numpy as np
import torch

from training.envs.bridge import OpenFrontBridge
from training.envs.game_state import EpisodeState, new_masks, apply_reset, apply_step, apply_unit_updates
from training.envs.map_view import draw_troop_counts, draw_units, outline_agent_territory, render
from training.envs.observation import encode_observation
from training.envs.action_space import build_entities, pad_entities, action_to_intents, diplomacy_from_code, DO_NOTHING
from training.models.policy import Agent, decode_action, drop_gold_inputs, load_padded
from training.models.train import latest_checkpoint, BOTS, NATIONS, DIFFICULTY


def main(
    checkpoint_path: str | Path, seed: str, scale: int, headless: bool = False, diagnostic: bool = False,
    bots: int = BOTS, nations: int = NATIONS, difficulty: str = DIFFICULTY,
    max_ticks: int | None = None, map_name: str = "mid_plains", render_every: int = 10,
) -> dict | None:
    print(f"using checkpoint: {checkpoint_path}")
    torch.manual_seed(zlib.crc32(seed.encode()))  # actions are sampled from the torch RNG
    checkpoint = torch.load(checkpoint_path)

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
        load_padded(agent, drop_gold_inputs(checkpoint["model"]) if "gold_mean" in checkpoint else checkpoint["model"])
        agent.eval()

        if not headless:
            import cv2

            cv2.namedWindow("map", cv2.WINDOW_NORMAL)
            # big maps are drawn on a strided grid so a frame stays cheap
            stride = max(1, -(-max(width, episode.height) // 1000))
            draw_scale = scale if stride == 1 else 1

            def show():
                owners, land = episode.owner_grid[::stride, ::stride], episode.land_mask[::stride, ::stride]
                img = render(owners, land, episode.agent_small_id, episode.player_types).to_bgr_uint8(scale=draw_scale)
                draw_units(img, units, width, draw_scale / stride)
                draw_troop_counts(img, owners, land, episode.player_stats, episode.agent_small_id, draw_scale)
                cv2.imshow("map", outline_agent_territory(img, owners, episode.agent_small_id, draw_scale))

            show()
            cv2.resizeWindow("map", 900, 900)
            cv2.moveWindow("map", 300, 60)
            cv2.waitKey(1)

        diplomacy_counts = {"request": 0, "break": 0}
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
            attack_row, size_frac, diplomacy_code = decode_action(action.reshape(1).double())
            action_idx, size_frac, diplomacy_code = int(attack_row[0]), float(size_frac[0]), int(diplomacy_code[0])
            diplomacy = diplomacy_from_code(diplomacy_code, targets)
            if diplomacy is not None:
                diplomacy_counts[diplomacy[0]] += 1
            target = targets[action_idx] if action_idx < len(targets) else DO_NOTHING

            troop_ratio_now = float(scalar[0])
            tile_fraction_now = float(scalar[1])

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
            intents = action_to_intents(target=target, agent_client_id=episode.agent_client_id, troops=troops, diplomacy=diplomacy)
            step_result = bridge.step(intents)

            entity_features, _prev_stats, curr_stats, done, _won = apply_step(episode, step_result)
            spatial, scalar, targets = episode.spatial, episode.scalar, episode.targets
            apply_unit_updates(units, step_result["unitUpdates"])
            eliminated = curr_stats.get("tilesOwned", 0.0) == 0.0

            episode.entities, episode.entity_mask = pad_entities(entity_features)
            entities, entity_mask = episode.entities, episode.entity_mask

            if not headless and i % render_every == 0:
                show()
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
            alliances = step_result.get("alliances", [])
            agent_alliances = sum(episode.agent_small_id in (a["a"], a["b"]) for a in alliances)
            nation_alliances = len(alliances) - agent_alliances
            eliminations = sum(c["conqueror"] == episode.agent_small_id and c["type"] == "NATION" for c in step_result.get("conquests", []))
            enclosures = step_result.get("enclosures", [])
            encl_gained = sum(e["tiles"] for e in enclosures if e["capturing"] == episode.agent_small_id)
            encl_lost = sum(e["tiles"] for e in enclosures if e["victim"] == episode.agent_small_id)
            encl_events = sum(e["capturing"] == episode.agent_small_id for e in enclosures)
            first_alliance = min((a["tick"] for a in alliances), default=None)
            print(
                f"[diagnostic] seed={seed} winner={step_result['winner']} won={summary['won']} "
                f"ticks={i} do_nothing={do_nothing_ticks} attack={attack_ticks} "
                f"first_attack_tick={first_attack_tick} first_attack_troop_ratio="
                f"{first_attack_troop_ratio if first_attack_troop_ratio is None else f'{first_attack_troop_ratio:.3f}'} "
                f"mean_attack_troop_ratio="
                f"{mean_attack_troop_ratio if mean_attack_troop_ratio is None else f'{mean_attack_troop_ratio:.3f}'} "
                f"peak_tile_fraction={peak_tile_fraction:.3f}@{peak_tick} "
                f"requests_sent={diplomacy_counts['request']} breaks_sent={diplomacy_counts['break']} alliances_nation_nation={nation_alliances} alliances_with_agent={agent_alliances} first_alliance_tick={first_alliance} eliminations={eliminations} enclosure_events={encl_events} enclosure_tiles_gained={encl_gained} enclosure_tiles_lost={encl_lost} enclosures_all={len(enclosures)}"
            )
            return summary

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="defaults to the most recent checkpoint")
    parser.add_argument("--seed", default="seedseed", help="comma-separated for multiple runs")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--render-every", type=int, default=10, help="redraw the map every N decisions")
    parser.add_argument("--headless", action="store_true", help="skip the cv2 window")
    parser.add_argument("--diagnostic", action="store_true", help="print per-game action stats and, over several seeds, an aggregate")
    parser.add_argument("--bots", type=int, default=BOTS)
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
            bots=args.bots, nations=args.nations, difficulty=args.difficulty,
            max_ticks=args.max_ticks, map_name=args.map_name, render_every=args.render_every,
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
