import argparse

import numpy as np
import torch

from training.envs.bridge import OpenFrontBridge
from training.envs.game_state import apply_player_updates, apply_tile_updates, apply_unit_updates
from training.envs.map_view import draw_units, outline_agent_territory, render
from training.envs.observation import encode_observation
from training.envs.action_space import build_entities, pad_entities, action_to_intents, DO_NOTHING
from training.models.policy import Agent
from training.models.train import latest_checkpoint


def main(checkpoint_path: str, seed: str, scale: int) -> None:
    print(f"using checkpoint: {checkpoint_path}")
    with OpenFrontBridge() as bridge:
        setup = bridge.reset(seed=seed, map="mid_plains", bots=3, nations=1)
        agent_client_id = setup["agentClientID"]
        agent_small_id = setup["agentSmallID"]
        player_types = {int(sid): t for sid, t in setup["playerTypes"].items()}
        width, height = setup["width"], setup["height"]

        owner_grid = np.zeros((height, width), dtype=np.uint32)
        land_mask = np.zeros((height, width), dtype=bool)
        player_stats: dict = {}
        units: dict = {}
        apply_tile_updates(owner_grid, land_mask, width, setup["packedTileUpdates"])
        apply_player_updates(player_stats, setup["packedPlayerUpdates"])
        apply_unit_updates(units, setup["unitUpdates"])

        spatial, scalar = encode_observation(owner_grid, land_mask, player_stats, agent_small_id)
        entity_features, targets = build_entities([], player_stats, player_types, land_mask.sum())
        entities, entity_mask = pad_entities(entity_features)

        agent = Agent.from_observation(spatial, scalar, entities)
        agent.load_state_dict(torch.load(checkpoint_path))
        agent.eval()

        import cv2

        cv2.namedWindow("map")
        img = render(owner_grid, land_mask, agent_small_id, player_types).to_bgr_uint8(scale=scale)
        draw_units(img, units, width, scale)
        cv2.imshow("map", outline_agent_territory(img, owner_grid, agent_small_id, scale))
        cv2.waitKey(1)

        for i in range(int(1.0e100)):
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(
                    spatial=torch.from_numpy(spatial).unsqueeze(0),
                    scalar=torch.from_numpy(scalar).unsqueeze(0),
                    entities=torch.from_numpy(entities).unsqueeze(0),
                    entity_mask=torch.from_numpy(entity_mask).unsqueeze(0),
                    action=None
                )
            action_idx = action.item()
            target = targets[action_idx] if action_idx < len(targets) else DO_NOTHING
            intents = action_to_intents(target=target, agent_client_id=agent_client_id)
            step_result = bridge.step(intents)
            attack_mask = step_result["attackMask"]

            apply_tile_updates(owner_grid, land_mask, width, step_result["packedTileUpdates"])
            apply_player_updates(player_stats, step_result["packedPlayerUpdates"])
            apply_unit_updates(units, step_result["unitUpdates"])

            spatial, scalar = encode_observation(owner_grid, land_mask, player_stats, agent_small_id)
            entity_features, targets = build_entities(attack_mask, player_stats, player_types, land_mask.sum())
            entities, entity_mask = pad_entities(entity_features)

            if i % 10 == 0:
                img = render(owner_grid, land_mask, agent_small_id, player_types).to_bgr_uint8(scale=scale)
                draw_units(img, units, width, scale)
                cv2.imshow("map", outline_agent_territory(img, owner_grid, agent_small_id, scale))
                # waitKey also pumps the window's event loop - required for it to
                # actually repaint, not just a keypress wait.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if step_result["done"]:
                print("game over, winner:", step_result["winner"])
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="defaults to the most recent checkpoint")
    parser.add_argument("--seed", default="seedseed")
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or latest_checkpoint()
    if checkpoint_path is None:
        raise SystemExit("no checkpoint found in training/models/checkpoints/ - train first")

    main(checkpoint_path=checkpoint_path, seed=args.seed, scale=args.scale)
