from dataclasses import dataclass

import random
import numpy as np
import torch.optim

from training.envs.bridge import OpenFrontBridge
from training.envs.game_state import apply_tile_updates, apply_player_updates
from training.envs.observation import encode_observation
from training.envs.action_space import build_action_mask, action_to_intents
from training.envs.reward import compute_reward
from training.models.policy import Agent

STEPS = 256

LR = 1e-4
GAMME = 0.99
GAE_LAMBDA = 0.95

@dataclass
class EpisodeState:
    width: int
    height: int
    owner_grid: np.ndarray
    land_mask: np.ndarray
    player_stats: dict
    agent_client_id: str
    agent_small_id: int
    agent_name: str
    spatial: np.ndarray
    scalar: np.ndarray
    mask: np.ndarray
    slot_targets: list


def start_episode(bridge) -> EpisodeState:
    """Reset the bridge and build everything a fresh episode needs. Same
    sequence needed both before the main loop starts and every time an
    episode ends inside it - pulled out once so neither call site can
    silently drift from the other.
    """
    setup = bridge.reset(
        seed=str(random.randint(0, 2 ** 31 - 1)),
        map="big_plains",
        bots=25,
        nations=1
    )
    width, height = setup["width"], setup["height"]

    owner_grid = np.zeros((height, width), dtype=np.uint32)
    land_mask = np.zeros((height, width), dtype=bool)
    player_stats = {}
    apply_tile_updates(owner_grid, land_mask, width, setup["packedTileUpdates"])
    apply_player_updates(player_stats, setup["packedPlayerUpdates"])

    agent_small_id = setup["agentSmallID"]
    spatial, scalar = encode_observation(owner_grid, land_mask, player_stats, agent_small_id)
    mask, slot_targets = build_action_mask([])

    return EpisodeState(
        width=width,
        height=height,
        owner_grid=owner_grid,
        land_mask=land_mask,
        player_stats=player_stats,
        agent_client_id=setup["agentClientID"],
        agent_small_id=agent_small_id,
        agent_name=setup["agentName"],
        spatial=spatial,
        scalar=scalar,
        mask=mask,
        slot_targets=slot_targets,
    )


def train():
    with OpenFrontBridge() as bridge:
        episode = start_episode(bridge)

        agent = Agent.from_observation(episode.spatial, episode.scalar, episode.mask)
        optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

        spatial_buf = np.zeros((STEPS, *episode.spatial.shape), dtype=np.float32)
        scalar_buf = np.zeros((STEPS, *episode.scalar.shape), dtype=np.float32)
        masks = np.zeros((STEPS, *episode.mask.shape), dtype=np.bool)
        actions = np.zeros(STEPS)
        log_probs = np.zeros(STEPS)
        values = np.zeros(STEPS)
        rewards = np.zeros(STEPS)
        dones = np.zeros(STEPS, dtype=np.bool)

        for i in range(STEPS):
            # ask the policy for an action given the current observation (no grad: not training yet, just collecting)
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(
                    spatial=torch.from_numpy(episode.spatial).unsqueeze(0),
                    scalar=torch.from_numpy(episode.scalar).unsqueeze(0),
                    mask=torch.from_numpy(episode.mask).unsqueeze(0),
                    action=None
                )
            # turn the chosen action index into real intents and advance the game one tick
            intents = action_to_intents(
                action_idx=action.item(),
                slot_targets=episode.slot_targets,
                agent_client_id=episode.agent_client_id
            )
            step_result = bridge.step(intents)
            attack_mask = step_result["attackMask"]
            tile_updates = step_result["packedTileUpdates"]
            player_updates = step_result["packedPlayerUpdates"]

            # snapshot the agent's stats before applying this tick's updates, for the reward delta
            prev_stats = episode.player_stats.get(episode.agent_small_id, {})
            spatial_buf[i] = episode.spatial
            scalar_buf[i] = episode.scalar
            masks[i] = episode.mask
            actions[i] = action
            log_probs[i] = log_prob
            values[i] = value



            # fold this tick's diffs into the running game state
            apply_player_updates(episode.player_stats, player_updates)
            apply_tile_updates(episode.owner_grid, episode.land_mask, episode.width, tile_updates)

            curr_stats = episode.player_stats.get(episode.agent_small_id, {})
            done = step_result["done"]
            dones[i] = done
            won = step_result["winner"] == episode.agent_name

            reward = compute_reward(prev_stats, curr_stats, done, won)
            rewards[i] = reward

            # re-encode the observation/mask for the next iteration's action selection
            episode.spatial, episode.scalar = encode_observation(
                episode.owner_grid, episode.land_mask, episode.player_stats, episode.agent_small_id
            )
            episode.mask, episode.slot_targets = build_action_mask(attack_mask)

            # game ended (win or loss) - start a fresh episode on the same bridge connection
            if done:
                episode = start_episode(bridge)

        with torch.no_grad():
            last_value = agent.get_value(
                spatial=torch.from_numpy(episode.spatial).unsqueeze(0),
                scalar=torch.from_numpy(episode.scalar).unsqueeze(0),
            )



if __name__ == "__main__": train()
