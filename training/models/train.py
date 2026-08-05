from collections import deque
from dataclasses import dataclass
from pathlib import Path

import random
import time
import numpy as np
import torch.optim

from training.envs.vec_bridge import VecOpenFrontBridge
from training.envs.game_state import apply_tile_updates, apply_player_updates
from training.envs.observation import encode_observation
from training.envs.action_space import build_action_mask, action_to_intents
from training.envs.reward import compute_reward
from training.models.policy import Agent

STEPS = 2000
NUM_UPDATES = 720  # ~8h overnight budget at the ~40s/update observed rate
NUM_ENVS = 8

# Curriculum: start with one Nation opponent; if it's winning nearly every
# game (not us, not a Tribe), the FFA field is too easy to dominate for it -
# bump to a full nation pool so nations spend more of the game fighting each
# other instead of just steamrolling the agent. One-shot (curriculum_done
# latches) so it doesn't oscillate once bumped.
NATIONS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_INTERVAL = 50
CHECKPOINT_KEEP = 5


def save_checkpoint(agent, update: int) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"agent_{time.strftime('%Y%m%d_%H%M%S')}_update{update}.pt"
    torch.save(agent.state_dict(), path)

    checkpoints = sorted(CHECKPOINT_DIR.glob("agent_*.pt"))
    for stale in checkpoints[:-CHECKPOINT_KEEP]:
        stale.unlink()

    return path


def latest_checkpoint() -> Path | None:
    checkpoints = sorted(CHECKPOINT_DIR.glob("agent_*.pt")) if CHECKPOINT_DIR.exists() else []
    return checkpoints[-1] if checkpoints else None

LR = 1e-4
GAMMA = 0.9995
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.05

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
    player_types: dict
    spatial: np.ndarray
    scalar: np.ndarray
    mask: np.ndarray
    slot_targets: list


def _episode_state_from_setup(setup: dict) -> EpisodeState:
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
        player_types={int(sid): t for sid, t in setup["playerTypes"].items()},
        spatial=spatial,
        scalar=scalar,
        mask=mask,
        slot_targets=slot_targets,
    )


def start_episode(bridge) -> EpisodeState:
    """Used for standalone scripts and per-slot resets in run_update."""
    setup = bridge.reset(
        seed=str(random.randint(0, 2 ** 31 - 1)),
        map="big_plains",
        bots=3,
        nations=NATIONS
    )
    return _episode_state_from_setup(setup)


def start_all_episodes(vec_bridge: VecOpenFrontBridge) -> list[EpisodeState]:
    """Each slot needs its own random seed - sharing one would clone the same
    deterministic game across every env."""
    return list(vec_bridge.executor.map(start_episode, vec_bridge.bridges))


def run_update(
    vec_bridge: VecOpenFrontBridge, agent, optimizer, episodes: list[EpisodeState], update: int
) -> tuple[list[EpisodeState], int, int]:
    """One STEPS-length rollout across all NUM_ENVS envs, GAE per env, one PPO
    update over the flattened STEPS*NUM_ENVS batch. Rollout buffers stay on
    CPU - too large for VRAM at STEPS=2000 - only per-tick and per-minibatch
    slices move to DEVICE.
    """
    rollout_start = time.perf_counter()

    n = len(episodes)
    spatial_buf = np.zeros((STEPS, n, *episodes[0].spatial.shape), dtype=np.float32)
    scalar_buf = np.zeros((STEPS, n, *episodes[0].scalar.shape), dtype=np.float32)
    masks = np.zeros((STEPS, n, *episodes[0].mask.shape), dtype=np.bool)
    actions = np.zeros((STEPS, n))
    log_probs = np.zeros((STEPS, n))
    values = np.zeros((STEPS, n))
    rewards = np.zeros((STEPS, n))
    dones = np.zeros((STEPS, n), dtype=np.bool)
    wins = np.zeros((STEPS, n), dtype=np.bool)

    for i in range(STEPS):
        spatial_batch = np.stack([e.spatial for e in episodes])
        scalar_batch = np.stack([e.scalar for e in episodes])
        mask_batch = np.stack([e.mask for e in episodes])

        with torch.no_grad():
            action, log_prob, entropy, value = agent.get_action_and_value(
                spatial=torch.from_numpy(spatial_batch).to(DEVICE),
                scalar=torch.from_numpy(scalar_batch).to(DEVICE),
                mask=torch.from_numpy(mask_batch).to(DEVICE),
                action=None
            )
        actions_np = action.cpu().numpy()
        log_probs_np = log_prob.cpu().numpy()
        values_np = value.squeeze(-1).cpu().numpy()

        intents_list = [
            action_to_intents(
                action_idx=int(actions_np[k]),
                slot_targets=episodes[k].slot_targets,
                agent_client_id=episodes[k].agent_client_id,
            )
            for k in range(n)
        ]
        step_results = vec_bridge.step(intents_list)

        for k in range(n):
            episode = episodes[k]
            step_result = step_results[k]
            attack_mask = step_result["attackMask"]
            tile_updates = step_result["packedTileUpdates"]
            player_updates = step_result["packedPlayerUpdates"]

            # snapshot before applying this tick's diffs, for the reward delta
            prev_stats = episode.player_stats.get(episode.agent_small_id, {})
            spatial_buf[i, k] = episode.spatial
            scalar_buf[i, k] = episode.scalar
            masks[i, k] = episode.mask
            actions[i, k] = actions_np[k]
            log_probs[i, k] = log_probs_np[k]
            values[i, k] = values_np[k]

            apply_player_updates(episode.player_stats, player_updates)
            apply_tile_updates(episode.owner_grid, episode.land_mask, episode.width, tile_updates)

            curr_stats = episode.player_stats.get(episode.agent_small_id, {})
            # the wider FFA game runs long after our agent is wiped out, so elimination
            # counts as its own loss rather than waiting for someone else to win
            eliminated = curr_stats.get("tilesOwned", 0.0) == 0.0
            done = step_result["done"] or eliminated
            dones[i, k] = done
            won = (not eliminated) and step_result["winner"] == episode.agent_name
            wins[i, k] = won

            reward = compute_reward(prev_stats, curr_stats, done, won, episode.land_mask.sum())
            rewards[i, k] = reward

            episode.spatial, episode.scalar = encode_observation(
                episode.owner_grid, episode.land_mask, episode.player_stats, episode.agent_small_id
            )
            episode.mask, episode.slot_targets = build_action_mask(attack_mask)

            # reset just this slot - the others keep ticking independently
            if done:
                episodes[k] = start_episode(vec_bridge.bridges[k])

    with torch.no_grad():
        last_spatial = np.stack([e.spatial for e in episodes])
        last_scalar = np.stack([e.scalar for e in episodes])
        last_value = agent.get_value(
            spatial=torch.from_numpy(last_spatial).to(DEVICE),
            scalar=torch.from_numpy(last_scalar).to(DEVICE),
        ).squeeze(-1).cpu()

    spatial_buf = torch.from_numpy(spatial_buf)
    scalar_buf = torch.from_numpy(scalar_buf)
    masks = torch.from_numpy(masks)
    actions = torch.from_numpy(actions)
    log_probs = torch.from_numpy(log_probs)
    values = torch.from_numpy(values)
    rewards = torch.from_numpy(rewards)
    dones = torch.from_numpy(dones)

    running_advantage = torch.zeros(n)
    next_value = last_value
    advantages = torch.zeros(STEPS, n)
    returns = torch.zeros(STEPS, n)

    for t in reversed(range(STEPS)):
        non_term = 1 - dones[t].float()

        delta = rewards[t] + (GAMMA * next_value * non_term) - values[t]
        running_advantage = delta + (GAMMA * GAE_LAMBDA * non_term * running_advantage)
        next_value = values[t]

        advantages[t] = running_advantage
        returns[t] = advantages[t] + values[t]

    advantages = (advantages-advantages.mean()) / (advantages.std()+1e-8)

    # flatten (STEPS, n, ...) -> (STEPS*n, ...) for minibatching
    total = STEPS * n
    spatial_buf = spatial_buf.reshape(total, *spatial_buf.shape[2:])
    scalar_buf = scalar_buf.reshape(total, *scalar_buf.shape[2:])
    masks = masks.reshape(total, *masks.shape[2:])
    actions = actions.reshape(total)
    log_probs = log_probs.reshape(total)
    advantages = advantages.reshape(total)
    returns = returns.reshape(total)

    rollout_elapsed = time.perf_counter() - rollout_start
    ppo_start = time.perf_counter()

    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    num_minibatches = 0

    for epoch in range(PPO_EPOCHS):
        indices = torch.randperm(total)
        for start in range(0, total, MINIBATCH_SIZE):
            batch_idx = indices[start:start+MINIBATCH_SIZE]

            new_action, new_log_prob, entropy, new_value = agent.get_action_and_value(
                spatial=spatial_buf[batch_idx].to(DEVICE),
                scalar=scalar_buf[batch_idx].to(DEVICE),
                mask=masks[batch_idx].to(DEVICE),
                action=actions[batch_idx].to(DEVICE)
            )

            batch_log_probs = log_probs[batch_idx].to(DEVICE)
            batch_advantages = advantages[batch_idx].to(DEVICE)
            batch_returns = returns[batch_idx].to(DEVICE)

            ratio = torch.exp(new_log_prob-batch_log_probs)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advantages

            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.square(new_value.squeeze(-1) - batch_returns).mean()

            entropy_bonus = entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy_bonus.item()
            num_minibatches += 1

    episodes_finished = int(dones.sum().item())
    episode_wins = int(wins.sum().item())

    ppo_elapsed = time.perf_counter() - ppo_start
    total_elapsed = rollout_elapsed + ppo_elapsed
    ticks_per_sec = (STEPS * n) / rollout_elapsed

    print(
        f"update {update}: mean_reward={rewards.mean().item():.3f} "
        f"episodes_finished={episodes_finished} wins={episode_wins} "
        f"policy_loss={policy_loss_sum / num_minibatches:.4f} "
        f"value_loss={value_loss_sum / num_minibatches:.4f} "
        f"entropy={entropy_sum / num_minibatches:.4f} "
        f"time={total_elapsed:.1f}s (rollout={rollout_elapsed:.1f}s ppo={ppo_elapsed:.1f}s) "
        f"ticks/sec={ticks_per_sec:.1f}"
    )

    return episodes, episodes_finished, episode_wins


def train():
    with VecOpenFrontBridge(NUM_ENVS) as vec_bridge:
        episodes = start_all_episodes(vec_bridge)

        agent = Agent.from_observation(episodes[0].spatial, episodes[0].scalar, episodes[0].mask).to(DEVICE)
        checkpoint = latest_checkpoint()
        if checkpoint is not None:
            agent.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
            print(f"resumed from {checkpoint}")
        optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
        print(f"training on device={DEVICE}, num_envs={NUM_ENVS}")

        total_episodes = 0
        total_wins = 0

        for update in range(NUM_UPDATES):
            episodes, episodes_finished, episode_wins = run_update(vec_bridge, agent, optimizer, episodes, update)
            total_episodes += episodes_finished
            total_wins += episode_wins

            if update % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(agent, update)

        save_checkpoint(agent, NUM_UPDATES)

        win_rate = total_wins / total_episodes if total_episodes > 0 else 0.0
        print(
            f"training done: {NUM_UPDATES} updates, {total_episodes} episodes played, "
            f"{total_wins} wins ({win_rate:.1%} win rate)"
        )


if __name__ == "__main__": train()
