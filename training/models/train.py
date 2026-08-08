from collections import deque
from dataclasses import dataclass
from pathlib import Path

import random
import time
import numpy as np
import torch.optim

from training.envs.vec_bridge import VecOpenFrontBridge
from training.envs.game_state import apply_tile_updates, apply_player_updates, apply_max_troops, apply_land_bitmap
from training.envs.observation import encode_observation
from training.envs.action_space import build_entities, pad_entities, action_to_intents, DO_NOTHING
from training.envs.reward import compute_reward
from training.envs.running_stats import RunningMeanStd
from training.models.policy import Agent


# 2000 undershoots mid_plains: measured avg episode length across a full run sat at
# ~2100-2150 ticks the whole time, meaning a chunk of episodes straddled the rollout
# boundary and never gave GAE their real terminal win/loss reward within one update.
STEPS = 3000
NUM_UPDATES = 720
NUM_ENVS = 24

# Tried bots=1,nations=0 briefly (single weak Tribe) to unstick a value function that
# only tracked tick number, not state (see the value-trace diagnostic: corr(V,tick)
# ~ -0.95 to -0.98, corr(V,troop_ratio) weak and sign-flipped, ~97% losses vs. a
# Nation). It did produce real wins immediately (4/8 at update 0) - but reverted back
# to nations=1: a Tribe barely fights back, so the existing greedy-spam-attack habit
# already wins there, meaning PPO would reinforce spamming as "correct" rather than
# push toward the patience that's actually needed against real competition. Back to
# bots=0,nations=1 (pure 1v1 vs. a strong Nation).
BOTS = 0
NATIONS = 1
# never set explicitly before now - bridge.reset()/harness.ts silently defaulted to
# Easy the whole time. Setting it explicitly so it can't silently drift (same class
# of bug as watch.py's stale hardcoded bots=3,nations=1 found earlier this session).
DIFFICULTY = "Easy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_INTERVAL = 32
CHECKPOINT_KEEP = 5


def save_checkpoint(agent, update: int, gold_stats: RunningMeanStd) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"agent_{time.strftime('%Y%m%d_%H%M%S')}_update{update}.pt"
    # gold_stats travels with the weights - watch.py or a resumed run need the exact
    # same normalization the weights were trained against, not a freshly-reset one.
    torch.save(
        {
            "model": agent.state_dict(),
            "gold_mean": gold_stats.mean,
            "gold_var": gold_stats.var,
            "gold_count": gold_stats.count,
        },
        path,
    )

    checkpoints = sorted(CHECKPOINT_DIR.glob("agent_*.pt"))
    for stale in checkpoints[:-CHECKPOINT_KEEP]:
        stale.unlink()

    return path


def latest_checkpoint() -> Path | None:
    checkpoints = sorted(CHECKPOINT_DIR.glob("agent_*.pt")) if CHECKPOINT_DIR.exists() else []
    return checkpoints[-1] if checkpoints else None

LR = 1e-4
GAMMA = 0.999667  # horizon = STEPS = 3000
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
    is_mine: np.ndarray
    is_enemy: np.ndarray
    player_stats: dict
    agent_client_id: str
    agent_small_id: int
    agent_name: str
    player_types: dict
    spatial: np.ndarray
    scalar: np.ndarray
    entities: np.ndarray
    entity_mask: np.ndarray
    targets: list


def _apply_gold_stats(scalar: np.ndarray, entity_features: np.ndarray, gold_stats: RunningMeanStd) -> None:
    """Update the running gold estimate from real observations only (wilderness/
    do-nothing rows carry a fixed 0.0 placeholder in the gold column, not a real
    reading, so they'd bias the mean if included), then normalize gold in-place
    everywhere it appears - the agent's own scalar and every entity's gold feature,
    placeholders included (they get the same affine shift, which is fine since the
    type one-hot already tells the network whether a row is real).
    """
    is_real = (entity_features[:, 0] == 1.0) | (entity_features[:, 1] == 1.0)  # is_tribe or is_nation
    real_gold = entity_features[is_real, 5]
    own_gold = scalar[0:1]
    gold_stats.update(np.concatenate([own_gold, real_gold]) if real_gold.size else own_gold)

    scalar[0] = gold_stats.normalize(scalar[0])
    entity_features[:, 5] = gold_stats.normalize(entity_features[:, 5])


def _episode_state_from_setup(setup: dict, gold_stats: RunningMeanStd) -> EpisodeState:
    width, height = setup["width"], setup["height"]

    owner_grid = np.zeros((height, width), dtype=np.uint32)
    land_mask = np.zeros((height, width), dtype=bool)
    is_mine = np.zeros((height, width), dtype=bool)
    is_enemy = np.zeros((height, width), dtype=bool)
    player_stats = {}
    agent_small_id = setup["agentSmallID"]
    apply_tile_updates(owner_grid, land_mask, width, setup["packedTileUpdates"], agent_small_id, is_mine, is_enemy)
    apply_land_bitmap(land_mask, width, setup["landBitmap"])
    apply_player_updates(player_stats, setup["packedPlayerUpdates"])
    apply_max_troops(player_stats, setup["maxTroops"])

    player_types = {int(sid): t for sid, t in setup["playerTypes"].items()}
    total_land = setup["totalLand"]
    spatial, scalar = encode_observation(land_mask, is_mine, is_enemy, player_stats, agent_small_id, total_land)
    entity_features, targets = build_entities([], player_stats, player_types, total_land)
    _apply_gold_stats(scalar, entity_features, gold_stats)
    entities, entity_mask = pad_entities(entity_features)

    return EpisodeState(
        width=width,
        height=height,
        owner_grid=owner_grid,
        land_mask=land_mask,
        is_mine=is_mine,
        is_enemy=is_enemy,
        player_stats=player_stats,
        agent_client_id=setup["agentClientID"],
        agent_small_id=agent_small_id,
        agent_name=setup["agentName"],
        player_types=player_types,
        spatial=spatial,
        scalar=scalar,
        entities=entities,
        entity_mask=entity_mask,
        targets=targets,
    )


def start_episode(bridge, gold_stats: RunningMeanStd) -> EpisodeState:
    """Used for standalone scripts and per-slot resets in run_update."""
    setup = bridge.reset(
        seed=str(random.randint(0, 2 ** 31 - 1)),
        map="mid_plains",
        bots=BOTS,
        nations=NATIONS,
        difficulty=DIFFICULTY,
    )
    return _episode_state_from_setup(setup, gold_stats)


def start_all_episodes(vec_bridge: VecOpenFrontBridge, gold_stats: RunningMeanStd) -> list[EpisodeState]:
    """Each slot needs its own random seed - sharing one would clone the same
    deterministic game across every env."""
    return list(vec_bridge.executor.map(lambda b: start_episode(b, gold_stats), vec_bridge.bridges))


def run_update(
    vec_bridge: VecOpenFrontBridge,
    agent,
    optimizer,
    episodes: list[EpisodeState],
    update: int,
    gold_stats: RunningMeanStd,
) -> tuple[list[EpisodeState], int, int]:
    """One STEPS-length rollout across all NUM_ENVS envs, GAE per env, one PPO
    update over the flattened STEPS*NUM_ENVS batch. Rollout buffers stay on
    CPU - too large for VRAM at STEPS=2000 - only per-tick and per-minibatch
    slices move to DEVICE.
    """
    rollout_start = time.perf_counter()

    n = len(episodes)
    # bool, not float32 - spatial planes are binary (is_land/is_mine/is_enemy) and this
    # buffer is STEPS*n*3*H*W, the single biggest allocation in the whole rollout by
    # far; 4x smaller here is the difference between fitting in RAM and not on bigger
    # maps. Cast back to float only for the per-minibatch GPU slice that needs it.
    spatial_buf = np.zeros((STEPS, n, *episodes[0].spatial.shape), dtype=np.bool)
    scalar_buf = np.zeros((STEPS, n, *episodes[0].scalar.shape), dtype=np.float32)
    entities_buf = np.zeros((STEPS, n, *episodes[0].entities.shape), dtype=np.float32)
    entity_masks = np.zeros((STEPS, n, *episodes[0].entity_mask.shape), dtype=np.bool)
    actions = np.zeros((STEPS, n))
    log_probs = np.zeros((STEPS, n))
    values = np.zeros((STEPS, n))
    rewards = np.zeros((STEPS, n))
    dones = np.zeros((STEPS, n), dtype=np.bool)
    wins = np.zeros((STEPS, n), dtype=np.bool)
    episode_end_golds = []

    for i in range(STEPS):
        spatial_batch = np.stack([e.spatial for e in episodes])
        scalar_batch = np.stack([e.scalar for e in episodes])
        entity_batch = np.stack([e.entities for e in episodes])
        entity_mask_batch = np.stack([e.entity_mask for e in episodes])

        with torch.no_grad():
            action, log_prob, entropy, value = agent.get_action_and_value(
                spatial=torch.from_numpy(spatial_batch).to(DEVICE),
                scalar=torch.from_numpy(scalar_batch).to(DEVICE),
                entities=torch.from_numpy(entity_batch).to(DEVICE),
                entity_mask=torch.from_numpy(entity_mask_batch).to(DEVICE),
                action=None
            )
        actions_np = action.cpu().numpy()
        log_probs_np = log_prob.cpu().numpy()
        values_np = value.squeeze(-1).cpu().numpy()

        intents_list = [
            action_to_intents(
                target=episodes[k].targets[actions_np[k]] if actions_np[k] < len(episodes[k].targets) else DO_NOTHING,
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
            entities_buf[i, k] = episode.entities
            entity_masks[i, k] = episode.entity_mask
            actions[i, k] = actions_np[k]
            log_probs[i, k] = log_probs_np[k]
            values[i, k] = values_np[k]

            apply_player_updates(episode.player_stats, player_updates)
            apply_max_troops(episode.player_stats, step_result["maxTroops"])
            apply_tile_updates(
                episode.owner_grid, episode.land_mask, episode.width, tile_updates,
                episode.agent_small_id, episode.is_mine, episode.is_enemy
            )

            curr_stats = episode.player_stats.get(episode.agent_small_id, {})
            # the wider FFA game runs long after our agent is wiped out, so elimination
            # counts as its own loss rather than waiting for someone else to win
            eliminated = curr_stats.get("tilesOwned", 0.0) == 0.0
            done = step_result["done"] or eliminated
            dones[i, k] = done
            won = (not eliminated) and step_result["winner"] == episode.agent_name
            wins[i, k] = won

            total_land = step_result["totalLand"]
            reward = compute_reward(prev_stats, curr_stats, done, won, total_land, GAMMA)
            rewards[i, k] = reward

            episode.spatial, episode.scalar = encode_observation(
                episode.land_mask, episode.is_mine, episode.is_enemy, episode.player_stats,
                episode.agent_small_id, total_land
            )
            entity_features, episode.targets = build_entities(
                attack_mask, episode.player_stats, episode.player_types, total_land
            )
            _apply_gold_stats(episode.scalar, entity_features, gold_stats)
            episode.entities, episode.entity_mask = pad_entities(entity_features)

            # reset just this slot - the others keep ticking independently
            if done:
                # gold at the moment a game ends - the action space has no way to
                # spend it yet, so this tracks how much is piling up unused
                episode_end_golds.append(curr_stats.get("gold", 0.0))
                episodes[k] = start_episode(vec_bridge.bridges[k], gold_stats)

    with torch.no_grad():
        last_spatial = np.stack([e.spatial for e in episodes])
        last_scalar = np.stack([e.scalar for e in episodes])
        last_entities = np.stack([e.entities for e in episodes])
        last_entity_mask = np.stack([e.entity_mask for e in episodes])
        last_value = agent.get_value(
            spatial=torch.from_numpy(last_spatial).to(DEVICE),
            scalar=torch.from_numpy(last_scalar).to(DEVICE),
            entities=torch.from_numpy(last_entities).to(DEVICE),
            entity_mask=torch.from_numpy(last_entity_mask).to(DEVICE),
        ).squeeze(-1).cpu()

    spatial_buf = torch.from_numpy(spatial_buf)
    scalar_buf = torch.from_numpy(scalar_buf)
    entities_buf = torch.from_numpy(entities_buf)
    entity_masks = torch.from_numpy(entity_masks)
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
    entities_buf = entities_buf.reshape(total, *entities_buf.shape[2:])
    entity_masks = entity_masks.reshape(total, *entity_masks.shape[2:])
    actions = actions.reshape(total)
    log_probs = log_probs.reshape(total)
    advantages = advantages.reshape(total)
    returns = returns.reshape(total)
    values = values.reshape(total)

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
                spatial=spatial_buf[batch_idx].to(DEVICE).float(),
                scalar=scalar_buf[batch_idx].to(DEVICE),
                entities=entities_buf[batch_idx].to(DEVICE),
                entity_mask=entity_masks[batch_idx].to(DEVICE),
                action=actions[batch_idx].to(DEVICE)
            )

            batch_log_probs = log_probs[batch_idx].to(DEVICE)
            batch_advantages = advantages[batch_idx].to(DEVICE)
            batch_returns = returns[batch_idx].to(DEVICE)
            batch_old_values = values[batch_idx].to(DEVICE)

            ratio = torch.exp(new_log_prob-batch_log_probs)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advantages

            policy_loss = -torch.min(unclipped, clipped).mean()

            # same clip-and-take-worse-case idea as the policy ratio above, applied to
            # the value regression - caps how far one epoch can drag the value estimate
            # for a given state from what it was at rollout time, instead of letting a
            # rare high-variance return (e.g. a just-completed win/loss) yank it freely
            new_value = new_value.squeeze(-1)
            value_clipped = batch_old_values + torch.clamp(
                new_value - batch_old_values, -CLIP_EPS, CLIP_EPS
            )
            value_loss_unclipped = torch.square(new_value - batch_returns)
            value_loss_clipped = torch.square(value_clipped - batch_returns)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

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
    mean_gold_at_done = sum(episode_end_golds) / len(episode_end_golds) if episode_end_golds else 0.0

    ppo_elapsed = time.perf_counter() - ppo_start
    total_elapsed = rollout_elapsed + ppo_elapsed
    ticks_per_sec = (STEPS * n) / rollout_elapsed

    print(
        f"update {update}: mean_reward={rewards.mean().item():.3f} "
        f"episodes_finished={episodes_finished} wins={episode_wins} "
        f"mean_gold_at_done={mean_gold_at_done:,.0f} "
        f"policy_loss={policy_loss_sum / num_minibatches:.4f} "
        f"value_loss={value_loss_sum / num_minibatches:.4f} "
        f"entropy={entropy_sum / num_minibatches:.4f} "
        f"time={total_elapsed:.1f}s (rollout={rollout_elapsed:.1f}s ppo={ppo_elapsed:.1f}s) "
        f"ticks/sec={ticks_per_sec:.1f}"
    )

    return episodes, episodes_finished, episode_wins


def train():
    gold_stats = RunningMeanStd()
    # Load (but don't apply to an agent yet) before building any episodes - the
    # first observations need to be normalized with the resumed stats, not a fresh
    # RunningMeanStd(), or a resumed run's initial states are miscalibrated relative
    # to the weights that are about to be loaded into the same agent.
    checkpoint_path = latest_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE) if checkpoint_path is not None else None
    if checkpoint is not None:
        gold_stats.mean = checkpoint["gold_mean"]
        gold_stats.var = checkpoint["gold_var"]
        gold_stats.count = checkpoint["gold_count"]

    with VecOpenFrontBridge(NUM_ENVS) as vec_bridge:
        episodes = start_all_episodes(vec_bridge, gold_stats)

        agent = Agent.from_observation(episodes[0].spatial, episodes[0].scalar, episodes[0].entities).to(DEVICE)
        if checkpoint is not None:
            # strict=False so an architecture change (e.g. adding gate_head) warm-starts
            # from whatever weights still match - backbone/entity encoder/value_head/
            # query_proj carry over, only the genuinely new layers start randomly
            # initialized, instead of throwing away everything already learned.
            missing, unexpected = agent.load_state_dict(checkpoint["model"], strict=False)
            if missing or unexpected:
                print(f"partial resume from {checkpoint_path}: missing={missing} unexpected={unexpected}")
            else:
                print(f"resumed from {checkpoint_path}")
        optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
        print(f"training on device={DEVICE}, num_envs={NUM_ENVS}")

        total_episodes = 0
        total_wins = 0

        for update in range(NUM_UPDATES):
            episodes, episodes_finished, episode_wins = run_update(
                vec_bridge, agent, optimizer, episodes, update, gold_stats
            )
            total_episodes += episodes_finished
            total_wins += episode_wins

            if update != 0 and update % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(agent, update, gold_stats)

        save_checkpoint(agent, NUM_UPDATES, gold_stats)

        win_rate = total_wins / total_episodes if total_episodes > 0 else 0.0
        print(
            f"training done: {NUM_UPDATES} updates, {total_episodes} episodes played, "
            f"{total_wins} wins ({win_rate:.1%} win rate)"
        )


if __name__ == "__main__": train()
