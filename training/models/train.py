from dataclasses import dataclass
from pathlib import Path

import json
import os
import random
import time
import numpy as np
import torch.optim

from training.envs.vec_bridge import VecOpenFrontBridge
from training.envs.game_state import EpisodeState, new_masks, apply_reset, apply_step
from training.envs.observation import encode_observation, reconstruct_spatial_torch, MAX_SMALL_ID
from training.envs.action_space import build_entities, pad_entities, action_to_intents, DO_NOTHING
from training.envs.reward import compute_reward
from training.envs.running_stats import RunningMeanStd
from training.models.policy import Agent
from training.models.anchor_buffer import AnchorBuffer

# Rollout length. Sized so most games finish inside one rollout; NUM_ENVS is bounded
# by RAM (the buffers store every tick).
STEPS = 6500
NUM_UPDATES = 10000
NUM_ENVS = int(os.environ.get("OFM_NUM_ENVS", "10"))
MAX_EPISODE_TICKS = 10000  # episodes still running at this tick count are ended as a loss

BOTS = 0
NATIONS = int(os.environ.get("OFM_NATIONS", "5"))
# One difficulty is drawn per episode; repeat a name to weight it, e.g. "Medium,Medium,Hard".
DIFFICULTIES = os.environ.get("OFM_DIFFICULTY", "Easy").split(",")
DIFFICULTY = DIFFICULTIES[0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS_DIR = Path(os.environ.get("OFM_RUNS_DIR", str(Path(__file__).parent.parent.parent / "runs")))
RUN_NAME = os.environ.get("OFM_RUN_NAME", "default")
CHECKPOINT_DIR = Path(os.environ.get("OFM_CHECKPOINT_DIR", str(RUNS_DIR / "checkpoints" / RUN_NAME)))
CHECKPOINT_INTERVAL = 16
CHECKPOINT_KEEP = 200
# Outside CHECKPOINT_DIR so clearing checkpoints doesn't reset the training-hours counter.
TRAIN_STATE_PATH = Path(os.environ.get("OFM_TRAIN_STATE_PATH", str(RUNS_DIR / "state" / f"{RUN_NAME}.json")))

LR = 1e-4
LR_WARMUP_UPDATES = 16  # ramp after any fresh optimizer (no saved Adam state)
GAMMA = 1 - 1 / STEPS
GAE_LAMBDA = 0.95
PPO_EPOCHS = 2
MINIBATCH_SIZE = 256
CLIP_EPS = 0.2
VALUE_COEF = 1.5
ENTROPY_COEF = float(os.environ.get("OFM_ENTROPY_COEF", "0.08"))
GATE_KL_COEF = float(os.environ.get("OFM_GATE_KL", "0.08"))
TARGET_TAU = 0.03  # EMA rate of the target network used for GAE's value estimates

# Anchor buffer: self-imitation on high-troop-ratio states that led to good returns.
HIGH_RATIO_THRESHOLD = 0.35
HIGH_RATIO_CEILING = 0.60
ANCHOR_BUFFER_CAPACITY = 500
ANCHOR_MAX_INSERT_PER_UPDATE = 32
ANCHOR_MAX_AGE_UPDATES = 300
ANCHOR_MIN_FILL = 64
ANCHOR_MINIBATCH_SIZE = 64
ANCHOR_COEF = 0.02

# Low-ratio buffer: keeps the value head calibrated on depleted states.
LOW_RATIO_THRESHOLD = 0.15
LOW_RATIO_BUFFER_CAPACITY = 500
LOW_RATIO_MAX_INSERT_PER_UPDATE = 32
LOW_RATIO_MAX_AGE_UPDATES = 300
LOW_RATIO_MIN_FILL = 64
LOW_RATIO_MINIBATCH_SIZE = 64
LOW_RATIO_COEF = 0.02


def load_total_train_hours() -> float:
    if TRAIN_STATE_PATH.exists():
        return json.loads(TRAIN_STATE_PATH.read_text()).get("total_train_hours", 0.0)
    return 0.0


def save_total_train_hours(total_train_hours: float) -> None:
    TRAIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_STATE_PATH.write_text(json.dumps({"total_train_hours": total_train_hours}))


def save_checkpoint(
    agent, update: int, gold_stats: RunningMeanStd, return_stats: RunningMeanStd, total_train_hours: float,
    optimizer: torch.optim.Optimizer,
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"agent_{time.strftime('%Y%m%d_%H%M%S')}_update{update}.pt"
    torch.save(
        {
            "model": agent.state_dict(),
            "gold_mean": gold_stats.mean,
            "gold_var": gold_stats.var,
            "gold_count": gold_stats.count,
            "return_mean": return_stats.mean,
            "return_var": return_stats.var,
            "return_count": return_stats.count,
            "total_train_hours": total_train_hours,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    for stale in sorted(CHECKPOINT_DIR.glob("agent_*.pt"))[:-CHECKPOINT_KEEP]:
        stale.unlink()
    return path


def latest_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("agent_*.pt")) if checkpoint_dir.exists() else []
    return checkpoints[-1] if checkpoints else None


@dataclass
class Rollout:
    # The spatial planes are stored compressed and rebuilt per minibatch, see
    # observation.py's reconstruct_spatial_torch. static_planes (land, impassable) are
    # stored once, which assumes the map never changes mid-episode.
    bool_planes: np.ndarray
    static_planes: np.ndarray
    owner_grid: np.ndarray
    ratio_table: np.ndarray
    scalar: np.ndarray
    entities: np.ndarray
    entity_mask: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    wins: np.ndarray
    last_value: torch.Tensor
    num_envs: int
    episodes_finished: int
    episode_wins: int
    episode_end_golds_win: list[float]
    episode_end_golds_loss: list[float]
    wins_by_setup: dict[str, list[int]]
    end_structs_win: list[dict]
    end_structs_loss: list[dict]
    end_records: list[tuple]
    elapsed: float


def _apply_gold_stats(scalar: np.ndarray, entity_features: np.ndarray, gold_stats: RunningMeanStd) -> None:
    """Update running gold stats from real entities only, then normalize in place."""
    is_real = (entity_features[:, 0] == 1.0) | (entity_features[:, 1] == 1.0)  # tribe or nation
    real_gold = entity_features[is_real, 5]
    own_gold = scalar[0:1]
    gold_stats.update(np.concatenate([own_gold, real_gold]) if real_gold.size else own_gold)

    scalar[0] = gold_stats.normalize(scalar[0])
    entity_features[:, 5] = gold_stats.normalize(entity_features[:, 5])


def _episode_state_from_setup(setup: dict, gold_stats: RunningMeanStd) -> EpisodeState:
    width, height = setup["width"], setup["height"]
    owner_grid, land_mask, impassable_mask, is_mine, is_enemy = new_masks(width, height)
    player_stats, agent_small_id, player_types = apply_reset(
        owner_grid, land_mask, impassable_mask, is_mine, is_enemy, width, setup,
    )

    total_land = setup["totalLand"]
    spatial, scalar, ratio_table = encode_observation(land_mask, impassable_mask, is_mine, is_enemy, player_stats, agent_small_id, total_land, owner_grid)
    entity_features, targets = build_entities([], player_stats, player_types, total_land)
    _apply_gold_stats(scalar, entity_features, gold_stats)
    entities, entity_mask = pad_entities(entity_features)

    return EpisodeState(
        width=width,
        height=height,
        owner_grid=owner_grid,
        land_mask=land_mask,
        impassable_mask=impassable_mask,
        is_mine=is_mine,
        is_enemy=is_enemy,
        ratio_table=ratio_table,
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
    difficulty = random.choice(DIFFICULTIES)
    setup = bridge.reset(
        seed=str(random.randint(0, 2 ** 31 - 1)),
        map="mid_plains",
        bots=BOTS,
        nations=NATIONS,
        difficulty=difficulty,
    )
    episode = _episode_state_from_setup(setup, gold_stats)
    episode.difficulty = difficulty
    return episode


def start_all_episodes(vec_bridge: VecOpenFrontBridge, gold_stats: RunningMeanStd) -> list[EpisodeState]:
    return list(vec_bridge.executor.map(lambda b: start_episode(b, gold_stats), vec_bridge.bridges))


def compute_value_for_episodes(agent, episodes: list[EpisodeState]) -> torch.Tensor:
    """V(s) for each env's current state."""
    with torch.no_grad():
        return agent.get_value(
            spatial=torch.from_numpy(np.stack([e.spatial for e in episodes])).to(DEVICE),
            scalar=torch.from_numpy(np.stack([e.scalar for e in episodes])).to(DEVICE),
            entities=torch.from_numpy(np.stack([e.entities for e in episodes])).to(DEVICE),
            entity_mask=torch.from_numpy(np.stack([e.entity_mask for e in episodes])).to(DEVICE),
        ).squeeze(-1).cpu()


def collect_rollout(
    vec_bridge: VecOpenFrontBridge,
    agent,
    episodes: list[EpisodeState],
    gold_stats: RunningMeanStd,
) -> tuple[list[EpisodeState], Rollout]:
    rollout_start = time.perf_counter()

    num_envs = len(episodes)
    height, width = episodes[0].land_mask.shape
    bool_planes_buf = np.zeros((STEPS, num_envs, 2, height, width), dtype=bool)  # is_mine, is_enemy
    static_planes = np.stack([episodes[0].land_mask, episodes[0].impassable_mask]).copy()
    owner_grid_buf = np.zeros((STEPS, num_envs, height, width), dtype=np.int16)
    ratio_table_buf = np.zeros((STEPS, num_envs, MAX_SMALL_ID), dtype=np.float16)
    scalar_buf = np.zeros((STEPS, num_envs, *episodes[0].scalar.shape), dtype=np.float32)
    entities_buf = np.zeros((STEPS, num_envs, *episodes[0].entities.shape), dtype=np.float32)
    entity_masks = np.zeros((STEPS, num_envs, *episodes[0].entity_mask.shape), dtype=np.bool)
    actions = np.zeros((STEPS, num_envs))
    log_probs = np.zeros((STEPS, num_envs))
    values = np.zeros((STEPS, num_envs))
    rewards = np.zeros((STEPS, num_envs))
    dones = np.zeros((STEPS, num_envs), dtype=np.bool)
    wins = np.zeros((STEPS, num_envs), dtype=np.bool)
    episode_end_golds_win = []
    episode_end_golds_loss = []
    wins_by_setup: dict[str, list[int]] = {}
    end_structs_win: list[dict] = []
    end_structs_loss: list[dict] = []
    end_records: list[tuple] = []

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

        # action = target index + attack size as a fraction of current troops
        tgt_idx = np.floor(actions_np).astype(np.int64)
        size_frac = actions_np - tgt_idx
        intents_list = [
            action_to_intents(
                target=episodes[k].targets[tgt_idx[k]] if tgt_idx[k] < len(episodes[k].targets) else DO_NOTHING,
                agent_client_id=episodes[k].agent_client_id,
                troops=size_frac[k] * episodes[k].player_stats.get(episodes[k].agent_small_id, {}).get("troops", 0.0),
            )
            for k in range(num_envs)
        ]
        step_results = vec_bridge.step(intents_list)

        for k in range(num_envs):
            episode = episodes[k]
            step_result = step_results[k]

            bool_planes_buf[i, k] = np.stack([episode.is_mine, episode.is_enemy])
            owner_grid_buf[i, k] = episode.owner_grid
            ratio_table_buf[i, k] = episode.ratio_table
            scalar_buf[i, k] = episode.scalar
            entities_buf[i, k] = episode.entities
            entity_masks[i, k] = episode.entity_mask
            actions[i, k] = actions_np[k]
            log_probs[i, k] = log_probs_np[k]
            values[i, k] = values_np[k]

            entity_features, prev_stats, curr_stats, done, won = apply_step(episode, step_result)
            episode.ticks += 1
            episode.peak = max(episode.peak, float(episode.scalar[2]))
            capped = False
            if not done and episode.ticks >= MAX_EPISODE_TICKS:
                print(f"[episode-cap] env {k} hit MAX_EPISODE_TICKS={MAX_EPISODE_TICKS}")
                done, won, capped = True, False, True
            dones[i, k] = done
            wins[i, k] = won
            rewards[i, k] = compute_reward(prev_stats, curr_stats, done, won, step_result["totalLand"], GAMMA)

            _apply_gold_stats(episode.scalar, entity_features, gold_stats)
            episode.entities, episode.entity_mask = pad_entities(entity_features)

            if done:
                # an eliminated player's gold reads 0 at the end, so use the previous tick for losses
                end_gold = curr_stats.get("gold", 0.0) if won else prev_stats.get("gold", curr_stats.get("gold", 0.0))
                (episode_end_golds_win if won else episode_end_golds_loss).append(end_gold)
                kind = "win" if won else "cap" if capped else "other" if step_result.get("winner") else "elim"
                end_records.append((kind, episode.ticks, episode.peak))
                structs = step_result.get("structures")
                if structs:
                    nation_ids = [sid for sid, t in episode.player_types.items() if t == "NATION"]
                    agg: dict = {}
                    for sid in nation_ids:
                        for name, c in structs.get(str(sid), {}).items():
                            agg[name] = agg.get(name, 0) + c
                    n_nat = max(len(nation_ids), 1)
                    (end_structs_win if won else end_structs_loss).append({name: c / n_nat for name, c in agg.items()})
                num_nations = sum(1 for t in episode.player_types.values() if t == "NATION")
                bucket = wins_by_setup.setdefault(f"{num_nations}{episode.difficulty[:1]}", [0, 0])
                bucket[1] += 1
                if won:
                    bucket[0] += 1
                episodes[k] = start_episode(vec_bridge.bridges[k], gold_stats)

    # an env that never finished is bootstrapped from last_value instead of a terminal reward
    for k in range(num_envs):
        if not dones[:, k].any():
            print(f"[game-length] env {k} ran the full {STEPS}-tick rollout without finishing (ticks={episodes[k].ticks})")

    rollout = Rollout(
        bool_planes=bool_planes_buf,
        static_planes=static_planes,
        owner_grid=owner_grid_buf,
        ratio_table=ratio_table_buf,
        scalar=scalar_buf,
        entities=entities_buf,
        entity_mask=entity_masks,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        wins=wins,
        last_value=compute_value_for_episodes(agent, episodes),
        num_envs=num_envs,
        episodes_finished=int(dones.sum().item()),
        episode_wins=int(wins.sum().item()),
        episode_end_golds_win=episode_end_golds_win,
        episode_end_golds_loss=episode_end_golds_loss,
        wins_by_setup=wins_by_setup,
        end_structs_win=end_structs_win,
        end_structs_loss=end_structs_loss,
        end_records=end_records,
        elapsed=time.perf_counter() - rollout_start,
    )
    return episodes, rollout


def _ends_str(records: list[tuple]) -> str:
    out = []
    for kind in ("win", "elim", "other", "cap"):
        r = [x for x in records if x[0] == kind]
        if r:
            ticks = sorted(x[1] for x in r)
            out.append(f"{kind}:n={len(r)},t_med={ticks[len(ticks) // 2]},peak={sum(x[2] for x in r) / len(r):.2f}")
    return ";".join(out) or "-"


def _structs_str(rows: list[dict]) -> str:
    if not rows:
        return "-"
    return ",".join(f"{k}={sum(r.get(k, 0) for r in rows) / len(rows):.2f}" for k in rows[0])


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def ppo_update(
    rollout: Rollout,
    agent,
    target_agent,
    target_last_value: torch.Tensor,
    optimizer,
    update: int,
    total_train_hours: float,
    return_stats: RunningMeanStd,
    anchor_buffer: AnchorBuffer,
    low_ratio_buffer: AnchorBuffer,
) -> float:
    params_before = torch.cat([p.detach().flatten() for p in agent.parameters()])

    bool_planes_buf = torch.from_numpy(rollout.bool_planes)
    static_planes = torch.from_numpy(rollout.static_planes).to(DEVICE)
    owner_grid_buf = torch.from_numpy(rollout.owner_grid)
    ratio_table_buf = torch.from_numpy(rollout.ratio_table)
    scalar_buf = torch.from_numpy(rollout.scalar)
    entities_buf = torch.from_numpy(rollout.entities)
    entity_masks = torch.from_numpy(rollout.entity_mask)
    actions = torch.from_numpy(rollout.actions)
    log_probs = torch.from_numpy(rollout.log_probs)
    values = torch.from_numpy(rollout.values)
    rewards = torch.from_numpy(rollout.rewards)
    dones = torch.from_numpy(rollout.dones)

    # GAE uses the slow-moving target network's values, in chunks to bound memory
    total = STEPS * rollout.num_envs
    flat_bool = bool_planes_buf.reshape(total, *bool_planes_buf.shape[2:])
    flat_owner = owner_grid_buf.reshape(total, *owner_grid_buf.shape[2:])
    flat_ratio = ratio_table_buf.reshape(total, *ratio_table_buf.shape[2:])
    flat_scalar = scalar_buf.reshape(total, *scalar_buf.shape[2:])
    flat_entities = entities_buf.reshape(total, *entities_buf.shape[2:])
    flat_entity_mask = entity_masks.reshape(total, *entity_masks.shape[2:])
    flat_target_values = torch.zeros(total)
    with torch.no_grad():
        for start in range(0, total, MINIBATCH_SIZE):
            idx = slice(start, start + MINIBATCH_SIZE)
            chunk_scalar = flat_scalar[idx].to(DEVICE)
            chunk_spatial = reconstruct_spatial_torch(
                flat_bool[idx].to(DEVICE), flat_owner[idx].to(DEVICE).long(),
                flat_ratio[idx].to(DEVICE).float(), chunk_scalar[:, 1], static_planes,
            )
            chunk_value = target_agent.get_value(
                spatial=chunk_spatial, scalar=chunk_scalar,
                entities=flat_entities[idx].to(DEVICE), entity_mask=flat_entity_mask[idx].to(DEVICE),
            )
            flat_target_values[idx] = chunk_value.squeeze(-1).cpu()
    target_values = flat_target_values.reshape(STEPS, rollout.num_envs)

    running_advantage = torch.zeros(rollout.num_envs)
    next_value = target_last_value
    advantages = torch.zeros(STEPS, rollout.num_envs)
    returns = torch.zeros(STEPS, rollout.num_envs)
    for t in reversed(range(STEPS)):
        non_term = 1 - dones[t].float()
        delta = rewards[t] + (GAMMA * next_value * non_term) - target_values[t]
        running_advantage = delta + (GAMMA * GAE_LAMBDA * non_term * running_advantage)
        next_value = target_values[t]
        advantages[t] = running_advantage
        returns[t] = advantages[t] + target_values[t]

    frac_low_ratio = (scalar_buf.reshape(total, -1)[:, 1] < LOW_RATIO_THRESHOLD).float().mean().item()
    raw_advantage_mean = advantages.mean().item()
    raw_advantage_std = advantages.std().item()
    raw_advantages_np = advantages.numpy().reshape(-1).copy()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # value errors are scaled by a running return std; rare wins have much larger returns than losses
    return_stats.update(returns.numpy().flatten())
    return_std = (return_stats.var + 1e-8) ** 0.5

    bool_planes_buf = flat_bool
    owner_grid_buf = flat_owner
    ratio_table_buf = flat_ratio
    scalar_buf = flat_scalar
    entities_buf = flat_entities
    entity_masks = flat_entity_mask
    actions = actions.reshape(total)
    log_probs = log_probs.reshape(total)
    advantages = advantages.reshape(total)
    returns = returns.reshape(total)
    values = values.reshape(total)

    anchor_buffer.evict_older_than(update, ANCHOR_MAX_AGE_UPDATES)
    anchor_inserted = anchor_buffer.insert_from_rollout(
        bool_planes_buf.numpy(), owner_grid_buf.numpy(), ratio_table_buf.numpy(),
        scalar_buf.numpy(), entities_buf.numpy(), entity_masks.numpy(),
        actions.numpy(), raw_advantages_np, returns.numpy(),
        HIGH_RATIO_THRESHOLD, update, ANCHOR_MAX_INSERT_PER_UPDATE,
        high_ratio_ceiling=HIGH_RATIO_CEILING,
    )

    low_ratio_buffer.evict_older_than(update, LOW_RATIO_MAX_AGE_UPDATES)
    low_ratio_inserted = low_ratio_buffer.insert_low_ratio_from_rollout(
        bool_planes_buf.numpy(), owner_grid_buf.numpy(), ratio_table_buf.numpy(),
        scalar_buf.numpy(), entities_buf.numpy(), entity_masks.numpy(),
        actions.numpy(), returns.numpy(),
        LOW_RATIO_THRESHOLD, update, LOW_RATIO_MAX_INSERT_PER_UPDATE,
    )

    rollout_elapsed = rollout.elapsed
    ppo_start = time.perf_counter()

    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    gate_kl_sum = 0.0
    num_minibatches = 0
    approx_kl = 0.0
    # value error by troop-ratio bucket, measured on the first epoch only
    low_ratio_err_sum, low_ratio_err_count = 0.0, 0
    normal_err_sum, normal_err_count = 0.0, 0
    anchor_loss_sum, anchor_steps = 0.0, 0
    low_ratio_loss_sum, low_ratio_steps = 0.0, 0

    for epoch in range(PPO_EPOCHS):
        indices = torch.randperm(total)
        for start in range(0, total, MINIBATCH_SIZE):
            batch_idx = indices[start:start + MINIBATCH_SIZE]

            batch_scalar = scalar_buf[batch_idx].to(DEVICE)
            batch_spatial = reconstruct_spatial_torch(
                bool_planes_buf[batch_idx].to(DEVICE), owner_grid_buf[batch_idx].to(DEVICE).long(),
                ratio_table_buf[batch_idx].to(DEVICE).float(), batch_scalar[:, 1], static_planes,
            )
            _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                spatial=batch_spatial,
                scalar=batch_scalar,
                entities=entities_buf[batch_idx].to(DEVICE),
                entity_mask=entity_masks[batch_idx].to(DEVICE),
                action=actions[batch_idx].to(DEVICE)
            )

            batch_log_probs = log_probs[batch_idx].to(DEVICE)
            batch_advantages = advantages[batch_idx].to(DEVICE)
            batch_returns = returns[batch_idx].to(DEVICE)
            batch_old_values = values[batch_idx].to(DEVICE)

            ratio = torch.exp(new_log_prob - batch_log_probs)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()  # logged, not acted on

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            new_value = new_value.squeeze(-1)
            value_loss_unclipped = torch.square((new_value - batch_returns) / return_std)
            value_clipped = batch_old_values + torch.clamp(new_value - batch_old_values, -CLIP_EPS, CLIP_EPS)
            value_loss_clipped = torch.square((value_clipped - batch_returns) / return_std)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            if epoch == 0:
                with torch.no_grad():
                    abs_err = (new_value - batch_returns).abs() / return_std
                    low_mask = batch_scalar[:, 1] < LOW_RATIO_THRESHOLD
                    if low_mask.any():
                        low_ratio_err_sum += abs_err[low_mask].sum().item()
                        low_ratio_err_count += int(low_mask.sum().item())
                    if (~low_mask).any():
                        normal_err_sum += abs_err[~low_mask].sum().item()
                        normal_err_count += int((~low_mask).sum().item())

            entropy_bonus = entropy.mean()
            gate_kl_mean = agent.gate_kl.mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_bonus + GATE_KL_COEF * gate_kl_mean
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy_bonus.item()
            gate_kl_sum += gate_kl_mean.item()
            num_minibatches += 1

        # once per epoch: self-imitation step on the anchor buffer
        anchor_batch = anchor_buffer.sample(ANCHOR_MINIBATCH_SIZE) if len(anchor_buffer) >= ANCHOR_MIN_FILL else None
        if anchor_batch is not None:
            a_scalar = torch.from_numpy(anchor_batch["scalar"]).to(DEVICE)
            a_spatial = reconstruct_spatial_torch(
                torch.from_numpy(anchor_batch["bool_planes"]).to(DEVICE),
                torch.from_numpy(anchor_batch["owner_grid"]).to(DEVICE).long(),
                torch.from_numpy(anchor_batch["ratio_table"]).to(DEVICE).float(),
                a_scalar[:, 1], static_planes,
            )
            a_return = torch.from_numpy(anchor_batch["ret"]).to(DEVICE)
            _, a_log_prob, _, a_value = agent.get_action_and_value(
                spatial=a_spatial, scalar=a_scalar,
                entities=torch.from_numpy(anchor_batch["entities"]).to(DEVICE),
                entity_mask=torch.from_numpy(anchor_batch["entity_mask"]).to(DEVICE),
                action=torch.from_numpy(anchor_batch["action"]).to(DEVICE),
            )
            a_value = a_value.squeeze(-1)
            with torch.no_grad():
                weight = torch.clamp((a_return - a_value) / return_std, min=0.0)
            policy_term = -(weight * a_log_prob).mean()
            # only pushes the value up toward a demonstrated return, never down
            value_term = 0.5 * torch.clamp((a_return - a_value) / return_std, min=0.0).square().mean()
            anchor_loss = ANCHOR_COEF * (policy_term + VALUE_COEF * value_term)

            optimizer.zero_grad()
            anchor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            anchor_loss_sum += anchor_loss.item()
            anchor_steps += 1

        low_ratio_batch = low_ratio_buffer.sample(LOW_RATIO_MINIBATCH_SIZE) if len(low_ratio_buffer) >= LOW_RATIO_MIN_FILL else None
        if low_ratio_batch is not None:
            lr_scalar = torch.from_numpy(low_ratio_batch["scalar"]).to(DEVICE)
            lr_spatial = reconstruct_spatial_torch(
                torch.from_numpy(low_ratio_batch["bool_planes"]).to(DEVICE),
                torch.from_numpy(low_ratio_batch["owner_grid"]).to(DEVICE).long(),
                torch.from_numpy(low_ratio_batch["ratio_table"]).to(DEVICE).float(),
                lr_scalar[:, 1], static_planes,
            )
            lr_return = torch.from_numpy(low_ratio_batch["ret"]).to(DEVICE)
            lr_value = agent.get_value(
                spatial=lr_spatial, scalar=lr_scalar,
                entities=torch.from_numpy(low_ratio_batch["entities"]).to(DEVICE),
                entity_mask=torch.from_numpy(low_ratio_batch["entity_mask"]).to(DEVICE),
            ).squeeze(-1)
            low_ratio_loss = LOW_RATIO_COEF * VALUE_COEF * 0.5 * (((lr_return - lr_value) / return_std) ** 2).mean()

            optimizer.zero_grad()
            low_ratio_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            low_ratio_loss_sum += low_ratio_loss.item()
            low_ratio_steps += 1

    params_after = torch.cat([p.detach().flatten() for p in agent.parameters()])
    weight_delta_l2 = (params_after - params_before).norm().item()

    ppo_elapsed = time.perf_counter() - ppo_start
    total_train_hours += (rollout_elapsed + ppo_elapsed) / 3600
    num_minibatches = max(num_minibatches, 1)

    setups = " ".join(f"n{n}={w}/{t}" for n, (w, t) in sorted(rollout.wins_by_setup.items()))
    print(
        f"update {update}: mean_reward={rewards.mean().item():.3f} "
        f"episodes_finished={rollout.episodes_finished} wins={rollout.episode_wins} "
        f"[{setups}] "
        f"ends({_ends_str(rollout.end_records)}) "
        f"nation_structs_win({_structs_str(rollout.end_structs_win)}) nation_structs_loss({_structs_str(rollout.end_structs_loss)}) "
        f"mean_gold_at_win={_mean(rollout.episode_end_golds_win):,.0f} mean_gold_at_loss={_mean(rollout.episode_end_golds_loss):,.0f} "
        f"policy_loss={policy_loss_sum / num_minibatches:.4f} "
        f"value_loss={value_loss_sum / num_minibatches:.4f} "
        f"entropy={entropy_sum / num_minibatches:.4f} gate_kl={gate_kl_sum / num_minibatches:.4f} "
        f"approx_kl={approx_kl:.4f} frac_low_ratio={frac_low_ratio:.4f} "
        f"value_err_low_ratio={low_ratio_err_sum / max(low_ratio_err_count, 1):.4f}(n={low_ratio_err_count}) "
        f"value_err_normal={normal_err_sum / max(normal_err_count, 1):.4f}(n={normal_err_count}) "
        f"anchor_buffer_size={len(anchor_buffer)} anchor_inserted={anchor_inserted} "
        f"anchor_loss={anchor_loss_sum / max(anchor_steps, 1):.4f} "
        f"low_ratio_buffer_size={len(low_ratio_buffer)} low_ratio_inserted={low_ratio_inserted} "
        f"low_ratio_loss={low_ratio_loss_sum / max(low_ratio_steps, 1):.4f} "
        f"raw_adv_mean={raw_advantage_mean:.4f} raw_adv_std={raw_advantage_std:.4f} "
        f"weight_delta_l2={weight_delta_l2:.4f} "
        f"time={rollout_elapsed + ppo_elapsed:.1f}s (rollout={rollout_elapsed:.1f}s ppo={ppo_elapsed:.1f}s) "
        f"ticks/sec={total / rollout_elapsed:.1f} total_train_hours={total_train_hours:.2f}"
    )
    return total_train_hours


def train():
    total_train_hours = max(load_total_train_hours(), 0.0)

    with VecOpenFrontBridge(NUM_ENVS) as vec_bridge:
        gold_stats = RunningMeanStd()
        return_stats = RunningMeanStd()
        checkpoint_path = latest_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE) if checkpoint_path is not None else None
        if checkpoint is not None:
            gold_stats.mean, gold_stats.var, gold_stats.count = (
                checkpoint["gold_mean"], checkpoint["gold_var"], checkpoint["gold_count"]
            )
            return_stats.mean, return_stats.var, return_stats.count = (
                checkpoint["return_mean"], checkpoint["return_var"], checkpoint["return_count"]
            )
            total_train_hours = max(total_train_hours, checkpoint.get("total_train_hours", 0.0))

        episodes = start_all_episodes(vec_bridge, gold_stats)
        agent = Agent.from_observation(episodes[0].spatial, episodes[0].scalar, episodes[0].entities).to(DEVICE)
        optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
        warmup_from_update = None  # set when the optimizer starts without saved state
        if checkpoint is not None:
            agent.load_state_dict(checkpoint["model"])
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except (KeyError, TypeError, ValueError, RuntimeError):
                warmup_from_update = 0
                print("optimizer state not restored, warming up the learning rate")
            print(f"resumed from {checkpoint_path}")
        else:
            warmup_from_update = 0

        # target network for GAE's value estimates, restarted as a copy on every launch
        target_agent = Agent.from_observation(episodes[0].spatial, episodes[0].scalar, episodes[0].entities).to(DEVICE)
        target_agent.load_state_dict(agent.state_dict())
        anchor_buffer = AnchorBuffer(ANCHOR_BUFFER_CAPACITY)
        low_ratio_buffer = AnchorBuffer(LOW_RATIO_BUFFER_CAPACITY)

        print(f"training on device={DEVICE}, num_envs={NUM_ENVS}, nations={NATIONS}, difficulty={DIFFICULTIES}")

        for update in range(NUM_UPDATES):
            episodes, rollout = collect_rollout(vec_bridge, agent, episodes, gold_stats)
            target_last_value = compute_value_for_episodes(target_agent, episodes)

            if warmup_from_update is not None:
                since = update - warmup_from_update
                warmup_frac = min((since + 1) / LR_WARMUP_UPDATES, 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = LR * warmup_frac
                if since + 1 >= LR_WARMUP_UPDATES:
                    warmup_from_update = None

            total_train_hours = ppo_update(
                rollout, agent, target_agent, target_last_value, optimizer, update, total_train_hours,
                return_stats, anchor_buffer, low_ratio_buffer,
            )
            target_agent.soft_update_from(agent, TARGET_TAU)
            save_total_train_hours(total_train_hours)
            del rollout  # frees the buffers before the next rollout allocates its own

            if update != 0 and update % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(agent, update, gold_stats, return_stats, total_train_hours, optimizer)

        save_checkpoint(agent, NUM_UPDATES, gold_stats, return_stats, total_train_hours, optimizer)


if __name__ == "__main__":
    train()
