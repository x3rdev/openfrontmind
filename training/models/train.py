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

# Rollout length per env. Games continue across updates; NUM_ENVS is bounded by RAM (the
# buffers store every tick).
STEPS = int(os.environ.get("OFM_STEPS", "6500"))
NUM_UPDATES = 10000
NUM_ENVS = int(os.environ.get("OFM_NUM_ENVS", "10"))
MAX_EPISODE_TICKS = 30000  # episodes still running here end; the highest tile share wins
CAP_FLAG_FRAC = 0.05  # warn when more than this share of an update's episodes hit the cap

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

LR = float(os.environ.get("OFM_LR", "1e-4"))
LR_WARMUP_UPDATES = 16  # ramp after any fresh optimizer (no saved Adam state)
GAMMA = 1 - 1 / 6500  # per-tick discount horizon, independent of STEPS
ACT_EVERY = max(1, int(os.environ.get("OFM_ACT_EVERY", "1")))
GAMMA_STEP = GAMMA ** ACT_EVERY
GAE_LAMBDA = float(os.environ.get("OFM_GAE_LAMBDA", "0.95"))
PPO_EPOCHS = 2
MINIBATCH_SIZE = 256
CLIP_EPS = 0.2
VALUE_COEF = 1.5
GATE_KL_COEF = float(os.environ.get("OFM_GATE_KL", "0.08"))
ENTROPY_COEF = float(os.environ.get("OFM_ENTROPY_COEF", "0"))


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
    # stored once, which assumes the map never changes mid-episode. is_mine/is_enemy
    # are rebuilt from owner_grid and agent_ids.
    agent_ids: np.ndarray
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
    wins_by_setup: dict[str, list[int]]
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
    seed = str(random.randint(0, 2 ** 31 - 1))
    setup = bridge.reset(
        seed=seed,
        map="mid_plains",
        nations=NATIONS,
        difficulty=difficulty,
    )
    episode = _episode_state_from_setup(setup, gold_stats)
    episode.difficulty = difficulty
    episode.seed = seed
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
    agent_id_buf = np.zeros((STEPS, num_envs), dtype=np.int16)
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
    wins_by_setup: dict[str, list[int]] = {}
    end_records: list[tuple] = []

    capped_count = 0
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

            agent_id_buf[i, k] = episode.agent_small_id
            owner_grid_buf[i, k] = episode.owner_grid
            ratio_table_buf[i, k] = episode.ratio_table
            scalar_buf[i, k] = episode.scalar
            entities_buf[i, k] = episode.entities
            entity_masks[i, k] = episode.entity_mask
            actions[i, k] = actions_np[k]
            log_probs[i, k] = log_probs_np[k]
            values[i, k] = values_np[k]

            entity_features, prev_stats, curr_stats, done, won = apply_step(episode, step_result)
            episode.ticks += ACT_EVERY
            episode.peak = max(episode.peak, float(episode.scalar[2]))
            capped = False
            if not done and episode.ticks >= MAX_EPISODE_TICKS:
                others = [v.get("tilesOwned", 0.0) for sid, v in episode.player_stats.items() if sid != episode.agent_small_id]
                won = curr_stats.get("tilesOwned", 0.0) > max(others, default=0.0)
                done, capped = True, True
                capped_count += 1
            dones[i, k] = done
            wins[i, k] = won
            rewards[i, k] = compute_reward(prev_stats, curr_stats, done, won, step_result["totalLand"], GAMMA_STEP)

            _apply_gold_stats(episode.scalar, entity_features, gold_stats)
            episode.entities, episode.entity_mask = pad_entities(entity_features)

            if done:
                kind = "cap" if capped else "win" if won else "other" if step_result.get("winner") else "elim"
                end_records.append((kind, episode.ticks, episode.peak))
                num_nations = sum(1 for t in episode.player_types.values() if t == "NATION")
                bucket = wins_by_setup.setdefault(f"{num_nations}{episode.difficulty[:1]}", [0, 0])
                bucket[1] += 1
                bucket[0] += int(won)
                episodes[k] = start_episode(vec_bridge.bridges[k], gold_stats)

    if capped_count > CAP_FLAG_FRAC * max(int(dones.sum()), 1):
        print(f"[cap-flag] {capped_count}/{int(dones.sum())} episodes ended at the tick cap this update")

    # an env that never finished is bootstrapped from last_value instead of a terminal reward
    for k in range(num_envs):
        if not dones[:, k].any():
            print(f"[game-length] env {k} ran the full {STEPS}-tick rollout without finishing (ticks={episodes[k].ticks})")

    rollout = Rollout(
        agent_ids=agent_id_buf,
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
        wins_by_setup=wins_by_setup,
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


def ppo_update(
    rollout: Rollout,
    agent,
    optimizer,
    update: int,
    total_train_hours: float,
    return_stats: RunningMeanStd,
) -> float:
    agent_id_buf = torch.from_numpy(rollout.agent_ids)
    static_planes = torch.from_numpy(rollout.static_planes).to(DEVICE)
    owner_grid_buf = torch.from_numpy(rollout.owner_grid)
    ratio_table_buf = torch.from_numpy(rollout.ratio_table)
    scalar_buf = torch.from_numpy(rollout.scalar)
    entities_buf = torch.from_numpy(rollout.entities)
    entity_masks = torch.from_numpy(rollout.entity_mask)
    actions = torch.from_numpy(rollout.actions)
    log_probs = torch.from_numpy(rollout.log_probs)
    values = torch.from_numpy(rollout.values).float()
    rewards = torch.from_numpy(rollout.rewards)
    dones = torch.from_numpy(rollout.dones)

    total = STEPS * rollout.num_envs
    flat_agent_id = agent_id_buf.reshape(total)
    flat_owner = owner_grid_buf.reshape(total, *owner_grid_buf.shape[2:])
    flat_ratio = ratio_table_buf.reshape(total, *ratio_table_buf.shape[2:])
    flat_scalar = scalar_buf.reshape(total, *scalar_buf.shape[2:])
    flat_entities = entities_buf.reshape(total, *entities_buf.shape[2:])
    flat_entity_mask = entity_masks.reshape(total, *entity_masks.shape[2:])

    running_advantage = torch.zeros(rollout.num_envs)
    next_value = rollout.last_value
    advantages = torch.zeros(STEPS, rollout.num_envs)
    returns = torch.zeros(STEPS, rollout.num_envs)
    for t in reversed(range(STEPS)):
        non_term = 1 - dones[t].float()
        delta = rewards[t] + (GAMMA_STEP * next_value * non_term) - values[t]
        running_advantage = delta + (GAMMA_STEP * GAE_LAMBDA * non_term * running_advantage)
        next_value = values[t]
        advantages[t] = running_advantage
        returns[t] = advantages[t] + values[t]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # value errors are scaled by a running return std; rare wins have much larger returns than losses
    return_stats.update(returns.numpy().flatten())
    return_std = (return_stats.var + 1e-8) ** 0.5

    agent_id_buf = flat_agent_id
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

    rollout_elapsed = rollout.elapsed
    ppo_start = time.perf_counter()

    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    gate_kl_sum = 0.0
    num_minibatches = 0
    approx_kl = 0.0

    for epoch in range(PPO_EPOCHS):
        indices = torch.randperm(total)
        for start in range(0, total, MINIBATCH_SIZE):
            batch_idx = indices[start:start + MINIBATCH_SIZE]

            batch_scalar = scalar_buf[batch_idx].to(DEVICE)
            batch_spatial = reconstruct_spatial_torch(
                agent_id_buf[batch_idx].to(DEVICE).long(), owner_grid_buf[batch_idx].to(DEVICE).long(),
                ratio_table_buf[batch_idx].to(DEVICE).float(), batch_scalar[:, 1], static_planes,
            )
            batch_entities = entities_buf[batch_idx].to(DEVICE)
            batch_entity_mask = entity_masks[batch_idx].to(DEVICE)
            _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                spatial=batch_spatial,
                scalar=batch_scalar,
                entities=batch_entities,
                entity_mask=batch_entity_mask,
                action=actions[batch_idx].to(DEVICE)
            )

            batch_log_probs = log_probs[batch_idx].to(DEVICE)
            batch_advantages = advantages[batch_idx].to(DEVICE)
            batch_returns = returns[batch_idx].to(DEVICE)

            ratio = torch.exp(new_log_prob - batch_log_probs)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()  # logged, not acted on

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            new_value = new_value.squeeze(-1)
            value_loss = 0.5 * torch.square((new_value - batch_returns) / return_std).mean()

            entropy_mean = entropy.mean()
            gate_kl_mean = agent.gate_kl.mean()
            loss = policy_loss + VALUE_COEF * value_loss + GATE_KL_COEF * gate_kl_mean - ENTROPY_COEF * entropy_mean
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy_mean.item()
            gate_kl_sum += gate_kl_mean.item()
            num_minibatches += 1

    ppo_elapsed = time.perf_counter() - ppo_start
    total_train_hours += (rollout_elapsed + ppo_elapsed) / 3600
    num_minibatches = max(num_minibatches, 1)

    setups = " ".join(f"n{n}={w}/{t}" for n, (w, t) in sorted(rollout.wins_by_setup.items()))
    print(
        f"update {update}: mean_reward={rewards.mean().item():.3f} "
        f"episodes_finished={rollout.episodes_finished} wins={rollout.episode_wins} "
        f"[{setups}] "
        f"ends({_ends_str(rollout.end_records)}) "
        f"policy_loss={policy_loss_sum / num_minibatches:.4f} "
        f"value_loss={value_loss_sum / num_minibatches:.4f} "
        f"entropy={entropy_sum / num_minibatches:.4f} gate_kl={gate_kl_sum / num_minibatches:.4f} "
        f"approx_kl={approx_kl:.4f} "
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
                if any(s["exp_avg"].shape != p.shape for p, s in ((p, optimizer.state[p]) for p in agent.parameters()) if s):
                    raise ValueError("optimizer state shape mismatch")
            except (KeyError, TypeError, ValueError, RuntimeError):
                optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
                warmup_from_update = 0
                print("optimizer state not restored, warming up the learning rate")
            print(f"resumed from {checkpoint_path}")
        else:
            warmup_from_update = 0

        print(f"training on device={DEVICE}, num_envs={NUM_ENVS}, nations={NATIONS}, difficulty={DIFFICULTIES}")

        for update in range(NUM_UPDATES):
            episodes, rollout = collect_rollout(vec_bridge, agent, episodes, gold_stats)

            if warmup_from_update is not None:
                since = update - warmup_from_update
                warmup_frac = min((since + 1) / LR_WARMUP_UPDATES, 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = LR * warmup_frac
                if since + 1 >= LR_WARMUP_UPDATES:
                    warmup_from_update = None

            total_train_hours = ppo_update(
                rollout, agent, optimizer, update, total_train_hours,
                return_stats,
            )
            save_total_train_hours(total_train_hours)
            del rollout  # frees the buffers before the next rollout allocates its own

            if update != 0 and update % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(agent, update, gold_stats, return_stats, total_train_hours, optimizer)

        save_checkpoint(agent, NUM_UPDATES, gold_stats, return_stats, total_train_hours, optimizer)


if __name__ == "__main__":
    train()
