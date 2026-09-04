from contextlib import ExitStack
from dataclasses import dataclass, field
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


# 2026-08-26: nations=2 games run long - STEPS bumped back to 5000 (horizon-matched via
# GAMMA=1-1/STEPS below). Caught main train.py process at 25.8GB RSS with 1.8GB already
# swapped at STEPS=2000/NUM_ENVS=8 (spatial_buf is float16 now, 6 channels, not the
# smaller bool/3-4-channel buffer the old sizing math assumed) - cut NUM_ENVS down to 2
# to fit STEPS=5000 plus the ~3.2GB SIL buffer in the ~31GB WSL2 cap.
# Then compressed the rollout/SIL buffers themselves instead of just shrinking NUM_ENVS:
# troop_ratio_plane/troop_advantage_plane (2 float16 channels, continuous) replaced by
# owner_grid (int16) + a small per-tick per-owner ratio_table, reconstructed via gather
# just before the forward pass (see observation.py's reconstruct_spatial_torch) instead
# of storing the materialized (H, W) planes for every one of STEPS ticks. The remaining
# 4 planes (is_land, is_impassable, is_mine, is_enemy) are genuinely binary, so they're
# back to bool instead of being dragged up to float16 alongside the continuous ones.
# ~49% smaller per-pixel (12 bytes/px -> ~6 bytes/px). NUM_ENVS=8 tried first (matching
# the original pre-thrashing config) - still thrashed hard: swap hit 8.0/8.0GB (fully
# exhausted) partway through update 0's PPO phase, main process dropped into D-state
# (uninterruptible sleep, CPU% collapsed 130%->51%) - classic thrashing, not a close
# call. The ~49% per-pixel win wasn't enough to make NUM_ENVS=8 safe on its own. Cut to
# 6 - re-verify memory stays flat (no swap growth) before trusting this either.
# 2026-08-26: WSL now reports 39GB total/37GB free (vs the ~31GB cap assumed above) -
# retried NUM_ENVS=8 anyway. Same failure repeated: memory climbed steadily across
# updates 0-2 (swap 0 -> 1.1GB -> 1.7GB, mem used 22GB -> 38GB/39GB, 278MB available)
# and was killed before it fully thrashed. The higher total RAM didn't help - something
# scales with NUM_ENVS beyond what the ~31GB-cap math accounted for. Back to 6.
# 2026-08-27: tried NUM_ENVS=7 (one step up from the confirmed-stable 6) - also climbed
# to 37GB used/1.7GB available within 3 updates and was killed before it thrashed. 6
# appears to be the actual ceiling at STEPS=5000 on this box, not just a conservative
# floor - don't keep nudging this up without also cutting STEPS or compressing buffers.
# 2026-09-02: nations=3 curriculum bump made games run longer again - the new
# [game-length] warning (train() loop below) fired twice in the very first update,
# meaning STEPS=5000 is already too small some of the time, not just approaching the
# edge. Bumped STEPS 5000->6500 (1.3x); traded NUM_ENVS 6->5 to keep total buffer
# memory (~STEPS*NUM_ENVS) roughly flat rather than risking the same thrashing this
# combination hit before at a higher product. Re-verify memory stays flat before
# trusting this either, same as every previous STEPS/NUM_ENVS change here.
STEPS = 6500
NUM_UPDATES = 10000
NUM_ENVS = 5

# 2026-08-27: stopgap cap, not a rigorously-correct time-limit-bootstrap implementation
# - games can otherwise run to the engine's own ~170-min hard cap, and a handful of very
# long episodes alive at once was the likely cause of a memory spike observed mid-run
# (35GB used vs. the normal ~20GB baseline, though it subsided on its own once those
# episodes finished). Flagged via a print every time it fires so frequency is visible -
# frequent hits mean 10000 is cutting off real games and should be raised.
MAX_EPISODE_TICKS = 10000

# bots=1 (Tribe) curriculum tried again 2026-08-17 with the gated action head, reverted
# again within 2 updates on explicit user instruction - Tribes are never to be used as
# the main training opponent (too weak, teaches spam-attack as "correct"). See CLAUDE.md.
BOTS = 0
# 2026-09-02: curriculum bump to nations=3, warm-started from the nations=2 checkpoint
# that reached 83% win rate (sustained 81 updates) after the adaptive-ENTROPY_COEF fix -
# same reasoning as the earlier nations=1->2 bump: nations=2 is solved enough to move on,
# and re-proving nations=2 further wouldn't validate anything new. Milestone preserved at
# checkpoints_milestones/agent_20260901_233001_update64_nations2_winrate83pct.pt.
NATIONS_MIN = 3
NATIONS_MAX = 3
NATIONS = NATIONS_MAX  # single-value fallback for watch.py's default
DIFFICULTY = "Easy"  # set explicitly so it can't silently drift (default was Easy anyway)

# Experiment (2026-08-22): fixed 16-seed pool instead of fully random. Result: negative
# - in-pool win rate stayed noisy/flat (same range as free-random seeds, no faster
# learning), zero transfer to held-out seeds, and the nations=1 skill (was 97.5%)
# degraded to 1/3 on the same fixed diagnostic seeds from lack of exposure during this
# run. Reverted to fully random. Empty = fully random (the normal behavior).
SEED_POOL: list[str] = []

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# overridable via env var so multiple isolated repro runs can execute in parallel
# against the same codebase without clobbering each other's checkpoints/state file -
# see the 2026-09-03 crash-reproducibility experiment in CLAUDE.md. Unset in normal use.
CHECKPOINT_DIR = Path(os.environ.get("OFM_CHECKPOINT_DIR", str(Path(__file__).parent / "checkpoints")))
CHECKPOINT_INTERVAL = 32
# was 10 (~288 updates of history at this INTERVAL) - too short to bracket a full
# burst-then-crash cycle (observed spanning ~200-300 updates), so every checkpoint
# from the last such cycle had already rotated out by the time it was worth
# investigating. 50 covers ~1568 updates at ~9MB/checkpoint (~450MB total) - trivial
# disk cost for keeping enough history to actually do that analysis next time.
CHECKPOINT_KEEP = 50

# Lives outside CHECKPOINT_DIR on purpose - archiving/clearing checkpoints/ for a cold
# restart (weights reset) should never also reset this. total_train_hours is real elapsed
# wall-clock compute, not a property of the current weights; it was silently reset twice
# (2026-08-25) because it previously only lived inside the checkpoint dict.
TRAIN_STATE_PATH = Path(os.environ.get("OFM_TRAIN_STATE_PATH", str(Path(__file__).parent / "train_state.json")))


def load_total_train_hours() -> float:
    if TRAIN_STATE_PATH.exists():
        return json.loads(TRAIN_STATE_PATH.read_text()).get("total_train_hours", 0.0)
    return 0.0


def save_total_train_hours(total_train_hours: float) -> None:
    TRAIN_STATE_PATH.write_text(json.dumps({"total_train_hours": total_train_hours}))


def save_checkpoint(
    checkpoint_dir: Path,
    agent, update: int, gold_stats: RunningMeanStd, return_stats: RunningMeanStd, total_train_hours: float,
    win_rate_ema: float, entropy_scale: float = 1.0, lr_scale: float = 1.0,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"agent_{time.strftime('%Y%m%d_%H%M%S')}_update{update}.pt"
    # gold_stats/return_stats/win_rate_ema/entropy_scale/lr_scale travel with the weights
    # so a resume normalizes/adapts consistently instead of resetting to defaults and
    # spiking ENTROPY_COEF back to max (or losing a PBT-inherited perturbation) on restart.
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
            "win_rate_ema": win_rate_ema,
            "entropy_scale": entropy_scale,
            "lr_scale": lr_scale,
        },
        path,
    )

    checkpoints = sorted(checkpoint_dir.glob("agent_*.pt"))
    for stale in checkpoints[:-CHECKPOINT_KEEP]:
        stale.unlink()

    return path


def latest_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("agent_*.pt")) if checkpoint_dir.exists() else []
    return checkpoints[-1] if checkpoints else None

LR = 1e-4
# derived from STEPS, not a separate literal - GAE's effective horizon is 1/(1-GAMMA), and
# a hand-set GAMMA silently drifting out of sync with STEPS (e.g. STEPS bumped for longer
# games, GAMMA left at the old value) was a real bug that went unnoticed for the whole
# nations=2 phase - see CLAUDE.md/session log.
GAMMA = 1 - 1 / STEPS
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
CLIP_EPS = 0.2
# was 0.5, then 5.0 - the 10x jump was too aggressive: entropy collapsed to ~0.0000
# and episodes stopped finishing normally for ~300 updates (approx_kl spiked to
# 0.02-0.06, well above the normal 0.0002-0.005, triggering repeated kl_stopped
# events) before drifting back to normal on its own. Backed off to a smaller 3x step.
VALUE_COEF = 1.5
# was a flat 0.05, then a flat 0.15 (see git history) - 0.05 alone left nations=2 win
# rate flat ~0.3% for 350 updates (PPO too rarely saw the reachable-but-rare patient
# trajectory to reinforce it); 0.15 alone fixed discovery but then couldn't hold onto
# it - repeated win-rate bursts (up to ~59%) always decayed back down within tens of
# updates, tracking entropy the whole way (corr(entropy, frac_low_ratio)=0.76,
# corr(entropy, win_rate)=-0.72 across a 516-update run) - a fixed entropy coefficient
# can't be both "enough exploration to find patience" and "little enough to keep it"
# at the same time. SAC-inspired fix: adapt the coefficient to an EMA of win rate
# instead of a fixed constant - high (slow-discovery regime) when win rate is low,
# low (letting confident behavior lock in) once win rate is healthy. Persisted in the
# checkpoint (like gold_stats/return_stats) so a resume doesn't reset the EMA to 0 and
# spike entropy back to max every restart.
ENTROPY_COEF_MAX = 0.15
ENTROPY_COEF_MIN = 0.05
WIN_RATE_EMA_ALPHA = 0.1
# win_rate_ema at/above this fully steps ENTROPY_COEF down to ENTROPY_COEF_MIN -
# chosen below the ~59% peak seen this session so it can actually engage before a
# streak's typical decay point, not just at an already-rare best case.
WIN_RATE_TARGET = 0.45


def entropy_coef_for(win_rate_ema: float) -> float:
    t = min(win_rate_ema / WIN_RATE_TARGET, 1.0)
    return ENTROPY_COEF_MAX - (ENTROPY_COEF_MAX - ENTROPY_COEF_MIN) * t


# 2026-09-02: small population-based training (Jaderberg et al. 2017) layered on top of
# the existing single-lineage PPO loop, to hedge against the collapse/oscillation pattern
# documented all session - one lineage drifting into a bad patch with no guarantee of
# self-recovery. Each member still trains via the exact same PPO update as before; PBT
# only adds a periodic "exploit" (a struggling member copies the best member's weights
# outright) + "explore" (the copy's entropy/LR scale gets perturbed, not its weights) step.
# POP_SIZE=2, not more: a real population divides NUM_ENVS (and rollout-buffer memory)
# per member, and this box is already tight (see the STEPS/NUM_ENVS memory notes above) -
# 2 members at NUM_ENVS_PER_MEMBER=2 each (4 total) stays below the 5 a single lineage was
# just running at, trading population diversity for a smaller per-member env count rather
# than growing total memory further.
# overridable via env var - the 2026-09-03 crash-reproducibility experiment runs
# POP_SIZE=1 (single lineage, PBT's exploit/explore trivially no-ops with one member)
# to isolate whether a crash reproduces without PBT's own dynamics as a confound.
POP_SIZE = int(os.environ.get("OFM_POP_SIZE", "2"))
NUM_ENVS_PER_MEMBER = 2
# matches CHECKPOINT_INTERVAL's cadence - no strong reason for a different period yet.
PBT_INTERVAL = 32
# win_rate_ema gap (0-1 scale) that triggers an exploit/explore step - deliberately large
# so a member isn't punished for ordinary noise, only a real, sustained gap between members.
PBT_EXPLOIT_GAP = 0.15
# multiplies the copied member's entropy_scale/lr_scale (not weights) on explore - a
# random one of these per perturbed hyperparameter, mirroring PBT's usual mutation step.
PBT_PERTURB_FACTORS = [0.8, 1.2]

# 2026-09-03: EMA target network for GAE's value bootstrap (DQN/DDPG/TD3/SAC-style),
# motivated by a reproducibility experiment (see CLAUDE.md) - two independent replays
# from the identical checkpoint, different seeds, both decayed on nearly the same
# timeline, meaning the recurring oscillation isn't primarily rollout-variance-driven.
# The live value estimate feeds directly into the very next update's advantage
# computation with no lag, a classic moving-target instability in TD-style learning.
# A full copy of backbone+value_head (not just value_head alone - the estimate depends
# on the shared backbone representation too, which policy gradients also move) is kept
# and updated via a slow EMA, used only to compute GAE's bootstrap values; the live
# agent still trains at full speed as before, including its own value_head (used for
# value_loss's old-value clipping baseline, unchanged). TARGET_TAU is the fraction
# pulled toward the live agent each update - small, so the target moves much slower.
TARGET_TAU = 0.01
# freezes everything but value_head - tests whether V(s) can learn genuine state-
# awareness against a stable, non-degrading policy, isolated from the self-
# reinforcing bad-policy/bad-data loop diagnosed this session (see CLAUDE.md).
# Unfrozen at update1056: ~500 updates of frozen retraining didn't converge toward
# state-awareness (tick-correlation stayed ~0.9, troop_ratio-correlation collapsed
# toward 0, early-game win/loss separation got worse, not better) - not worth more
# time on this lever, so falling back to letting PPO's own advantage signal train
# the whole network jointly despite the imperfect value function.
FREEZE_POLICY = False

@dataclass
class Rollout:
    # compressed form of what encode_observation's "spatial" plane stack holds - see
    # observation.py's reconstruct_spatial_torch. bool_planes are [is_land,
    # is_impassable, is_mine, is_enemy]; owner_grid + ratio_table reconstruct
    # troop_ratio/troop_advantage on demand instead of storing them as float planes
    # for every one of STEPS ticks (~49% smaller than the old 6-channel float16 form).
    bool_planes: np.ndarray
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
    wins_by_nations: dict[int, list[int]]
    elapsed: float


def _apply_gold_stats(scalar: np.ndarray, entity_features: np.ndarray, gold_stats: RunningMeanStd) -> None:
    """Update running gold stats from real entities only, then normalize in place everywhere."""
    is_real = (entity_features[:, 0] == 1.0) | (entity_features[:, 1] == 1.0)  # is_tribe or is_nation
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
    """Used for standalone scripts and per-slot resets in collect_rollout."""
    setup = bridge.reset(
        seed=random.choice(SEED_POOL) if SEED_POOL else str(random.randint(0, 2 ** 31 - 1)),
        map="mid_plains",
        bots=BOTS,
        nations=random.randint(NATIONS_MIN, NATIONS_MAX),
        difficulty=DIFFICULTY,
    )
    return _episode_state_from_setup(setup, gold_stats)


def start_all_episodes(vec_bridge: VecOpenFrontBridge, gold_stats: RunningMeanStd) -> list[EpisodeState]:
    """Each slot needs its own random seed to avoid cloning the same game."""
    return list(vec_bridge.executor.map(lambda b: start_episode(b, gold_stats), vec_bridge.bridges))


def compute_value_for_episodes(agent, episodes: list[EpisodeState]) -> torch.Tensor:
    """V(s) for each env's current state - used both for collect_rollout's own bootstrap
    and (with a target_agent) to compute the target-network bootstrap value in train()."""
    with torch.no_grad():
        spatial = np.stack([e.spatial for e in episodes])
        scalar = np.stack([e.scalar for e in episodes])
        entities = np.stack([e.entities for e in episodes])
        entity_mask = np.stack([e.entity_mask for e in episodes])
        return agent.get_value(
            spatial=torch.from_numpy(spatial).to(DEVICE),
            scalar=torch.from_numpy(scalar).to(DEVICE),
            entities=torch.from_numpy(entities).to(DEVICE),
            entity_mask=torch.from_numpy(entity_mask).to(DEVICE),
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
    # is_land, is_impassable, is_mine, is_enemy - genuinely binary, unlike
    # troop_ratio/troop_advantage below, so bool (not float16) is lossless here.
    bool_planes_buf = np.zeros((STEPS, num_envs, 4, height, width), dtype=bool)
    # int16, not uint16: PyTorch's uint16 op support is spotty, and smallID (12-bit
    # protocol field, max 4095) fits int16 (max 32767) with room to spare.
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
    wins_by_nations: dict[int, list[int]] = {}

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
            for k in range(num_envs)
        ]
        step_results = vec_bridge.step(intents_list)

        for k in range(num_envs):
            episode = episodes[k]
            step_result = step_results[k]

            bool_planes_buf[i, k] = np.stack([
                episode.land_mask, episode.impassable_mask, episode.is_mine, episode.is_enemy,
            ])
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
            if not done and episode.ticks >= MAX_EPISODE_TICKS:
                print(f"[episode-cap] env {k} hit MAX_EPISODE_TICKS={MAX_EPISODE_TICKS} - forcing reset")
                done, won = True, False
            dones[i, k] = done
            wins[i, k] = won

            reward = compute_reward(prev_stats, curr_stats, done, won, step_result["totalLand"], GAMMA)
            rewards[i, k] = reward

            _apply_gold_stats(episode.scalar, entity_features, gold_stats)
            episode.entities, episode.entity_mask = pad_entities(entity_features)

            # reset just this slot - the others keep ticking independently
            if done:
                # an eliminated player's gold is destroyed by the engine itself (see
                # CLAUDE.md's game-mechanics notes), so curr_stats.gold reads as 0 on
                # a death-by-elimination loss - that's the engine's real state, not a
                # logging gap, but it tells us nothing about economic slack right
                # before dying. Use the last tick's stats (pre-elimination) for losses
                # instead, so mean_gold_at_loss reflects unspent gold like the win side
                # does, not just "player is dead."
                end_gold = curr_stats.get("gold", 0.0) if won else prev_stats.get("gold", curr_stats.get("gold", 0.0))
                (episode_end_golds_win if won else episode_end_golds_loss).append(end_gold)
                num_nations = sum(1 for t in episode.player_types.values() if t == "NATION")
                bucket = wins_by_nations.setdefault(num_nations, [0, 0])
                bucket[1] += 1
                if won:
                    bucket[0] += 1
                episodes[k] = start_episode(vec_bridge.bridges[k], gold_stats)

    # an env that never went `done` across the whole STEPS window means its current
    # episode has been running at least that long without a real terminal reward -
    # GAE bootstraps off last_value for it instead, the same horizon-mismatch risk
    # that bit STEPS=500 undershooting a ~1850-tick decline arc (see CLAUDE.md).
    # episode.ticks is the episode's own total age, not just this rollout's share of
    # it, so it also flags an episode that straddled several rollouts in a row.
    for k in range(num_envs):
        if not dones[:, k].any():
            print(
                f"[game-length] env {k} ran the full STEPS={STEPS}-tick rollout without "
                f"finishing (episode.ticks={episodes[k].ticks}) - buffer may be undersized"
            )

    last_value = compute_value_for_episodes(agent, episodes)

    rollout = Rollout(
        bool_planes=bool_planes_buf,
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
        last_value=last_value,
        num_envs=num_envs,
        episodes_finished=int(dones.sum().item()),
        episode_wins = int(wins.sum().item()),
        episode_end_golds_win = episode_end_golds_win,
        episode_end_golds_loss = episode_end_golds_loss,
        wins_by_nations = wins_by_nations,
        elapsed = time.perf_counter()-rollout_start)

    return episodes, rollout

def ppo_update(
    rollout: Rollout,
    agent,
    target_agent,
    target_last_value: torch.Tensor,
    optimizer,
    update: int,
    total_train_hours: float,
    return_stats: RunningMeanStd,
    entropy_coef: float,
    member_id: int = 0,
) -> tuple[int, int, float]:
    # snapshot before any gradient steps this update - compared against the post-update
    # snapshot below to log how far this update actually moved the weights (L2), so a
    # burst update can be compared directly against a calm one instead of only inferring
    # drift from win rate after the fact, and without depending on which checkpoints
    # CHECKPOINT_KEEP happened to still have on disk.
    params_before = torch.cat([p.detach().flatten() for p in agent.parameters()])

    bool_planes_buf = torch.from_numpy(rollout.bool_planes)
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

    # target-network values for GAE's bootstrap (see TARGET_TAU above) - a slow-moving
    # copy of the value-computation path, not the live agent's own just-collected
    # `values` above (those still seed value_loss's old-value clipping baseline below,
    # unchanged). Chunked like the PPO epoch loop's minibatches to avoid materializing
    # the full-rollout float spatial tensor at once (the same memory concern that
    # motivated per-minibatch reconstruction there).
    total_ticks = STEPS * rollout.num_envs
    flat_bool = bool_planes_buf.reshape(total_ticks, *bool_planes_buf.shape[2:])
    flat_owner = owner_grid_buf.reshape(total_ticks, *owner_grid_buf.shape[2:])
    flat_ratio = ratio_table_buf.reshape(total_ticks, *ratio_table_buf.shape[2:])
    flat_scalar = scalar_buf.reshape(total_ticks, *scalar_buf.shape[2:])
    flat_entities = entities_buf.reshape(total_ticks, *entities_buf.shape[2:])
    flat_entity_mask = entity_masks.reshape(total_ticks, *entity_masks.shape[2:])
    flat_target_values = torch.zeros(total_ticks)
    with torch.no_grad():
        for start in range(0, total_ticks, MINIBATCH_SIZE):
            idx = slice(start, start + MINIBATCH_SIZE)
            chunk_scalar = flat_scalar[idx].to(DEVICE)
            chunk_spatial = reconstruct_spatial_torch(
                flat_bool[idx].to(DEVICE), flat_owner[idx].to(DEVICE).long(),
                flat_ratio[idx].to(DEVICE).float(), chunk_scalar[:, 1],
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

    # raw (pre-normalization) scale, logged below - lets a burst-vs-calm update be
    # compared directly instead of only inferring it from win rate after the fact.
    raw_advantage_mean = advantages.mean().item()
    raw_advantage_std = advantages.std().item()
    advantages = (advantages-advantages.mean()) / (advantages.std()+1e-8)

    # persistent (not per-update) scale estimate - a rare win's return is 10-20x a
    # typical loss trajectory's, which was swamping value_loss's contribution to the
    # shared backbone gradient. Only variance, no mean-shift: value_loss is computed
    # from a difference (new_value - batch_returns), so a shared mean subtracted from
    # both sides would cancel anyway - this only rescales the error's magnitude.
    return_stats.update(returns.numpy().flatten())
    return_std = (return_stats.var + 1e-8) ** 0.5

    # flatten (STEPS, num_envs, ...) -> (STEPS*num_envs, ...) for minibatching
    total = STEPS * rollout.num_envs
    bool_planes_buf = bool_planes_buf.reshape(total, *bool_planes_buf.shape[2:])
    owner_grid_buf = owner_grid_buf.reshape(total, *owner_grid_buf.shape[2:])
    ratio_table_buf = ratio_table_buf.reshape(total, *ratio_table_buf.shape[2:])
    scalar_buf = scalar_buf.reshape(total, *scalar_buf.shape[2:])
    entities_buf = entities_buf.reshape(total, *entities_buf.shape[2:])
    entity_masks = entity_masks.reshape(total, *entity_masks.shape[2:])
    actions = actions.reshape(total)
    log_probs = log_probs.reshape(total)
    advantages = advantages.reshape(total)
    returns = returns.reshape(total)
    values = values.reshape(total)

    # 2026-08-27: direct test of the "state-distribution narrowing" collapse hypothesis
    # (see CLAUDE.md) - LOW_RATIO_THRESHOLD picks out the rare/dangerous tail of states
    # a winning streak should visit less and less. frac_low_ratio tracks whether that
    # tail actually shrinks over updates (confirms/denies narrowing is even happening);
    # the value-error split below (low-ratio vs rest) tests whether prediction quality
    # specifically degrades in that shrinking tail while the aggregate value_loss (which
    # is dominated by the common, well-covered states) looks fine - the actual
    # falsifiable prediction of the hypothesis, not just an analogy to it.
    LOW_RATIO_THRESHOLD = 0.15
    frac_low_ratio = (scalar_buf[:, 1] < LOW_RATIO_THRESHOLD).float().mean().item()

    rollout_elapsed = rollout.elapsed
    ppo_start = time.perf_counter()

    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    num_minibatches = 0
    approx_kl = 0.0
    epochs_run = 0

    # 2026-08-27: TARGET_KL/kl_stopped removed. Historical-log analysis across 9
    # independent collapse events found kl_stopped fires routinely (harmlessly) during
    # healthy winning streaks - one streak hit it 8 times over 119 updates with no
    # issue - and doesn't reliably precede the actual collapses (one had 27 clean
    # updates between the last kl_stopped and the eventual break). It was never real
    # protection: it only measures divergence on the CURRENT rollout batch, which
    # narrows to the winning-trajectory state distribution as win rate rises - per "RL
    # Forgets! Towards Continual Policy Optimization" (arXiv:2607.04364), this is
    # exactly the blind spot that causes this class of failure - forgetting happens on
    # the *prior*-task/rarer-state distribution the current-batch KL never looks at.
    # Kept as a logged diagnostic only, no longer acted on. See CLAUDE.md.
    # value-error-by-bucket: only accumulated on epoch 0, so it reflects the value
    # function as it entered this update (trained on the *previous* update's data),
    # comparable update-to-update - not blended with within-update gradient steps.
    low_ratio_err_sum, low_ratio_err_count = 0.0, 0
    normal_err_sum, normal_err_count = 0.0, 0
    for epoch in range(PPO_EPOCHS):
        epochs_run = epoch + 1
        indices = torch.randperm(total)
        for start in range(0, total, MINIBATCH_SIZE):
            batch_idx = indices[start:start+MINIBATCH_SIZE]

            batch_scalar = scalar_buf[batch_idx].to(DEVICE)
            batch_spatial = reconstruct_spatial_torch(
                bool_planes_buf[batch_idx].to(DEVICE), owner_grid_buf[batch_idx].to(DEVICE).long(),
                ratio_table_buf[batch_idx].to(DEVICE).float(), batch_scalar[:, 1],
            )

            new_action, new_log_prob, entropy, new_value = agent.get_action_and_value(
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

            ratio = torch.exp(new_log_prob-batch_log_probs)

            # http://joschu.net/blog/kl-approx.html - unbiased, always non-negative -
            # logged only now, see the note above the epoch loop.
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advantages

            policy_loss = -torch.min(unclipped, clipped).mean()

            # same clip-and-take-worse-case idea as the policy ratio, applied to value
            new_value = new_value.squeeze(-1)
            value_clipped = batch_old_values + torch.clamp(
                new_value - batch_old_values, -CLIP_EPS, CLIP_EPS
            )
            value_loss_unclipped = torch.square((new_value - batch_returns) / return_std)
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

            loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy_bonus.item()
            num_minibatches += 1

    mean_gold_at_win = (
        sum(rollout.episode_end_golds_win) / len(rollout.episode_end_golds_win) if rollout.episode_end_golds_win else 0.0
    )
    mean_gold_at_loss = (
        sum(rollout.episode_end_golds_loss) / len(rollout.episode_end_golds_loss) if rollout.episode_end_golds_loss else 0.0
    )

    params_after = torch.cat([p.detach().flatten() for p in agent.parameters()])
    weight_delta_l2 = (params_after - params_before).norm().item()

    ppo_elapsed = time.perf_counter() - ppo_start
    total_elapsed = rollout_elapsed + ppo_elapsed
    ticks_per_sec = (STEPS * rollout.num_envs) / rollout_elapsed
    total_train_hours += total_elapsed / 3600

    num_minibatches = max(num_minibatches, 1)
    wins_by_nations_str = " ".join(
        f"n{n}={w}/{t}" for n, (w, t) in sorted(rollout.wins_by_nations.items())
    )
    print(
        f"member{member_id} update {update}: mean_reward={rewards.mean().item():.3f} "
        f"episodes_finished={rollout.episodes_finished} wins={rollout.episode_wins} "
        f"[{wins_by_nations_str}] "
        f"mean_gold_at_win={mean_gold_at_win:,.0f} mean_gold_at_loss={mean_gold_at_loss:,.0f} "
        f"policy_loss={policy_loss_sum / num_minibatches:.4f} "
        f"value_loss={value_loss_sum / num_minibatches:.4f} "
        f"entropy={entropy_sum / num_minibatches:.4f} "
        f"approx_kl={approx_kl:.4f} epochs_run={epochs_run} "
        f"frac_low_ratio={frac_low_ratio:.4f} "
        f"value_err_low_ratio={low_ratio_err_sum / max(low_ratio_err_count, 1):.4f}(n={low_ratio_err_count}) "
        f"value_err_normal={normal_err_sum / max(normal_err_count, 1):.4f}(n={normal_err_count}) "
        f"raw_adv_mean={raw_advantage_mean:.4f} raw_adv_std={raw_advantage_std:.4f} "
        f"weight_delta_l2={weight_delta_l2:.4f} entropy_coef={entropy_coef:.4f} "
        f"time={total_elapsed:.1f}s (rollout={rollout_elapsed:.1f}s ppo={ppo_elapsed:.1f}s) "
        f"ticks/sec={ticks_per_sec:.1f} total_train_hours={total_train_hours:.2f}"
    )

    return rollout.episodes_finished, rollout.episode_wins, total_train_hours


@dataclass
class Member:
    id: int
    checkpoint_dir: Path
    vec_bridge: VecOpenFrontBridge
    agent: object
    target_agent: object
    optimizer: torch.optim.Optimizer
    episodes: list = field(default_factory=list)
    gold_stats: RunningMeanStd = field(default_factory=RunningMeanStd)
    return_stats: RunningMeanStd = field(default_factory=RunningMeanStd)
    win_rate_ema: float = 0.0
    entropy_scale: float = 1.0
    lr_scale: float = 1.0
    total_episodes: int = 0
    total_wins: int = 0
    checkpoint_total_train_hours: float = 0.0


def _load_member(member_id: int, vec_bridge: VecOpenFrontBridge) -> Member:
    checkpoint_dir = CHECKPOINT_DIR / f"member{member_id}"
    gold_stats = RunningMeanStd()
    return_stats = RunningMeanStd()
    checkpoint_path = latest_checkpoint(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE) if checkpoint_path is not None else None
    win_rate_ema = 0.0
    entropy_scale = 1.0
    lr_scale = 1.0
    checkpoint_total_train_hours = 0.0
    if checkpoint is not None:
        gold_stats.mean = checkpoint["gold_mean"]
        gold_stats.var = checkpoint["gold_var"]
        gold_stats.count = checkpoint["gold_count"]
        if "return_mean" in checkpoint:
            return_stats.mean = checkpoint["return_mean"]
            return_stats.var = checkpoint["return_var"]
            return_stats.count = checkpoint["return_count"]
        win_rate_ema = checkpoint.get("win_rate_ema", 0.0)
        entropy_scale = checkpoint.get("entropy_scale", 1.0)
        lr_scale = checkpoint.get("lr_scale", 1.0)
        checkpoint_total_train_hours = checkpoint.get("total_train_hours", 0.0)

    episodes = start_all_episodes(vec_bridge, gold_stats)
    agent = Agent.from_observation(episodes[0].spatial, episodes[0].scalar, episodes[0].entities).to(DEVICE)
    if checkpoint is not None:
        # strict=False only skips missing/unexpected keys - a shape mismatch (e.g.
        # entity_dim growing when a new per-entity feature is added) still raises, so
        # drop those keys ourselves first and let them init fresh instead of failing
        # the whole resume.
        own_state = agent.state_dict()
        saved_state = checkpoint["model"]
        reinitialized = [
            k for k, v in saved_state.items() if k in own_state and v.shape != own_state[k].shape
        ]
        for k in reinitialized:
            del saved_state[k]
        missing, unexpected = agent.load_state_dict(saved_state, strict=False)
        if reinitialized:
            print(f"member{member_id} resumed from {checkpoint_path}: reinitialized (shape changed) {reinitialized}")
        if missing or unexpected:
            print(f"member{member_id} partial resume from {checkpoint_path}: missing={missing} unexpected={unexpected}")
        else:
            print(f"member{member_id} resumed from {checkpoint_path}")

    if FREEZE_POLICY:
        for name, param in agent.named_parameters():
            param.requires_grad = "value_head" in name
        print(f"member{member_id} policy frozen - training value_head only")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=LR * lr_scale)

    # target network for GAE's value bootstrap (see TARGET_TAU above) - always
    # (re)started as an exact copy of the just-resumed live agent rather than
    # separately persisted across restarts; losing a few updates of lag right after
    # a restart is a negligible cost next to the complexity of persisting a second
    # full checkpoint's worth of weights.
    target_agent = Agent.from_observation(episodes[0].spatial, episodes[0].scalar, episodes[0].entities).to(DEVICE)
    target_agent.load_state_dict(agent.state_dict())

    return Member(
        id=member_id, checkpoint_dir=checkpoint_dir, vec_bridge=vec_bridge, agent=agent, target_agent=target_agent,
        optimizer=optimizer, episodes=episodes, gold_stats=gold_stats, return_stats=return_stats,
        win_rate_ema=win_rate_ema, entropy_scale=entropy_scale, lr_scale=lr_scale,
        checkpoint_total_train_hours=checkpoint_total_train_hours,
    )


def _pbt_step(members: list[Member], update: int) -> None:
    """Exploit (copy weights from the best member) + explore (perturb the copy's
    entropy/LR scale, not its weights) - see the PBT constants above for rationale."""
    ranked = sorted(members, key=lambda m: m.win_rate_ema, reverse=True)
    best, worst = ranked[0], ranked[-1]
    if best.win_rate_ema - worst.win_rate_ema < PBT_EXPLOIT_GAP:
        return
    worst_ema_before_copy = worst.win_rate_ema  # logged below - overwritten right after

    worst.agent.load_state_dict(best.agent.state_dict())
    # sync the target network too - otherwise the just-copied policy would bootstrap
    # against a stale target left over from its own discarded trajectory, mismatched
    # with the weights it now actually has.
    worst.target_agent.load_state_dict(best.target_agent.state_dict())
    worst.win_rate_ema = best.win_rate_ema
    worst.gold_stats.mean, worst.gold_stats.var, worst.gold_stats.count = (
        best.gold_stats.mean, best.gold_stats.var, best.gold_stats.count
    )
    worst.return_stats.mean, worst.return_stats.var, worst.return_stats.count = (
        best.return_stats.mean, best.return_stats.var, best.return_stats.count
    )
    # clamped so repeated exploit/explore rounds can't compound toward an extreme
    # multiplier over a long run (0.8x or 1.2x every ~32 updates, unbounded, would
    # drift arbitrarily far given enough rounds).
    worst.entropy_scale = max(0.3, min(3.0, best.entropy_scale * random.choice(PBT_PERTURB_FACTORS)))
    worst.lr_scale = max(0.3, min(3.0, best.lr_scale * random.choice(PBT_PERTURB_FACTORS)))
    # stale Adam momentum/variance estimates would otherwise fight the just-copied
    # weights - fresh optimizer state at the new (perturbed) LR instead.
    worst.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, worst.agent.parameters()), lr=LR * worst.lr_scale
    )
    print(
        f"[pbt] update {update}: member{worst.id} (ema={worst_ema_before_copy:.3f} before copy) "
        f"exploited member{best.id} (ema={best.win_rate_ema:.3f}) - "
        f"entropy_scale={worst.entropy_scale:.3f} lr_scale={worst.lr_scale:.3f}"
    )


def train():
    total_train_hours = max(load_total_train_hours(), 0.0)

    with ExitStack() as stack:
        vec_bridges = [stack.enter_context(VecOpenFrontBridge(NUM_ENVS_PER_MEMBER)) for _ in range(POP_SIZE)]
        members = [_load_member(i, vec_bridges[i]) for i in range(POP_SIZE)]
        # a member's own checkpoint can carry a stale total_train_hours from before this
        # ran as a population - same "take whichever source is ahead" rule as before.
        total_train_hours = max([total_train_hours] + [m.checkpoint_total_train_hours for m in members])

        print(f"training on device={DEVICE}, pop_size={POP_SIZE}, num_envs_per_member={NUM_ENVS_PER_MEMBER}")

        for update in range(NUM_UPDATES):
            for m in members:
                m.episodes, rollout = collect_rollout(m.vec_bridge, m.agent, m.episodes, m.gold_stats)
                target_last_value = compute_value_for_episodes(m.target_agent, m.episodes)

                current_win_rate = rollout.episode_wins / rollout.episodes_finished if rollout.episodes_finished else m.win_rate_ema
                m.win_rate_ema = WIN_RATE_EMA_ALPHA * current_win_rate + (1 - WIN_RATE_EMA_ALPHA) * m.win_rate_ema

                episodes_finished, episode_wins, total_train_hours = ppo_update(
                    rollout, m.agent, m.target_agent, target_last_value, m.optimizer, update, total_train_hours,
                    m.return_stats, entropy_coef_for(m.win_rate_ema) * m.entropy_scale, member_id=m.id,
                )
                m.target_agent.soft_update_from(m.agent, TARGET_TAU)
                save_total_train_hours(total_train_hours)
                m.total_episodes += episodes_finished
                m.total_wins += episode_wins

            if update != 0 and update % PBT_INTERVAL == 0:
                _pbt_step(members, update)

            if update != 0 and update % CHECKPOINT_INTERVAL == 0:
                for m in members:
                    save_checkpoint(
                        m.checkpoint_dir, m.agent, update, m.gold_stats, m.return_stats, total_train_hours,
                        m.win_rate_ema, m.entropy_scale, m.lr_scale,
                    )
                # also mirror the current best member's checkpoint into the root
                # checkpoints/ dir - keeps watch.py's plain latest_checkpoint() call
                # (no population awareness) pointing at the best available model.
                best = max(members, key=lambda m: m.win_rate_ema)
                save_checkpoint(
                    CHECKPOINT_DIR, best.agent, update, best.gold_stats, best.return_stats, total_train_hours,
                    best.win_rate_ema, best.entropy_scale, best.lr_scale,
                )

        for m in members:
            save_checkpoint(
                m.checkpoint_dir, m.agent, NUM_UPDATES, m.gold_stats, m.return_stats, total_train_hours,
                m.win_rate_ema, m.entropy_scale, m.lr_scale,
            )
            win_rate = m.total_wins / m.total_episodes if m.total_episodes > 0 else 0.0
            print(
                f"member{m.id} training done: {NUM_UPDATES} updates, {m.total_episodes} episodes played, "
                f"{m.total_wins} wins ({win_rate:.1%} win rate)"
            )


if __name__ == "__main__": train()
