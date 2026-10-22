import os
import math

WIN_REWARD = 150.0
LOSS_REWARD = -150.0
SCALE = 100.0

# Peak of Config.ts's troopIncreaseRate().
TROOP_RATIO_TARGET = 0.422
SHAPING_COEF = 100.0

# Config.attackAmount() fraction spent per attack.
ATTACK_COST = 0.20
TROOP_RATIO_GRACE = TROOP_RATIO_TARGET * ATTACK_COST
# Softened within one attack's grace distance of the floor; full quadratic beyond it.
TROOP_RATIO_GRACE_SOFTEN = 0.25

# log(ratio/grace_floor) below - unlike the quadratic, doesn't saturate as ratio->0,
# so continued attacking after depletion keeps costing more instead of going "free".
# Zero throughout the grace zone (deficit<=GRACE) so it doesn't touch the single-
# necessary-attack case the grace-softening above already handles; only kicks in
# deeper. log grows slowly enough to stay sane in magnitude (unlike 1/ratio, tried
# first - blew up to -195 by ratio=0.01, dwarfing every other reward term).
TROOP_DEPLETION_COEF = 0.0
TROOP_DEPLETION_EPS = 1e-4
GRACE_FLOOR_RATIO = TROOP_RATIO_TARGET - TROOP_RATIO_GRACE

# 2026-10-10: flat per-tick cost, independent of troop_ratio - the troop-ratio
# shaping above is potential-based, so it telescopes to ~zero over any round trip
# (Ng-Harada-Russell) and can't penalize "waited too long" without also penalizing
# the single necessary attack that ends the wait (tried and reverted earlier this
# session - see CLAUDE.md). A flat time cost doesn't have that problem since it's
# not a function of state. Needed because mean per-tick reward stays positive even
# through 0-win stretches - the dense tile/ratio shaping accumulates every tick
# while the terminal -150 is one tick out of thousands, diluted to near-nothing in
# the per-tick average the policy is actually trained against. Small enough that a
# normal ~70-300 tick wait (-0.7 to -3 total) barely registers; a multi-thousand-
# tick stall costs tens of points against the 150-point terminal reward.
TIME_COST_PER_TICK = float(os.environ.get("OFM_TIME_COST", "0.01"))


# weak penalty above target: regen slows there, but banking is not punished hard.
TROOP_RATIO_ABOVE_COEF = 0.25


def _troop_potential(ratio: float, depletion_coef: float = TROOP_DEPLETION_COEF) -> float:
    d = ratio - TROOP_RATIO_TARGET
    if d >= 0:
        return -TROOP_RATIO_ABOVE_COEF * d * d
    base = -(d * d)
    if depletion_coef > 0:
        base += depletion_coef * math.log(max(ratio, TROOP_DEPLETION_EPS) / GRACE_FLOOR_RATIO)
    return base


# Config.troopIncreaseRate: (10 + troops^0.73 / 4) * (1 - troops / max). Potential is the
# fraction of peak production lost, so the cost of an attack is the regen it forfeits.
PROD_SHAPING = os.environ.get("OFM_PROD_SHAPING", "1") == "1"
PROD_COEF = 20.0
PROD_PEAK_RATIO = 0.73 / 1.73


def _regen(troops: float, max_troops: float) -> float:
    return (10.0 + max(troops, 0.0) ** 0.73 / 4.0) * max(1.0 - troops / max_troops, 0.0)


def _prod_potential(stats: dict) -> float:
    max_troops = max(stats.get("maxTroops", 1.0), 1.0)
    peak = _regen(PROD_PEAK_RATIO * max_troops, max_troops)
    return -PROD_COEF * (1.0 - _regen(stats.get("troops", 0.0), max_troops) / peak)


def compute_reward(
    prev_stats: dict, curr_stats: dict, done: bool, won: bool, total_land: float, gamma: float,
    depletion_coef: float = TROOP_DEPLETION_COEF,
) -> float:
    if done:
        return WIN_REWARD if won else LOSS_REWARD
    prev_tiles = prev_stats.get("tilesOwned", 0.0)
    curr_tiles = curr_stats.get("tilesOwned", 0.0)

    prev_fraction = prev_tiles / max(total_land, 1)
    curr_fraction = curr_tiles / max(total_land, 1)

    prev_ratio = prev_stats.get("troops", 0.0) / max(prev_stats.get("maxTroops", 1.0), 1)
    curr_ratio = curr_stats.get("troops", 0.0) / max(curr_stats.get("maxTroops", 1.0), 1)
    if PROD_SHAPING:
        shaping = gamma * _prod_potential(curr_stats) - _prod_potential(prev_stats)
    else:
        shaping = SHAPING_COEF * (gamma * _troop_potential(curr_ratio, depletion_coef) - _troop_potential(prev_ratio, depletion_coef))

    return SCALE*(curr_fraction-prev_fraction) + shaping - TIME_COST_PER_TICK


REL_WEIGHT = 0.5
REL_CLIP = 1.0


def strongest_adjacent_troops(attack_mask: list, player_stats: dict) -> float:
    best = 0.0
    for c in attack_mask:
        if c["targetID"] is not None:
            best = max(best, player_stats.get(c["smallID"], {}).get("troops", 0.0))
    return best


def blended_potential(stats: dict, contact: float, opp_troops: float, depletion_coef: float = TROOP_DEPLETION_COEF) -> float:
    troops = stats.get("troops", 0.0)
    econ = _troop_potential(troops / max(stats.get("maxTroops", 1.0), 1), depletion_coef)
    if opp_troops <= 0:
        rel = REL_WEIGHT * REL_CLIP
    else:
        rel = REL_WEIGHT * max(-REL_CLIP, min(REL_CLIP, math.log(max(troops, 1.0) / opp_troops)))
    return (1 - contact) * econ + contact * rel


def compute_reward_blend(
    prev_phi: float | None, prev_stats: dict, curr_stats: dict, done: bool, won: bool, total_land: float,
    gamma: float, contact: float, opp_troops: float, depletion_coef: float = TROOP_DEPLETION_COEF,
) -> tuple[float, float]:
    """Contact-blended potential: self-relative econ target when no one is fighting,
    relative strength vs the strongest bordering opponent in direct combat."""
    if done:
        return (WIN_REWARD if won else LOSS_REWARD), 0.0
    if prev_phi is None:
        prev_phi = blended_potential(prev_stats, 0.0, 0.0, depletion_coef)
    phi = blended_potential(curr_stats, contact, opp_troops, depletion_coef)
    prev_fraction = prev_stats.get("tilesOwned", 0.0) / max(total_land, 1)
    curr_fraction = curr_stats.get("tilesOwned", 0.0) / max(total_land, 1)
    reward = SCALE * (curr_fraction - prev_fraction) + SHAPING_COEF * (gamma * phi - prev_phi) - TIME_COST_PER_TICK
    return reward, phi


LOG_RATIO_FLOOR = 0.05


def compute_reward_logratio(prev_stats: dict, curr_stats: dict, done: bool, won: bool, total_land: float, gamma: float) -> float:
    if done:
        return WIN_REWARD if won else LOSS_REWARD

    def phi(stats: dict) -> float:
        ratio = stats.get("troops", 0.0) / max(stats.get("maxTroops", 1.0), 1)
        return math.log(max(ratio, LOG_RATIO_FLOOR))

    prev_fraction = prev_stats.get("tilesOwned", 0.0) / max(total_land, 1)
    curr_fraction = curr_stats.get("tilesOwned", 0.0) / max(total_land, 1)
    return SCALE * (curr_fraction - prev_fraction) + SHAPING_COEF * (gamma * phi(curr_stats) - phi(prev_stats)) - TIME_COST_PER_TICK
