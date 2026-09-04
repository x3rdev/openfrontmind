WIN_REWARD = 150.0  # was 1000 - kept terminal reward from swamping the dense signal below
LOSS_REWARD = -150.0
SCALE = 100.0

# Peak of the engine's troop production formula (Config.ts troopIncreaseRate()).
TROOP_RATIO_TARGET = 0.422
# was 0.3 - full-episode shaping totaled ~0.015 at that scale (5000x smaller than the
# terminal reward or even a modest tile-fraction swing), effectively inert regardless
# of how reasonable the coefficient looked. Rescaled so a full climb from spawn ratio
# to target lands around ~15, same order of magnitude as the other reward terms.
# 2026-08-27: cut 300 -> 100 (distinct experiment from the grace-softening below - that
# reshaped the curve near the floor and still collapsed at update12, if anything faster
# than a fix that didn't touch shaping at all, which is evidence against magnitude
# being the dominant driver). Testing a straight cut anyway: if collapse persists at
# the same rate even with shaping this much weaker, that's a real signal the cause is
# structural (state-distribution narrowing, see CLAUDE.md's "RL Forgets" discussion),
# not reward-magnitude tuning. See CLAUDE.md.
SHAPING_COEF = 100.0

# Config.attackAmount() for Human/agent players - every attack spends this fraction of
# current troops, so troop_ratio necessarily dips below TARGET after attacking even
# from a perfectly-timed starting point. Verified empirically (2026-08-27): tracing a
# real rollout showed every large shaping-reward spike (~-5.5 to -5.8, vs. a full
# climb from spawn only being worth ~15 total) was this exact mechanic - not a bug -
# but penalized quadratically as if any dip below TARGET were equally bad, which
# punished the necessary multi-attack sequences that actually finish off a win.
ATTACK_COST = 0.20
# deficit produced by exactly one attack landing right at TARGET.
TROOP_RATIO_GRACE = TROOP_RATIO_TARGET * ATTACK_COST
# Tried and reverted: a symmetric bonus for banking above TARGET (to "bias toward more
# than 42%" directly). Potential-based shaping telescopes to zero over any round trip,
# so a bonus for building a buffer is mathematically forced to be clawed back in full
# the moment it's spent - checked numerically, a clean attack from a well-prepared
# ratio down to exactly TARGET went from 0 penalty (already fine under the original
# formula) to -31.65 (worse than any real spike found) with the bonus added. Can't get
# "bias toward more" as a free lunch from a state potential; dropped.
# Instead: soften (not zero out) the penalty within one attack's grace distance of the
# floor, quadratic growth resuming beyond it (continuous at the boundary) - still
# discourages chronic below-floor attacking, just isn't as harsh for the unavoidable
# single-attack case as treating every deficit identically was.
TROOP_RATIO_GRACE_SOFTEN = 0.25


def _troop_potential(ratio: float) -> float:
    if ratio >= TROOP_RATIO_TARGET:
        return 0.0
    deficit = TROOP_RATIO_TARGET - ratio
    if deficit <= TROOP_RATIO_GRACE:
        return -(deficit ** 2) * TROOP_RATIO_GRACE_SOFTEN
    boundary = -(TROOP_RATIO_GRACE ** 2) * TROOP_RATIO_GRACE_SOFTEN
    return boundary - (deficit - TROOP_RATIO_GRACE) ** 2


def compute_reward(
    prev_stats: dict, curr_stats: dict, done: bool, won: bool, total_land: float, gamma: float
) -> float:
    if done:
        return WIN_REWARD if won else LOSS_REWARD
    prev_tiles = prev_stats.get("tilesOwned", 0.0)
    curr_tiles = curr_stats.get("tilesOwned", 0.0)

    prev_fraction = prev_tiles / max(total_land, 1)
    curr_fraction = curr_tiles / max(total_land, 1)

    # Potential-based shaping (Ng, Harada & Russell 1999) - telescopes to zero over an episode.
    prev_ratio = prev_stats.get("troops", 0.0) / max(prev_stats.get("maxTroops", 1.0), 1)
    curr_ratio = curr_stats.get("troops", 0.0) / max(curr_stats.get("maxTroops", 1.0), 1)
    shaping = SHAPING_COEF * (gamma * _troop_potential(curr_ratio) - _troop_potential(prev_ratio))

    return SCALE*(curr_fraction-prev_fraction) + shaping
