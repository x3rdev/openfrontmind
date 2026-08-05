"""Fixed-size, maskable action space built on attackMask (harness.ts's
computeAttackMask()).

Layout: ATTACK_SLOTS player slots, then a dedicated wilderness slot, then a
dedicated do-nothing slot. Wilderness gets its own index rather than sharing
the generic slots because attackMask marks it with targetID: None, which
would otherwise be indistinguishable from an unassigned/padded slot.
"""

import numpy as np

ATTACK_SLOTS = 10
WILDERNESS_ACTION = ATTACK_SLOTS
DO_NOTHING_ACTION = ATTACK_SLOTS + 1
TOTAL_ACTIONS = ATTACK_SLOTS + 2


def build_action_mask(attack_mask: list) -> tuple[np.ndarray, list]:
    """Turn attackMask into (mask, slot_targets).

    mask: bool, shape (TOTAL_ACTIONS,) - legal actions this tick.
    slot_targets: list of length ATTACK_SLOTS - targetID per player slot, or
        None if unassigned. Player candidates beyond ATTACK_SLOTS are
        truncated (rare per testing).
    """
    # Skip candidates already under an active attack - re-selecting them would just
    # reinforce it (AttackExecution merges troops into the existing attack rather than
    # starting a fresh one), draining troop reserve for little to no extra effectiveness.
    player_candidates = [
        c for c in attack_mask if c["targetID"] is not None and not c["alreadyAttacking"]
    ]
    has_wilderness = any(
        c["targetID"] is None and not c["alreadyAttacking"] for c in attack_mask
    )

    candidates = player_candidates[:ATTACK_SLOTS]
    num_candidates = len(candidates)
    slot_targets = [c["targetID"] for c in candidates] + [None] * (ATTACK_SLOTS - num_candidates)

    mask = np.arange(ATTACK_SLOTS) < num_candidates
    mask = np.append(mask, [has_wilderness, True])  # wilderness, then do-nothing

    return mask, slot_targets


def action_to_intents(action_idx: int, slot_targets: list, agent_client_id: str) -> list[dict]:
    """Turn a sampled action index into an intents list for bridge.step().

    Falls back to [] for an unassigned player slot - shouldn't happen if
    masking is correct, but safer than a malformed intent or a crash.
    """
    if action_idx == DO_NOTHING_ACTION:
        return []
    if action_idx == WILDERNESS_ACTION:
        return [{"type": "attack", "targetID": None, "troops": None, "clientID": agent_client_id}]

    target = slot_targets[action_idx]
    if target is None:
        return []
    return [{"type": "attack", "targetID": target, "troops": None, "clientID": agent_client_id}]
