"""Entity-based action space built on attackMask (harness.ts's computeAttackMask()).

Each legal candidate this tick (bordering players, wilderness, do-nothing) becomes
one row of a variable-length entity feature array, padded to MAX_ENTITIES for
batching. The network scores each row by its own features via a pointer-style head
(see policy.py) rather than by a fixed slot index - so unlike the old ATTACK_SLOTS
scheme, "which row" carries no meaning of its own and doesn't need to be stable
across ticks. do-nothing is folded in as a fixed all-zero-stats entity so one code
path handles every candidate uniformly.
"""

import numpy as np

# [is_tribe, is_nation, is_wilderness, is_do_nothing, tile_fraction, log_gold, log_troops]
ENTITY_FEATURES = 7

# Buffer-size cap, not a meaningful-identity cap like the old ATTACK_SLOTS - truncation
# just means "ignore the least-relevant excess candidates this one tick," not "this
# opponent permanently has no slot." Real candidate counts are expected to stay well
# under this given map/opponent scale so far.
MAX_ENTITIES = 16

# Sentinel distinct from any real targetID (string) or wilderness's target (None).
DO_NOTHING = object()


def build_entities(
    attack_mask: list, player_stats: dict, player_types: dict, total_land: float
) -> tuple[np.ndarray, list]:
    """Turn attackMask + current player stats into (features, targets).

    features: float32, shape (N, ENTITY_FEATURES) - N = live candidates this tick
        (bordering players not already under attack, wilderness if available) plus
        one for do-nothing (always legal, always last).
    targets: list of length N - what to pass to action_to_intents for each row,
        parallel to features.
    """
    rows = []
    targets = []
    for c in attack_mask:
        if c["alreadyAttacking"]:
            continue
        if c["targetID"] is not None:
            stats = player_stats.get(c["smallID"], {})
            ptype = player_types.get(c["smallID"], "BOT")
            rows.append([
                float(ptype == "BOT"),
                float(ptype == "NATION"),
                0.0,
                0.0,
                stats.get("tilesOwned", 0.0) / max(total_land, 1),
                float(np.log1p(stats.get("gold", 0.0))),
                float(np.log1p(stats.get("troops", 0.0))),
            ])
            targets.append(c["targetID"])
        else:
            rows.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            targets.append(None)

    rows.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    targets.append(DO_NOTHING)

    return np.array(rows, dtype=np.float32), targets


def pad_entities(features: np.ndarray, max_n: int = MAX_ENTITIES) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate (N, ENTITY_FEATURES) to a fixed (max_n, ENTITY_FEATURES) for
    batching, plus a bool mask (max_n,) marking real (non-padding) rows."""
    n = min(len(features), max_n)
    padded = np.zeros((max_n, ENTITY_FEATURES), dtype=np.float32)
    mask = np.zeros(max_n, dtype=bool)
    padded[:n] = features[:n]
    mask[:n] = True
    return padded, mask


def action_to_intents(target, agent_client_id: str) -> list[dict]:
    """Turn a chosen candidate's target (from `targets`, indexed by the sampled
    action) into an intents list for bridge.step()."""
    if target is DO_NOTHING:
        return []
    return [{"type": "attack", "targetID": target, "troops": None, "clientID": agent_client_id}]
