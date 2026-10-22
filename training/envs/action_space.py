"""Entity-based action space built on harness.ts's attackMask.

Each legal candidate this tick (bordering players, wilderness, do-nothing) becomes one
row of an entity feature array, padded to MAX_ENTITIES. The network scores rows by their
features (policy.py), so row order carries no meaning. do-nothing is an all-zero row.
"""

import numpy as np

# [is_tribe, is_nation, is_wilderness, is_do_nothing, tile_fraction, log_gold, troop_ratio,
#  attacking_me_ratio, already_attacking, agent_ratio - troop_ratio]
ENTITY_FEATURES = 10
MAX_ENTITIES = 16  # buffer size; excess candidates are dropped for that tick

DO_NOTHING = object()  # sentinel target, distinct from a real id or wilderness (None)


def build_entities(
    attack_mask: list, player_stats: dict, player_types: dict, total_land: float,
    agent_ratio: float = 0.0,
) -> tuple[np.ndarray, list]:
    """Returns (features, targets): one row per candidate plus a final do-nothing row,
    and the matching target for action_to_intents."""
    rows = []
    targets = []
    for c in attack_mask:
        # re-attacking a target merges into the existing attack, so these rows stay legal
        already_attacking = float(c["alreadyAttacking"])
        if c["targetID"] is not None:
            stats = player_stats.get(c["smallID"], {})
            ptype = player_types.get(c["smallID"], "BOT")
            max_troops = max(stats.get("maxTroops", 1.0), 1)
            troop_ratio = stats.get("troops", 0.0) / max_troops
            # share of this player's army currently attacking the agent
            attacking_me_ratio = c["attackingMeTroops"] / max_troops
            rows.append([
                float(ptype == "BOT"),
                float(ptype == "NATION"),
                0.0,
                0.0,
                stats.get("tilesOwned", 0.0) / max(total_land, 1),
                float(np.log1p(stats.get("gold", 0.0))),
                troop_ratio,
                attacking_me_ratio,
                already_attacking,
                agent_ratio - troop_ratio,
            ])
            targets.append(c["targetID"])
        else:
            rows.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, already_attacking, 0.0])
            targets.append(None)

    rows.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    targets.append(DO_NOTHING)

    return np.array(rows, dtype=np.float32), targets


def pad_entities(features: np.ndarray, max_n: int = MAX_ENTITIES) -> tuple[np.ndarray, np.ndarray]:
    """Pads or truncates to (max_n, ENTITY_FEATURES); the mask marks real rows."""
    n = min(len(features), max_n)
    padded = np.zeros((max_n, ENTITY_FEATURES), dtype=np.float32)
    mask = np.zeros(max_n, dtype=bool)
    padded[:n] = features[:n]
    mask[:n] = True
    return padded, mask


def action_to_intents(target, agent_client_id: str, troops: float | None = None) -> list[dict]:
    """Intents list for bridge.step(); `troops` is the attack size."""
    if target is DO_NOTHING:
        return []
    return [{"type": "attack", "targetID": target, "troops": troops, "clientID": agent_client_id}]
