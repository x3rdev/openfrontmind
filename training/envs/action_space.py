"""Entity-based action space built on harness.ts's attackMask.

Each legal candidate this tick (bordering players, wilderness, do-nothing) becomes one
row of an entity feature array, padded to MAX_ENTITIES. The network scores rows by their
features (policy.py), so row order carries no meaning. do-nothing is an all-zero row.
"""

import os

import numpy as np

ABLATE = set(os.environ.get("OFM_ABLATE", "").split(","))  # temporary test switch

# [is_tribe, is_nation, is_wilderness, is_do_nothing, tile_fraction, log_gold, troop_ratio,
#  attacking_me_ratio, already_attacking, agent_ratio - troop_ratio,
#  log(their troops / my troops), log(their troops per tile / mine)]
ENTITY_FEATURES = 16
MAX_ENTITIES = 16  # buffer size; excess candidates are dropped for that tick

DO_NOTHING = object()  # sentinel target, distinct from a real id or wilderness (None)


def build_entities(
    attack_mask: list, player_stats: dict, player_types: dict, total_land: float,
    agent_ratio: float = 0.0, agent_troops: float = 0.0, agent_tiles: float = 0.0,
    memory: dict | None = None,
) -> tuple[np.ndarray, list]:
    """Returns (features, targets): one row per candidate plus a final do-nothing row,
    and the matching target for action_to_intents.

    Candidates are sorted by smallID (wilderness first) so a target keeps its row.
    `memory` maps a target key (smallID or None) to its persistent features."""
    memory = memory or {}
    attack_mask = sorted(attack_mask, key=lambda c: -1 if c["targetID"] is None else c["smallID"])
    rows = []
    targets = []
    agent_density = agent_troops / max(agent_tiles, 1)
    for c in attack_mask:
        # re-attacking a target merges into the existing attack, so these rows stay legal
        already_attacking = float(c["alreadyAttacking"])
        committed = min(c.get("outgoingTroops", 0.0) / max(agent_troops, 1.0), 1.0)
        persist = [*memory.get(c["smallID"], (0.0, 1.0, 0.0)), committed]
        if "persist" in ABLATE:
            persist = [0.0] * 4
        if c["targetID"] is not None:
            stats = player_stats.get(c["smallID"], {})
            ptype = player_types.get(c["smallID"], "BOT")
            max_troops = max(stats.get("maxTroops", 1.0), 1)
            troop_ratio = stats.get("troops", 0.0) / max_troops
            # share of this player's army currently attacking the agent
            attacking_me_ratio = c["attackingMeTroops"] / max_troops
            their_troops = stats.get("troops", 0.0)
            their_density = their_troops / max(stats.get("tilesOwned", 0.0), 1)
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
                float(np.log((their_troops + 1) / (agent_troops + 1))),
                float(np.log((their_density + 1) / (agent_density + 1))),
                *persist,
            ])
            targets.append(c["targetID"])
        else:
            rows.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, already_attacking, 0.0, 0.0, 0.0, *persist])
            targets.append(None)

    rows.append([0.0, 0.0, 0.0, 1.0] + [0.0] * (ENTITY_FEATURES - 4))
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
