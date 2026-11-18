"""Entity-based action space built on harness.ts's attackMask.

Every living player, wilderness (when bordering) and do-nothing becomes one row of an
entity feature array, padded to MAX_ENTITIES. The network scores rows by their features
(policy.py), so row order carries no meaning. do-nothing is row 0.
"""


import numpy as np

# [is_tribe, is_nation, is_wilderness, is_do_nothing, tile_fraction, troop_ratio,
#  attacking_me_ratio, already_attacking, agent_ratio - troop_ratio,
#  log(their troops / my troops), log(their troops per tile / mine),
#  4 persistent target-memory features,
#  can_attack, can_request, allied, requested_me, is_traitor, traitor_ticks_left,
#  log1p(betrayals), log1p(n_allies), allies' tile fraction,
#  tile growth over the last 5 and 20 decisions]
ENTITY_FEATURES = 26
MAX_ENTITIES = 32  # buffer size; lowest-priority rows are dropped when exceeded
TRAITOR_TICKS = 300.0

DO_NOTHING = object()  # sentinel target, distinct from a real id or wilderness (None)


def build_entities(
    attack_mask: list, player_stats: dict, player_types: dict, total_land: float,
    agent_ratio: float = 0.0, agent_troops: float = 0.0, agent_tiles: float = 0.0,
    memory: dict | None = None, momentum: dict | None = None,
) -> tuple[np.ndarray, list]:
    """Returns (features, targets): do-nothing first, then one row per candidate, and the
    matching target for action_to_intents.

    Candidates are sorted by smallID (wilderness first) so a target keeps its row. When
    there are more than MAX_ENTITIES - 1, the least relevant are dropped.
    `memory` maps a target key (smallID or None) to its persistent features,
    `momentum` a smallID to its (short, long) tile growth."""
    memory = memory or {}
    momentum = momentum or {}
    agent_density = agent_troops / max(agent_tiles, 1)
    candidates = []
    for c in attack_mask:
        # re-attacking a target merges into the existing attack, so these rows stay legal
        already_attacking = float(c["alreadyAttacking"])
        committed = min(c.get("outgoingTroops", 0.0) / max(agent_troops, 1.0), 1.0)
        persist = [*memory.get(c["smallID"], (0.0, 1.0, 0.0)), committed]
        if c["targetID"] is not None:
            stats = player_stats.get(c["smallID"], {})
            ptype = player_types.get(c["smallID"], "BOT")
            max_troops = max(stats.get("maxTroops", 1.0), 1)
            troop_ratio = stats.get("troops", 0.0) / max_troops
            # share of this player's army currently attacking the agent
            attacking_me_ratio = c["attackingMeTroops"] / max_troops
            their_troops = stats.get("troops", 0.0)
            their_density = their_troops / max(stats.get("tilesOwned", 0.0), 1)
            ally_tiles = sum(player_stats.get(a, {}).get("tilesOwned", 0.0) for a in c["allies"])
            row = [
                float(ptype == "BOT"),
                float(ptype == "NATION"),
                0.0,
                0.0,
                stats.get("tilesOwned", 0.0) / max(total_land, 1),
                troop_ratio,
                attacking_me_ratio,
                already_attacking,
                agent_ratio - troop_ratio,
                float(np.log((their_troops + 1) / (agent_troops + 1))),
                float(np.log((their_density + 1) / (agent_density + 1))),
                *persist,
                float(c["canAttack"]),
                float(c["canRequest"]),
                float(c["allied"]),
                float(c["requestedMe"]),
                float(c["traitorTicks"] > 0),
                min(c["traitorTicks"] / TRAITOR_TICKS, 1.0),
                float(np.log1p(c["betrayals"])),
                float(np.log1p(len(c["allies"]))),
                ally_tiles / max(total_land, 1),
                *momentum.get(c["smallID"], (0.0, 0.0)),
            ]
            relevance = 4.0 * row[15] + 3.0 * (attacking_me_ratio > 0) + 2.0 * (row[17] + row[18]) + row[4]
        else:
            row = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, already_attacking, 0.0, 0.0, 0.0, *persist,
                   1.0] + [0.0] * (ENTITY_FEATURES - 16)
            relevance = 1e9
        candidates.append((relevance, c["smallID"], row, c["targetID"]))

    if len(candidates) > MAX_ENTITIES - 1:
        candidates = sorted(candidates, key=lambda x: -x[0])[:MAX_ENTITIES - 1]
    candidates.sort(key=lambda x: -1 if x[1] is None else x[1])

    rows = [[0.0, 0.0, 0.0, 1.0] + [0.0] * (ENTITY_FEATURES - 4)]
    targets = [DO_NOTHING]
    for _, _, row, target in candidates:
        rows.append(row)
        targets.append(target)
    return np.array(rows, dtype=np.float32), targets


def pad_entities(features: np.ndarray, max_n: int = MAX_ENTITIES) -> tuple[np.ndarray, np.ndarray]:
    """Pads or truncates to (max_n, ENTITY_FEATURES); the mask marks real rows."""
    n = min(len(features), max_n)
    padded = np.zeros((max_n, ENTITY_FEATURES), dtype=np.float32)
    mask = np.zeros(max_n, dtype=bool)
    padded[:n] = features[:n]
    mask[:n] = True
    return padded, mask


def action_to_intents(target, agent_client_id: str, troops: float | None = None, diplomacy: tuple | None = None) -> list[dict]:
    """Intents list for bridge.step(); `troops` is the attack size.
    `diplomacy` is (kind, player_id) with kind "request" or "break"."""
    intents = []
    if target is not DO_NOTHING:
        intents.append({"type": "attack", "targetID": target, "troops": troops, "clientID": agent_client_id})
    if diplomacy is not None:
        kind, player_id = diplomacy
        intents.append({"type": "allianceRequest" if kind == "request" else "breakAlliance",
                        "recipient": player_id, "clientID": agent_client_id})
    return intents


def diplomacy_from_code(code: int, targets: list) -> tuple | None:
    """Maps policy.py's diplomacy code (-1 none, r request row r, N + r break row r) to (kind, player_id)."""
    if code < 0:
        return None
    row = code % MAX_ENTITIES
    if row >= len(targets) or targets[row] is None or targets[row] is DO_NOTHING:
        return None
    return ("request" if code < MAX_ENTITIES else "break", targets[row])
