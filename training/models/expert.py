import os

import torch

BC_TAU = float(os.environ.get("OFM_BC_TAU", "0.4"))
EXPERT_SIZE = float(os.environ.get("OFM_BC_SIZE", "0.15"))


def expert_actions(scalar: torch.Tensor, entities: torch.Tensor, entity_mask: torch.Tensor, reserve_floor: float) -> torch.Tensor:
    """Scripted rule as a batched policy: once troop ratio >= BC_TAU, attack wilderness first,
    else the current focus, else the weakest Nation. Returns (B,) double actions."""
    is_do_nothing = entities[..., 3] > 0.5
    real = entity_mask & ~is_do_nothing
    wild = real & (entities[..., 2] > 0.5)
    nation = real & ~wild
    focus = nation & (entities[..., 14] > 0.5)
    neg_inf = torch.full_like(entities[..., 10], -1e9)
    score = torch.where(wild, torch.full_like(neg_inf, 3e3),
            torch.where(focus, torch.full_like(neg_inf, 2e3),
            torch.where(nation, -entities[..., 10], neg_inf)))
    target = score.argmax(dim=-1)
    do_nothing_idx = is_do_nothing.float().argmax(dim=-1)
    ratio = scalar[:, 1]
    gate = real.any(dim=-1) & (ratio >= BC_TAU) & (ratio >= reserve_floor)
    action = torch.where(gate, target.double() + EXPERT_SIZE, do_nothing_idx.double() + 0.5)
    return action
