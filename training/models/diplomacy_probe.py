"""Compares the diplomacy gate's act probability with its prior, split by state."""

import argparse

import numpy as np
import torch

from training.envs.owned_env import EnvConfig, OwnedVecEnv
from training.models.policy import (
    ALLIED, CAN_REQUEST, DIPLOMACY_PRIOR, DIPLOMACY_PRIOR_PENDING, REQUESTED_ME, Agent, drop_gold_inputs, load_padded,
)
from training.models.train import ACT_EVERY, DEVICE, MAX_EPISODE_TICKS


def main(checkpoint: str, decisions: int, envs: int) -> None:
    state = torch.load(checkpoint, map_location=DEVICE)
    cfg = EnvConfig(map="mid_plains", bots=15, nations=10, difficulties=["Hard"],
                    max_episode_ticks=MAX_EPISODE_TICKS, act_every=ACT_EVERY, gamma_step=1.0)
    rows = []
    with OwnedVecEnv(envs, cfg) as env:
        env.start()
        obs = env.obs
        agent = Agent.from_observation(obs["spatial"][0], obs["scalar"][0], obs["entities"][0]).to(DEVICE)
        load_padded(agent, drop_gold_inputs(state["model"]) if "gold_mean" in state else state["model"])
        agent.eval()
        for _ in range(decisions):
            t = {k: torch.from_numpy(obs[k]).to(DEVICE) for k in ("spatial", "scalar", "entities", "entity_mask")}
            with torch.no_grad():
                action, *_ = agent.get_action_and_value(**t, action=None)
                hidden, _ = agent.backbone(t["spatial"], t["scalar"], t["entities"], t["entity_mask"])
                logits = agent.diplomacy_gate(torch.cat([hidden, t["scalar"]], dim=-1))
                p_act = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            ent, mask, scalar = obs["entities"], obs["entity_mask"], obs["scalar"]
            can_req = ((ent[..., CAN_REQUEST] > 0.5) & mask).any(-1)
            can_brk = ((ent[..., ALLIED] > 0.5) & mask).any(-1)
            pending = (((ent[..., CAN_REQUEST] > 0.5) & (ent[..., REQUESTED_ME] > 0.5)) & mask).any(-1)
            for k in range(envs):
                if can_req[k] or can_brk[k]:
                    rows.append((p_act[k], pending[k], can_brk[k], scalar[k][0]))
            env.step(action.cpu().numpy())
    r = np.array(rows, dtype=float)
    print(f"{len(r)} decisions with a legal diplomacy action (of {decisions * envs})")
    print(f"prior {DIPLOMACY_PRIOR}, pending prior {DIPLOMACY_PRIOR_PENDING}")

    def show(name, m):
        if m.sum():
            p = r[m, 0]
            print(f"{name:34s} n={int(m.sum()):6d} mean p_act={p.mean():.3f} p10={np.percentile(p, 10):.3f} p50={np.median(p):.3f} p90={np.percentile(p, 90):.3f}")

    pend, allied, ratio = r[:, 1] > 0.5, r[:, 2] > 0.5, r[:, 3]
    show("all", np.ones(len(r), bool))
    show("request pending", pend)
    show("no request pending", ~pend)
    show("have an ally", allied)
    show("no ally", ~allied)
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)):
        show(f"troop ratio {lo}-{hi}, none pending", ~pend & (ratio >= lo) & (ratio < hi))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--decisions", type=int, default=300)
    p.add_argument("--envs", type=int, default=8)
    a = p.parse_args()
    main(a.checkpoint, a.decisions, a.envs)
