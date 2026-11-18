"""Compares the attack gate's act probability with its KL reference curve, by troop ratio."""

import argparse

import numpy as np
import torch

from training.envs.owned_env import EnvConfig, OwnedVecEnv
from training.models.policy import CAN_ATTACK, RESERVE_FLOOR, Agent, drop_gold_inputs, load_padded
from training.models.train import ACT_EVERY, DEVICE, MAX_EPISODE_TICKS


def reference(ratio: np.ndarray) -> np.ndarray:
    return 0.005 + 0.495 / (1 + np.exp(-(ratio - 0.5) / 0.03))


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
                logits = agent.gate_head(torch.cat([hidden, t["scalar"]], dim=-1))
                p_act = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            ent, mask, scalar = obs["entities"], obs["entity_mask"], obs["scalar"]
            has_target = ((ent[..., CAN_ATTACK] > 0.5) & (ent[..., 3] < 0.5) & mask).any(-1)
            for k in range(envs):
                if has_target[k] and scalar[k][0] >= RESERVE_FLOOR:
                    rows.append((scalar[k][0], p_act[k]))
            env.step(action.cpu().numpy())
    r = np.array(rows, dtype=float)
    print(f"{len(r)} decisions where acting is legal")
    print("troop ratio    n    gate p_act (mean, p10/p50/p90)   reference")
    for lo, hi in ((0.2, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.85), (0.85, 1.01)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum():
            p = r[m, 1]
            print(f"{lo:.2f}-{hi:.2f} {int(m.sum()):6d}   {p.mean():.3f}  ({np.percentile(p, 10):.3f}/{np.median(p):.3f}/{np.percentile(p, 90):.3f})   {reference(r[m, 0]).mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--decisions", type=int, default=300)
    p.add_argument("--envs", type=int, default=8)
    a = p.parse_args()
    main(a.checkpoint, a.decisions, a.envs)
