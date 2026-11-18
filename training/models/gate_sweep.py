"""Frozen-policy sweep: replaces the attack gate with a reference-shaped curve and plays games for each variant."""

import argparse
import json
import math

import torch

from training.envs.owned_env import EnvConfig, OwnedVecEnv
from training.models.policy import Agent, drop_gold_inputs, load_padded
from training.models.train import ACT_EVERY, DEVICE, MAX_EPISODE_TICKS

WIDTH = 0.03
FLOOR = 0.005
VARIANTS = {  # name: (center, cap), None keeps the learned gate
    "own": None,
    "c0.40": (0.40, 0.5), "c0.45": (0.45, 0.5), "c0.50": (0.50, 0.5), "c0.55": (0.55, 0.5), "c0.60": (0.60, 0.5),
    "cap0.25": (0.50, 0.25), "cap0.75": (0.50, 0.75),
}
PATIENCE = {"wait25": 25, "wait50": 50, "wait100": 100}  # learned gate, waits N decisions unless attacked
INCOMING_EPS = 0.001


def curve(center: float, cap: float):
    return lambda ratio: FLOOR + (cap - FLOOR) * torch.sigmoid((ratio - center) / WIDTH)


def run_variant(agent, name: str, games: int, envs: int, cfg: EnvConfig) -> dict:
    spec = VARIANTS.get(name)
    agent.gate_override = None if spec is None else curve(*spec)
    steps = torch.zeros(envs, device=DEVICE)
    agent.force_wait = None
    if name in PATIENCE:
        agent.force_wait = lambda scalar: (steps < PATIENCE[name]) & (scalar[:, 3] <= INCOMING_EPS)
    wins = finished = 0
    kinds = {}
    with OwnedVecEnv(envs, cfg) as env:
        env.start()
        obs = env.obs
        while finished < games:
            with torch.no_grad():
                action, *_ = agent.get_action_and_value(
                    spatial=torch.from_numpy(obs["spatial"]).to(DEVICE), scalar=torch.from_numpy(obs["scalar"]).to(DEVICE),
                    entities=torch.from_numpy(obs["entities"]).to(DEVICE), entity_mask=torch.from_numpy(obs["entity_mask"]).to(DEVICE),
                    action=None,
                )
            steps += 1
            for i, info in enumerate(env.step(action.cpu().numpy())):
                if info["done"]:
                    steps[i] = 0
                    finished += 1
                    wins += info["won"]
                    kinds[info["kind"]] = kinds.get(info["kind"], 0) + 1
    rate = wins / finished
    return {"variant": name, "games": finished, "wins": int(wins), "win_rate": rate,
            "se": math.sqrt(rate * (1 - rate) / finished), "kinds": kinds}


def main(checkpoint: str, games: int, envs: int, variants: list[str], out: str) -> None:
    state = torch.load(checkpoint, map_location=DEVICE)
    cfg = EnvConfig(map="mid_plains", bots=15, nations=10, difficulties=["Hard"],
                    max_episode_ticks=MAX_EPISODE_TICKS, act_every=ACT_EVERY, gamma_step=1.0)
    with OwnedVecEnv(1, cfg) as probe:
        probe.start()
        agent = Agent.from_observation(probe.obs["spatial"][0], probe.obs["scalar"][0], probe.obs["entities"][0]).to(DEVICE)
    load_padded(agent, drop_gold_inputs(state["model"]) if "gold_mean" in state else state["model"])
    agent.eval()
    for name in variants:
        result = run_variant(agent, name, games, envs, cfg)
        print(json.dumps(result), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.checkpoint, a.games, a.envs, a.variants.split(","), a.out)
