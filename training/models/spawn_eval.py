"""Plays many games with a frozen checkpoint on all worker envs, logging each game's spawn layout and result."""

import argparse
import json
import os

import torch

from training.envs.owned_env import EnvConfig, OwnedVecEnv
from training.models.policy import Agent, drop_gold_inputs, load_padded
from training.models.train import ACT_EVERY, DEVICE, MAX_EPISODE_TICKS


def main(checkpoint: str, games: int, envs: int, out: str, nations: int, bots: int, difficulty: str, map_name: str) -> None:
    state = torch.load(checkpoint, map_location=DEVICE)
    cfg = EnvConfig(map=map_name, bots=bots, nations=nations, difficulties=[difficulty],
                    max_episode_ticks=MAX_EPISODE_TICKS, act_every=ACT_EVERY, gamma_step=1.0)
    done_games = 0
    with OwnedVecEnv(envs, cfg) as env, open(out, "a") as f:
        env.start()
        obs = env.obs
        agent = Agent.from_observation(obs["spatial"][0], obs["scalar"][0], obs["entities"][0]).to(DEVICE)
        load_padded(agent, drop_gold_inputs(state["model"]) if "gold_mean" in state else state["model"])
        agent.eval()
        while done_games < games:
            with torch.no_grad():
                action, *_ = agent.get_action_and_value(
                    spatial=torch.from_numpy(obs["spatial"]).to(DEVICE), scalar=torch.from_numpy(obs["scalar"]).to(DEVICE),
                    entities=torch.from_numpy(obs["entities"]).to(DEVICE), entity_mask=torch.from_numpy(obs["entity_mask"]).to(DEVICE),
                    action=None,
                )
            for info in env.step(action.cpu().numpy()):
                if info["done"]:
                    f.write(json.dumps({"won": bool(info["won"]), "kind": info["kind"], "ticks": info["end_ticks"],
                                        "peak": info["peak"], "spawn": info["spawn"]}) + "\n")
                    f.flush()
                    done_games += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--out", required=True)
    p.add_argument("--nations", type=int, default=10)
    p.add_argument("--bots", type=int, default=15)
    p.add_argument("--difficulty", default="Hard")
    p.add_argument("--map", dest="map_name", default="mid_plains")
    a = p.parse_args()
    main(a.checkpoint, a.games, a.envs, a.out, a.nations, a.bots, a.difficulty, a.map_name)
