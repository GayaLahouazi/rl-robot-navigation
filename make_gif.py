"""Animated demo of the trained PPO policy on unseen maps (figures/demo.gif).

    python make_gif.py            # uses models/ppo.pt, first held-out evaluation seeds (not cherry-picked)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from robotnav.env import ARENA, GOAL_RADIUS, MAX_RANGE, N_RAYS, ROBOT_RADIUS, NavEnv
from robotnav.evaluation import EVAL_SEED0
from robotnav.ppo import ActorCritic


def rollout(model, seed, **env_kwargs):
    env = NavEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    frames = []
    while True:
        frames.append((env.pos.copy(), env.theta, env._lidar().copy()))
        obs, _, term, trunc, info = env.step(model.act(obs))
        if term or trunc:
            frames.append((env.pos.copy(), env.theta, env._lidar().copy()))
            break
    return {"obs_xy": env.obs_xy.copy(), "obs_r": env.obs_r.copy(), "goal": env.goal.copy(),
            "frames": frames, "event": info["event"]}


def main(a):
    model = ActorCritic(N_RAYS + 4, 2)
    model.load_state_dict(torch.load("models/ppo.pt", map_location="cpu"))
    episodes = [rollout(model, EVAL_SEED0 + i) for i in range(a.episodes)]
    held = []  # (episode, step) shown in each GIF frame; the last frame of each episode is held a little
    for e, ep in enumerate(episodes):
        ks = list(range(0, len(ep["frames"]), a.stride)) + [len(ep["frames"]) - 1]
        held += [(e, k) for k in ks] + [(e, len(ep["frames"]) - 1)] * 8

    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)
    angles = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)

    def draw(i):
        e, k = held[i]
        ep = episodes[e]
        ax.clear()
        for (x, y), r in zip(ep["obs_xy"], ep["obs_r"]):
            ax.add_patch(plt.Circle((x, y), r, color="#6b7280", alpha=0.45, linewidth=0))
        ax.add_patch(plt.Circle(ep["goal"], GOAL_RADIUS, color="#10b981", alpha=0.85, linewidth=0))
        path = np.array([f[0] for f in ep["frames"][: k + 1]])
        ax.plot(path[:, 0], path[:, 1], color="#2563eb", linewidth=1.6)
        pos, th, lidar = ep["frames"][k]
        for ang, d in zip(angles, lidar):
            end = pos + d * np.array([np.cos(th + ang), np.sin(th + ang)])
            ax.plot([pos[0], end[0]], [pos[1], end[1]], color="#f59e0b",
                    linewidth=0.7, alpha=0.9 if d < MAX_RANGE else 0.35)
        ax.add_patch(plt.Circle(pos, ROBOT_RADIUS, color="#111827", linewidth=0))
        ax.plot([pos[0], pos[0] + 0.45 * np.cos(th)], [pos[1], pos[1] + 0.45 * np.sin(th)], color="#111827", lw=2)
        ax.set_xlim(0, ARENA); ax.set_ylim(0, ARENA); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        done = k == len(ep["frames"]) - 1
        status = f" - {ep['event']}" if done else ""
        ax.set_title(f"PPO on unseen map {e + 1}/{len(episodes)}   t = {k * 0.1:.1f} s{status}", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(held))
    anim.save(a.out, writer=PillowWriter(fps=a.fps))
    print(a.out, len(held), "frames", [ep["event"] for ep in episodes])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", default="figures/demo.gif")
    main(p.parse_args())
