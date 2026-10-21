"""Evaluation on held-out random maps (seeds never seen during training)."""
from __future__ import annotations

import numpy as np

from .env import NavEnv

EVAL_SEED0 = 10_000_000  # training never uses seeds this large


def evaluate(policy_fn, n_episodes: int = 100, seed0: int = EVAL_SEED0, **env_kwargs) -> dict:
    env = NavEnv(**env_kwargs)
    events, lengths, returns = [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + i)
        total, n = 0.0, 0
        while True:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            total += r
            n += 1
            if term or trunc:
                break
        events.append(info["event"])
        lengths.append(n)
        returns.append(total)
    events = np.array(events)
    ok = events == "success"
    return {
        "success": float(ok.mean()),
        "collision": float((events == "collision").mean()),
        "timeout": float((events == "timeout").mean()),
        "mean_return": float(np.mean(returns)),
        "mean_steps_when_success": float(np.mean(np.array(lengths)[ok])) if ok.any() else None,
    }


def trajectories(policy_fn, seeds, **env_kwargs):
    """Run episodes and keep the map + path, for plotting."""
    env = NavEnv(**env_kwargs)
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            obs, _, term, trunc, info = env.step(policy_fn(obs))
            if term or trunc:
                break
        out.append({"obs_xy": env.obs_xy.copy(), "obs_r": env.obs_r.copy(), "goal": env.goal.copy(),
                    "path": np.array(env.path), "event": info["event"]})
    return out
