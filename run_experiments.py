"""Train PPO and CMA-ES on the same navigation task and reproduce the README results.

    python run_experiments.py --ppo-steps 1000000 --es-steps 3000000

Outputs: results/*.json, models/*.pt|.npy, figures/*.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robotnav import neuroevolution, ppo
from robotnav.baselines import PotentialFieldController
from robotnav.env import ARENA, GOAL_RADIUS, NavEnv
from robotnav.evaluation import EVAL_SEED0, evaluate, trajectories

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
})

# validation maps used during training (different from the final test maps)
VAL_SEED0 = 5_000_000
SCENARIOS = {
    "standard (4-8 obstacles)": {},
    "dense (10-14 obstacles)": {"n_obstacles": (10, 14)},
    "noisy lidar (sigma = 10 cm)": {"sensor_noise": 0.10},
}


def main(a):
    out, figs, models = Path("results"), Path("figures"), Path("models")
    for p in (out, figs, models):
        p.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    val = lambda fn: evaluate(fn, n_episodes=50, seed0=VAL_SEED0)
    make_env = lambda seed: NavEnv()

    # ---- PPO --------------------------------------------------------------------
    t0 = time.time()
    model, ppo_hist = ppo.train(make_env, ppo.PPOConfig(total_steps=a.ppo_steps, seed=a.seed),
                                evaluate=val, eval_every=10)
    ppo_time = time.time() - t0
    torch.save(model.state_dict(), models / "ppo.pt")

    # ---- CMA-ES -------------------------------------------------------------------
    t0 = time.time()
    es_policy, es_hist = neuroevolution.train(make_env, total_steps=a.es_steps, seed=a.seed,
                                              evaluate=val, eval_every=5)
    es_time = time.time() - t0
    np.save(models / "cmaes.npy", es_policy.params)

    # ---- final test on held-out maps --------------------------------------------------
    controllers = {
        "Potential field (hand-designed)": PotentialFieldController().act,
        "CMA-ES neuroevolution": es_policy.act,
        "PPO (from scratch)": model.act,
    }
    table = {}
    for name, fn in controllers.items():
        table[name] = {sc: evaluate(fn, n_episodes=a.test_episodes, seed0=EVAL_SEED0, **kw)
                       for sc, kw in SCENARIOS.items()}
        print(name, {sc: round(r["success"], 3) for sc, r in table[name].items()})

    n_ppo = sum(p.numel() for p in model.actor.parameters())
    results = {
        "test_episodes_per_scenario": a.test_episodes, "seed": a.seed,
        "ppo": {"env_steps": a.ppo_steps, "train_seconds": ppo_time, "actor_params": n_ppo,
                "history": ppo_hist},
        "cmaes": {"env_steps": es_hist[-1]["env_steps"] if es_hist else None, "train_seconds": es_time,
                  "params": es_policy.n_params, "history": es_hist},
        "test": table,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    plot_curves(ppo_hist, es_hist, table, figs)
    plot_trajectories(controllers, figs)


def plot_curves(ppo_hist, es_hist, table, figs):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    base = table["Potential field (hand-designed)"]["standard (4-8 obstacles)"]["success"]
    ax.axhline(base, color=INK2, linewidth=1.2, linestyle="--")
    x_max = max(h["env_steps"] for h in ppo_hist + es_hist) / 1e6
    ax.text(x_max * 0.55, base - 0.06, "hand-designed potential field", color=INK2, fontsize=8)
    for hist, color, label in ((ppo_hist, BLUE, "PPO"), (es_hist, ORANGE, "CMA-ES")):
        x = [h["env_steps"] / 1e6 for h in hist]
        y = [h["success"] for h in hist]
        ax.plot(x, y, color=color, label=label)
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("success rate (50 validation maps)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Learning curves: sample efficiency of PPO vs neuroevolution", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "learning_curves.png"); plt.close(fig)

    # grouped bars: success rate per scenario
    names = list(table)
    scen = list(SCENARIOS)
    colors = [INK2, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.26
    for i, (n, c) in enumerate(zip(names, colors)):
        vals = [table[n][s]["success"] for s in scen]
        bars = ax.bar(np.arange(len(scen)) + (i - 1) * w, vals, width=w - 0.02, color=c, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(np.arange(len(scen)), scen)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate (held-out maps)")
    ax.set_title("Generalisation and robustness on unseen maps", color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=3)
    fig.tight_layout(); fig.savefig(figs / "test_success.png"); plt.close(fig)


def plot_trajectories(controllers, figs, seeds=(EVAL_SEED0 + 1, EVAL_SEED0 + 7, EVAL_SEED0 + 12)):
    fig, axes = plt.subplots(len(controllers), len(seeds), figsize=(7.5, 7.5))
    for row, (name, fn) in enumerate(controllers.items()):
        for col, ep in enumerate(trajectories(fn, seeds, n_obstacles=(10, 14))):
            ax = axes[row, col]
            for (x, y), r in zip(ep["obs_xy"], ep["obs_r"]):
                ax.add_patch(plt.Circle((x, y), r, color=INK2, alpha=0.35, linewidth=0))
            ax.add_patch(plt.Circle(ep["goal"], GOAL_RADIUS, color=AQUA, alpha=0.8, linewidth=0))
            p = ep["path"]
            ax.plot(p[:, 0], p[:, 1], color=BLUE, linewidth=1.5)
            ax.plot(*p[0], marker="o", color=INK, markersize=4)
            ax.set_xlim(0, ARENA); ax.set_ylim(0, ARENA); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(INK2)
            ax.set_title(ep["event"], fontsize=8, color=INK)
            if col == 0:
                ax.set_ylabel(name, fontsize=8, color=INK)
    fig.suptitle("Trajectories on unseen dense maps (start = black dot, goal = green)", color=INK, fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "trajectories.png"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ppo-steps", type=int, default=1_000_000)
    p.add_argument("--es-steps", type=int, default=3_000_000)
    p.add_argument("--test-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
