"""Neuroevolution: optimise the weights of a small neural controller with CMA-ES.

No gradient, no value function: each candidate weight vector is a controller
that is simply *tested* on a few random maps; its fitness is the mean return.
CMA-ES (Hansen, 2001) adapts a multivariate Gaussian over the weights towards
the best candidates. This is the classic approach of evolutionary robotics.
"""
from __future__ import annotations

import time

import cma
import numpy as np


class MLPPolicy:
    """obs -> tanh hidden layer -> tanh actions. Weights stored in one flat vector."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 32):
        self.shapes = [(obs_dim, hidden), (hidden,), (hidden, act_dim), (act_dim,)]
        self.n_params = sum(int(np.prod(s)) for s in self.shapes)
        self.params = np.zeros(self.n_params)

    def set(self, flat: np.ndarray) -> "MLPPolicy":
        self.params = np.asarray(flat, dtype=np.float64)
        return self

    def _unpack(self):
        out, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(self.params[i:i + n].reshape(s))
            i += n
        return out

    def act(self, obs: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self._unpack()
        return np.tanh(np.tanh(obs @ w1 + b1) @ w2 + b2)


def rollout(env, policy_fn, seed: int) -> tuple[float, int, str]:
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(policy_fn(obs))
        total += r
        steps += 1
        if term or trunc:
            return total, steps, info["event"]


def train(make_env, total_steps: int = 2_000_000, episodes_per_eval: int = 4, hidden: int = 32,
          sigma0: float = 0.5, seed: int = 0, evaluate=None, eval_every: int = 5, log=print):
    env = make_env(seed)
    policy = MLPPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden)
    es = cma.CMAEvolutionStrategy(np.zeros(policy.n_params), sigma0,
                                  {"seed": seed + 1, "verbose": -9})
    rng = np.random.default_rng(seed)
    history, steps, gen, t0 = [], 0, 0, time.time()
    while steps < total_steps:
        gen += 1
        candidates = es.ask()
        # common random maps for the whole population -> fair comparison between candidates
        map_seeds = rng.integers(0, 1_000_000, size=episodes_per_eval)
        losses = []
        for w in candidates:
            policy.set(w)
            returns = []
            for s in map_seeds:
                ret, n, _ = rollout(env, policy.act, int(s))
                returns.append(ret)
                steps += n
            losses.append(-np.mean(returns))              # CMA-ES minimises
        es.tell(candidates, losses)
        if evaluate is not None and gen % eval_every == 0:
            res = evaluate(policy.set(es.result.xfavorite).act)  # mean of the search distribution
            res["env_steps"] = steps
            history.append(res)
            log(f"[CMA-ES] gen {gen}  steps {steps:>8}  best fitness {-min(losses):6.2f}  "
                f"success {res['success']:.2f}  ({time.time() - t0:.0f}s)")
    policy.set(es.result.xfavorite)
    return policy, history
