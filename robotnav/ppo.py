"""Proximal Policy Optimization (Schulman et al., 2017), written from scratch.

Single file, no RL library: every step of the algorithm is visible.
    1. collect `n_steps` transitions in `n_envs` environments with the current policy
    2. compute advantages with GAE (Generalized Advantage Estimation)
    3. update the actor-critic for a few epochs with the clipped surrogate loss
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 256            # per env and per update -> batch of 2048 transitions
    epochs: int = 10
    minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 64
    seed: int = 0


def layer(n_in, n_out, std=np.sqrt(2)):
    l = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


class ActorCritic(nn.Module):
    """Gaussian policy pi(a|s) = N(mu(s), sigma) and value function V(s)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.critic = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(),
                                    layer(hidden, 1, std=1.0))
        self.actor = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(),
                                   layer(hidden, act_dim, std=0.01))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def dist(self, obs):
        return Normal(self.actor(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Used at test time: the mean action, no exploration noise."""
        o = torch.as_tensor(obs, dtype=torch.float32)
        a = self.actor(o) if deterministic else self.dist(o).sample()
        return a.clamp(-1, 1).numpy()


def train(make_env, cfg: PPOConfig, evaluate=None, eval_every: int = 10, log=print):
    """make_env(seed) -> env. evaluate(policy_fn) -> dict, called every `eval_every` updates.
    Returns the trained model and a history list of evaluation results."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    envs = [make_env(cfg.seed * 1000 + i) for i in range(cfg.n_envs)]
    obs = np.stack([e.reset(seed=cfg.seed * 1000 + i)[0] for i, e in enumerate(envs)])
    obs_dim, act_dim = envs[0].observation_space.shape[0], envs[0].action_space.shape[0]
    model = ActorCritic(obs_dim, act_dim, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.n_steps, cfg.n_envs
    buf_obs = torch.zeros(T, N, obs_dim)
    buf_act = torch.zeros(T, N, act_dim)
    buf_logp, buf_rew, buf_done, buf_val = (torch.zeros(T, N) for _ in range(4))

    n_updates = cfg.total_steps // (T * N)
    history, steps, t0 = [], 0, time.time()
    for update in range(1, n_updates + 1):
        for g in opt.param_groups:                           # linear learning-rate annealing
            g["lr"] = cfg.lr * (1 - (update - 1) / n_updates)

        # ---- 1. rollout ------------------------------------------------------
        for t in range(T):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = model.dist(o)
                a = d.sample()
                buf_logp[t] = d.log_prob(a).sum(-1)
                buf_val[t] = model.value(o)
            buf_obs[t], buf_act[t] = o, a
            for i, env in enumerate(envs):
                o2, r, term, trunc, _ = env.step(a[i].numpy())
                buf_rew[t, i] = r
                # time-limit truncation: bootstrap with V(s') instead of treating it as a terminal state
                if trunc and not term:
                    with torch.no_grad():
                        buf_rew[t, i] += cfg.gamma * model.value(torch.as_tensor(o2, dtype=torch.float32)).item()
                buf_done[t, i] = float(term or trunc)
                if term or trunc:
                    o2, _ = env.reset()
                obs[i] = o2
        steps += T * N

        # ---- 2. GAE advantages ------------------------------------------------
        with torch.no_grad():
            next_val = model.value(torch.as_tensor(obs, dtype=torch.float32))
        adv = torch.zeros(T, N)
        last = torch.zeros(N)
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else buf_val[t + 1]
            not_done = 1.0 - buf_done[t]
            delta = buf_rew[t] + cfg.gamma * nv * not_done - buf_val[t]
            last = delta + cfg.gamma * cfg.gae_lambda * not_done * last
            adv[t] = last
        ret = adv + buf_val

        # ---- 3. clipped-surrogate update ------------------------------------------
        b_obs, b_act = buf_obs.reshape(-1, obs_dim), buf_act.reshape(-1, act_dim)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        idx = np.arange(T * N)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, T * N, cfg.minibatch):
                mb = idx[s:s + cfg.minibatch]
                d = model.dist(b_obs[mb])
                logp = d.log_prob(b_act[mb]).sum(-1)
                ratio = (logp - b_logp[mb]).exp()
                a_mb = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = -torch.min(ratio * a_mb, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_mb).mean()
                v_loss = 0.5 * (model.value(b_obs[mb]) - b_ret[mb]).pow(2).mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        if evaluate is not None and (update % eval_every == 0 or update == n_updates):
            res = evaluate(model.act)
            res["env_steps"] = steps
            history.append(res)
            log(f"[PPO] update {update}/{n_updates}  steps {steps:>8}  success {res['success']:.2f}  "
                f"collision {res['collision']:.2f}  ({time.time() - t0:.0f}s)")
    return model, history
