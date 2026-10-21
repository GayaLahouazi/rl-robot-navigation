"""2D mobile-robot navigation environment (Gymnasium API).

A differential-drive robot must reach a goal in a square arena filled with
random circular obstacles, using only:
    - N_RAYS range sensors (a 2D lidar), max range MAX_RANGE
    - the distance and the bearing to the goal (as from a GPS / odometry)

Every reset draws a NEW random map, so the agent cannot memorise a map:
it has to learn a general navigation behaviour (reactive controller).

Observation (N_RAYS + 4 values, all in [-1, 1] or [0, 1]):
    lidar distances / MAX_RANGE, goal distance / arena diagonal,
    cos(bearing), sin(bearing), current linear speed
Action (2 continuous values in [-1, 1]):
    linear speed command, angular speed command
Reward:
    + progress towards the goal (metres gained x 10)
    - 0.01 per step (be fast)
    - small penalty when an obstacle is very close
    + 10 when the goal is reached, - 10 on collision
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ARENA = 10.0          # metres
ROBOT_RADIUS = 0.2
GOAL_RADIUS = 0.4
N_RAYS = 16
MAX_RANGE = 3.0
DT = 0.1              # seconds per step
V_MAX, W_MAX = 1.0, 2.5   # m/s, rad/s


class NavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_obstacles=(4, 8), max_steps: int = 400, sensor_noise: float = 0.0):
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.sensor_noise = sensor_noise
        self.ray_angles = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(N_RAYS + 4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------------------ map
    def _random_map(self):
        rng = self.np_random
        lo, hi = self.n_obstacles
        k = rng.integers(lo, hi + 1)
        self.obs_xy = rng.uniform(1.0, ARENA - 1.0, size=(k, 2))
        self.obs_r = rng.uniform(0.4, 1.0, size=k)
        self.pos = self._free_point()
        for _ in range(100):  # goal far enough from the start
            self.goal = self._free_point()
            if np.linalg.norm(self.goal - self.pos) > 4.0:
                break
        self.theta = rng.uniform(-np.pi, np.pi)

    def _free_point(self):
        for _ in range(1000):
            p = self.np_random.uniform(0.7, ARENA - 0.7, size=2)
            if np.all(np.linalg.norm(self.obs_xy - p, axis=1) > self.obs_r + 0.6):
                return p
        raise RuntimeError("could not place a free point")

    # ------------------------------------------------------------------ sensors
    def _lidar(self):
        """Distance along each ray to the nearest wall or obstacle (vectorised ray casting)."""
        angles = self.theta + self.ray_angles
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)          # (R, 2)
        # walls: x = 0, x = ARENA, y = 0, y = ARENA
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(d[:, 0] > 0, (ARENA - self.pos[0]) / d[:, 0], -self.pos[0] / d[:, 0])
            ty = np.where(d[:, 1] > 0, (ARENA - self.pos[1]) / d[:, 1], -self.pos[1] / d[:, 1])
        dist = np.minimum(np.abs(tx), np.abs(ty))
        # circles: solve |pos + t d - c|^2 = r^2 for the smallest t > 0
        oc = self.pos[None, :] - self.obs_xy                             # (K, 2)
        b = d @ oc.T                                                     # (R, K)
        c = (oc ** 2).sum(1) - self.obs_r ** 2                           # (K,)
        disc = b ** 2 - c[None, :]
        t = -b - np.sqrt(np.maximum(disc, 0))
        t = np.where((disc >= 0) & (t > 0), t, np.inf)
        dist = np.minimum(dist, t.min(axis=1))
        dist = np.minimum(dist, MAX_RANGE)
        if self.sensor_noise > 0:
            dist = np.clip(dist + self.np_random.normal(0, self.sensor_noise, size=dist.shape), 0, MAX_RANGE)
        return dist

    def _obs(self):
        to_goal = self.goal - self.pos
        bearing = np.arctan2(to_goal[1], to_goal[0]) - self.theta
        return np.concatenate([
            self._lidar() / MAX_RANGE,
            [np.linalg.norm(to_goal) / (ARENA * np.sqrt(2)), np.cos(bearing), np.sin(bearing), self.v / V_MAX],
        ]).astype(np.float32)

    def _clearance(self):
        """Distance from the robot's edge to the nearest obstacle or wall."""
        d_obs = np.min(np.linalg.norm(self.obs_xy - self.pos, axis=1) - self.obs_r)
        d_wall = min(self.pos[0], self.pos[1], ARENA - self.pos[0], ARENA - self.pos[1])
        return min(d_obs, d_wall) - ROBOT_RADIUS

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._random_map()
        self.v, self.steps = 0.0, 0
        self.dist = np.linalg.norm(self.goal - self.pos)
        self.path = [self.pos.copy()]
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.v = (a[0] + 1) / 2 * V_MAX                     # forward only, in [0, V_MAX]
        w = a[1] * W_MAX
        self.theta = (self.theta + w * DT + np.pi) % (2 * np.pi) - np.pi
        self.pos = self.pos + self.v * DT * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.steps += 1
        self.path.append(self.pos.copy())

        new_dist = np.linalg.norm(self.goal - self.pos)
        reward = 10.0 * (self.dist - new_dist) - 0.01
        self.dist = new_dist
        clearance = self._clearance()
        if clearance < 0.3:
            reward -= 0.05 * (0.3 - clearance) / 0.3

        terminated, event = False, None
        if clearance <= 0:
            reward, terminated, event = reward - 10.0, True, "collision"
        elif new_dist < GOAL_RADIUS:
            reward, terminated, event = reward + 10.0, True, "success"
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            event = "timeout"
        return self._obs(), float(reward), terminated, truncated, {"event": event}
