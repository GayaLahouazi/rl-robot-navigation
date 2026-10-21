"""Hand-designed reactive controller (no learning), used as a reference.

Inspired by Braitenberg vehicles / potential fields: turn towards the goal,
and get pushed away by close obstacles detected by the lidar.
"""
import numpy as np

from .env import N_RAYS


class PotentialFieldController:
    def __init__(self, k_goal: float = 1.5, k_obs: float = 0.6, safe: float = 0.35):
        self.angles = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)
        self.k_goal, self.k_obs, self.safe = k_goal, k_obs, safe

    def act(self, obs: np.ndarray) -> np.ndarray:
        lidar = obs[:N_RAYS]                        # distances / MAX_RANGE, in [0, 1]
        cos_b, sin_b = obs[N_RAYS + 1], obs[N_RAYS + 2]
        bearing = np.arctan2(sin_b, cos_b)
        # repulsion: each close obstacle pushes the heading away from its direction
        front = np.abs(self.angles) < np.pi / 2
        push = np.where(front, (1.0 / np.maximum(lidar, 0.05) - 1.0), 0.0)
        turn = self.k_goal * bearing - self.k_obs * np.sum(push * np.sin(self.angles)) / N_RAYS
        front_clear = lidar[np.abs(self.angles) < np.pi / 6].min()
        speed = 1.0 if front_clear > self.safe else -0.6   # slow down in front of an obstacle
        return np.clip(np.array([speed, turn]), -1, 1)
