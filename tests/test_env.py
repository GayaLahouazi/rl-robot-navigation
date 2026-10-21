import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from robotnav.baselines import PotentialFieldController
from robotnav.env import MAX_RANGE, N_RAYS, NavEnv
from robotnav.evaluation import evaluate
from robotnav.neuroevolution import MLPPolicy


def test_gymnasium_api_compliance():
    check_env(NavEnv(), skip_render_check=True)


def test_reset_is_deterministic_with_seed():
    a, _ = NavEnv().reset(seed=3)
    b, _ = NavEnv().reset(seed=3)
    assert np.allclose(a, b)


def test_lidar_sees_a_wall_straight_ahead():
    env = NavEnv()
    env.reset(seed=0)
    env.obs_xy, env.obs_r = np.array([[50.0, 50.0]]), np.array([0.1])   # move obstacles away
    env.pos, env.theta = np.array([9.0, 5.0]), 0.0                       # facing the x = 10 wall
    lidar = env._lidar()
    front = N_RAYS // 2                                                   # ray angle 0
    assert lidar[front] == pytest.approx(1.0)
    assert lidar[0] == pytest.approx(MAX_RANGE)                           # behind: wall at 9 m > range


def test_collision_terminates_with_penalty():
    env = NavEnv()
    env.reset(seed=0)
    env.obs_xy, env.obs_r = np.array([[5.6, 5.0]]), np.array([0.3])
    env.pos, env.theta = np.array([5.0, 5.0]), 0.0
    env.goal = np.array([1.0, 1.0])
    env.dist = np.linalg.norm(env.goal - env.pos)
    _, r, term, _, info = env.step(np.array([1.0, 0.0]))
    assert term and info["event"] == "collision" and r < -5


def test_policies_run_and_return_metrics():
    pol = MLPPolicy(N_RAYS + 4, 2, hidden=8)
    pol.set(np.random.randn(pol.n_params) * 0.1)
    for fn in (pol.act, PotentialFieldController().act):
        res = evaluate(fn, n_episodes=3)
        assert res["success"] + res["collision"] + res["timeout"] == pytest.approx(1.0)
