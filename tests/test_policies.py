import numpy as np
import pytest

from rover import (
    GreedyExpectedValuePolicy,
    NearestTargetPolicy,
    RandomPolicy,
    RockBelief,
    RockSampleEnv,
    RoverConfig,
    evaluate,
)
from rover.env import EAST, NORTH, SAMPLE, SOUTH, WEST
from rover.planners import RolloutPlanner, candidate_actions, cvar


def test_baselines_run_and_return():
    env = RockSampleEnv()
    for policy in (RandomPolicy(), NearestTargetPolicy(), GreedyExpectedValuePolicy()):
        r = evaluate(env, policy, episodes=20, seed=1)
        assert np.isfinite(r["mean_return"])
    assert evaluate(env, NearestTargetPolicy(), episodes=20)["return_rate"] == 1.0
    assert evaluate(env, GreedyExpectedValuePolicy(), episodes=20)["return_rate"] == 1.0


def test_greedy_beats_nearest():
    env = RockSampleEnv()
    g = evaluate(env, GreedyExpectedValuePolicy(), episodes=300)["mean_return"]
    n = evaluate(env, NearestTargetPolicy(), episodes=300)["mean_return"]
    assert g > n


# ----- planners --------------------------------------------------------------


def test_cvar():
    r = np.array([-10.0, 0.0, 5.0, 20.0])
    assert cvar(r, 1.0) == pytest.approx(3.75)
    assert cvar(r, 0.5) == pytest.approx(-5.0)
    assert cvar(r, 0.01) == pytest.approx(-10.0)


def test_candidate_actions_prune_illegal():
    env = RockSampleEnv()
    b = RockBelief((0, 0), np.array([0.5, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]))
    acts = candidate_actions(env, b)
    assert WEST not in acts and SOUTH not in acts  # walls
    assert EAST in acts and NORTH in acts
    assert SAMPLE not in acts  # no rock at (0, 0)
    assert env.check_action(1) not in acts and env.check_action(2) not in acts  # known rocks
    on_rock = RockBelief(env.cfg.rocks[0], np.full(8, 0.5))
    assert SAMPLE in candidate_actions(env, on_rock)


def test_rollout_planner_runs_and_returns():
    env = RockSampleEnv(RoverConfig(grid_size=3, rocks=((1, 1), (2, 0)), start=(0, 1)))
    r = evaluate(env, RolloutPlanner(num_rollouts=8), episodes=20, seed=0)
    assert r["return_rate"] == 1.0
    assert r["bad_samples"] < 0.5
