import numpy as np
import pytest

from rover import GreedyExpectedValuePolicy, RockBelief, RockSampleEnv, RoverConfig, State, evaluate
from rover.env import EAST, NORTH, SAMPLE
from rover.planners import RolloutPlanner
from rover.sarsop import write_pomdp

ROCKS = ((1, 1), (2, 0))


def make_env(capacity, **kw):
    return RockSampleEnv(RoverConfig(grid_size=3, rocks=ROCKS, start=(0, 1), battery_capacity=capacity, **kw))


def test_battery_off_by_default():
    env = RockSampleEnv()
    s = State((0, 3), (True,) * 8)
    assert env.initial_battery == 0
    assert not env.is_stranded(s, NORTH)
    assert env.transition(s, NORTH) == (State((0, 4), (True,) * 8), 0.0)


def test_actions_consume_energy():
    env = make_env(10, move_energy=1, check_energy=2, sample_energy=3)
    s = State((1, 1), (True, False), battery=10)
    assert env.transition(s, NORTH)[0].battery == 9
    assert env.transition(s, env.check_action(0))[0].battery == 8
    sp, r = env.transition(s, SAMPLE)
    assert sp.battery == 7 and r == 10.0 and sp.rocks == (False, False)


def test_stranded_when_battery_cannot_pay():
    env = make_env(10, sample_energy=2, depletion_penalty=-50.0)
    s = State((1, 1), (True, False), battery=1)
    assert env.is_stranded(s, SAMPLE)
    sp, r = env.transition(s, SAMPLE)
    assert sp.terminal and r == -50.0
    assert sp.rocks == (True, False)  # the sample didn't happen
    assert not env.is_exit(State((2, 1), (True, False), battery=0), EAST)


def test_exit_with_exactly_enough_energy_succeeds():
    env = make_env(10)
    s = State((2, 1), (False, False), battery=1)
    assert env.is_exit(s, EAST) and not env.is_stranded(s, EAST)
    sp, r = env.transition(s, EAST)
    assert sp.terminal and r == env.cfg.exit_reward and sp.battery == 0


def test_belief_tracks_battery_and_stranding():
    env = make_env(3)
    b = RockBelief.initial(env)
    assert b.battery == 3 and b.sample_state(np.random.default_rng(0)).battery == 3
    b = b.update(env, NORTH, 0)
    assert b.battery == 2 and not b.terminal
    b = b.update(env, env.check_action(0), 1)
    b = b.update(env, env.check_action(0), 1)
    assert b.battery == 0 and not b.terminal
    b = b.update(env, NORTH, 0)
    assert b.terminal


def test_episode_reports_failure_and_energy():
    r = evaluate(make_env(2), GreedyExpectedValuePolicy(), episodes=20)
    assert r["failure_rate"] == 1.0 and r["return_rate"] == 0.0
    r = evaluate(make_env(100), GreedyExpectedValuePolicy(), episodes=20)
    assert r["failure_rate"] == 0.0 and r["return_rate"] == 1.0
    assert r["energy_used"] > 0


def test_ample_battery_matches_no_battery():
    """A battery that never runs out must not change any decision or reward."""
    base = evaluate(RockSampleEnv(), GreedyExpectedValuePolicy(), episodes=50, seed=4)
    big = evaluate(RockSampleEnv(RoverConfig(battery_capacity=10_000)), GreedyExpectedValuePolicy(), episodes=50, seed=4)
    for k in ("mean_return", "return_rate", "good_samples", "bad_samples", "steps", "energy_used"):
        assert big[k] == base[k], k


def test_rollout_avoids_stranding_better_than_greedy():
    env = RockSampleEnv(RoverConfig(battery_capacity=20))
    g = evaluate(env, GreedyExpectedValuePolicy(), episodes=20, seed=0)
    r = evaluate(env, RolloutPlanner(num_rollouts=8), episodes=20, seed=0)
    assert g["failure_rate"] > 0.5  # greedy is battery-blind
    assert r["failure_rate"] < g["failure_rate"]


def test_sarsop_export_refuses_battery(tmp_path):
    with pytest.raises(NotImplementedError):
        write_pomdp(make_env(5), tmp_path / "m.pomdp")
