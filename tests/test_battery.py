import numpy as np
import pytest

from rover import (
    BatteryAwareGreedyPolicy,
    GreedyExpectedValuePolicy,
    RockBelief,
    RockSampleEnv,
    RoverConfig,
    State,
    evaluate,
)
from rover.env import EAST, NORTH, SAMPLE
from rover.planners import RolloutPlanner
from rover.sarsop import SarsopPolicy, StateIndex, find_pomdpsol, initial_distribution, solve, write_pomdp


def _have_pomdpsol():
    try:
        find_pomdpsol()
        return True
    except FileNotFoundError:
        return False

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


def test_sarsop_state_index_with_battery():
    env = make_env(4)
    idx = StateIndex(env)
    assert idx.size == 5 * 9 * 4 + 1
    assert sorted(idx.index(s) for s in idx.states()) == list(range(idx.size))
    b0 = initial_distribution(env, idx)
    assert b0.sum() == pytest.approx(1.0)
    assert all(s.battery == 4 for s in idx.states() if b0[idx.index(s)] > 0)


def test_sarsop_export_with_battery(tmp_path):
    env = make_env(4)
    idx = write_pomdp(env, tmp_path / "m.pomdp")
    assert (tmp_path / "m.pomdp").read_text().count("T: ") == idx.size * env.num_actions


def test_sarsop_export_refuses_huge_models(tmp_path):
    with pytest.raises(ValueError):
        write_pomdp(RockSampleEnv(RoverConfig(battery_capacity=25)), tmp_path / "m.pomdp")


@pytest.mark.skipif(not _have_pomdpsol(), reason="pomdpsol not built (scripts/install_sarsop.sh)")
def test_sarsop_with_battery_end_to_end(tmp_path):
    """SARSOP's battery-constrained policy should never strand and its simulated return should match its value."""
    env = make_env(6)
    write_pomdp(env, tmp_path / "m.pomdp")
    solve(tmp_path / "m.pomdp", tmp_path / "m.policy", timeout=30)
    policy = SarsopPolicy.load(tmp_path / "m.policy", env)
    v0 = policy.value(RockBelief.initial(env))
    r = evaluate(env, policy, episodes=1000, seed=0)
    assert r["failure_rate"] == 0.0
    assert r["mean_return"] == pytest.approx(v0, abs=4 * r["stderr"] + 0.1)


def test_battery_aware_greedy_matches_greedy_without_battery():
    env = RockSampleEnv()
    g = evaluate(env, GreedyExpectedValuePolicy(), episodes=50, seed=2)
    ga = evaluate(env, BatteryAwareGreedyPolicy(), episodes=50, seed=2)
    assert g == {**ga, "policy": g["policy"]}


@pytest.mark.parametrize("capacity", [7, 15, 25, 35])
def test_battery_aware_greedy_never_strands(capacity):
    env = RockSampleEnv(RoverConfig(battery_capacity=capacity))  # 7 = exact cost of driving home from start
    r = evaluate(env, BatteryAwareGreedyPolicy(), episodes=50, seed=0)
    assert r["failure_rate"] == 0.0 and r["return_rate"] == 1.0
