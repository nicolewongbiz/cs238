import numpy as np
import pytest

from rover import RockBelief, RockSampleEnv, RoverConfig, State
from rover.env import EAST, NORTH, OBS_BAD, OBS_GOOD, OBS_NONE, SAMPLE, WEST


@pytest.fixture
def env():
    return RockSampleEnv(RoverConfig(grid_size=4, rocks=((1, 0), (3, 3)), start=(0, 0)))


def test_moves_and_walls(env):
    s = State((0, 0), (True, False))
    assert env.transition(s, NORTH)[0].pos == (0, 1)
    sp, r = env.transition(s, WEST)  # illegal: -100 and terminate
    assert sp.terminal and r == -100


def test_walls_without_termination():
    env = RockSampleEnv(RoverConfig(grid_size=4, rocks=((1, 0),), start=(0, 0), illegal_terminates=False))
    s = State((0, 0), (True,))
    assert env.transition(s, WEST) == (s, -100.0)


def test_default_is_published_rocksample_7_8():
    cfg = RoverConfig()
    assert cfg.grid_size == 7 and cfg.num_rocks == 8 and cfg.start == (0, 3)


def test_exit_east_is_terminal_with_reward(env):
    sp, r = env.transition(State((3, 1), (True, False)), EAST)
    assert sp.terminal and r == env.cfg.exit_reward


def test_sampling(env):
    sp, r = env.transition(State((1, 0), (True, False)), SAMPLE)
    assert r == 10 and sp.rocks == (False, False)
    _, r = env.transition(sp, SAMPLE)
    assert r == -10
    sp, r = env.transition(State((0, 0), (True, False)), SAMPLE)  # empty cell is illegal
    assert r == -100 and sp.terminal


def test_sensor_accuracy(env):
    assert env.sensor_accuracy((1, 0), 0) == pytest.approx(1.0)
    far = env.sensor_accuracy((0, 3), 0)
    assert 0.5 < far < 1.0


def test_observation_probs(env):
    s = State((0, 0), (True, False))
    assert env.observation_probs(NORTH, s)[OBS_NONE] == 1.0
    p = env.observation_probs(env.check_action(0), s)
    assert p[OBS_GOOD] > p[OBS_BAD] and p.sum() == pytest.approx(1.0)


def test_belief_update_is_bayes(env):
    b = RockBelief.initial(env)
    a = env.check_action(1)
    acc = env.sensor_accuracy(b.pos, 1)
    b2 = b.update(env, a, OBS_GOOD)
    assert b2.p_good[1] == pytest.approx(acc)  # uniform prior -> posterior = accuracy
    assert b2.p_good[0] == 0.5
    assert b.p_good[1] == 0.5  # update doesn't mutate


def test_belief_tracks_position_and_sampling(env):
    b = RockBelief.initial(env).update(env, EAST, OBS_NONE)
    assert b.pos == (1, 0)
    b = b.update(env, SAMPLE, OBS_NONE)
    assert b.p_good[0] == 0.0
