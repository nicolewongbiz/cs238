import numpy as np
import pytest

from rover import RockBelief, RockSampleEnv, RoverConfig
from rover.sarsop import StateIndex, config_probs, initial_distribution, write_pomdp


@pytest.fixture
def env():
    return RockSampleEnv(RoverConfig(grid_size=3, rocks=((1, 1), (2, 0)), start=(0, 1)))


def test_state_index_is_bijection(env):
    idx = StateIndex(env)
    seen = [idx.index(s) for s in idx.states()]
    assert sorted(seen) == list(range(idx.size))


def test_initial_distribution(env):
    b0 = initial_distribution(env, StateIndex(env))
    assert b0.sum() == pytest.approx(1.0)
    assert np.count_nonzero(b0) == 4


def test_config_probs():
    p = config_probs(np.array([0.2, 0.9]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0b11] == pytest.approx(0.18)


def test_write_pomdp(env, tmp_path):
    idx = write_pomdp(env, tmp_path / "m.pomdp")
    text = (tmp_path / "m.pomdp").read_text()
    assert f"states: {idx.size}" in text
    assert text.count("T: ") == idx.size * env.num_actions
