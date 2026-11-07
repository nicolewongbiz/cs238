import numpy as np
import pytest

from rover import RockBelief, RockSampleEnv, RoverConfig, evaluate
from rover.sarsop import (
    SarsopPolicy,
    StateIndex,
    config_probs,
    find_pomdpsol,
    initial_distribution,
    solve,
    write_pomdp,
)


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


def _have_pomdpsol():
    try:
        find_pomdpsol()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _have_pomdpsol(), reason="pomdpsol not built (scripts/install_sarsop.sh)")
def test_solve_and_simulate(env, tmp_path):
    """End to end: export, solve, load alpha vectors, and check simulation agrees with the bound."""
    write_pomdp(env, tmp_path / "m.pomdp")
    solve(tmp_path / "m.pomdp", tmp_path / "m.policy", timeout=20, precision=1e-3)
    policy = SarsopPolicy.load(tmp_path / "m.policy", env)
    v0 = policy.value(RockBelief.initial(env))
    r = evaluate(env, policy, episodes=2000, seed=0)
    assert r["return_rate"] == 1.0
    assert r["mean_return"] == pytest.approx(v0, abs=4 * r["stderr"] + 0.1)
