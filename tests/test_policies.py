import numpy as np

from rover import GreedyExpectedValuePolicy, NearestTargetPolicy, RandomPolicy, RockSampleEnv, evaluate


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
