"""Episode rollout and Monte Carlo evaluation.

Episode i of `evaluate(..., seed)` uses its own environment stream
default_rng([seed, i, 0]) and policy stream default_rng([seed, i, 1]). So
every policy faces the same hidden rock configurations episode by episode
(paired comparisons), and results are identical for any number of `jobs`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .belief import RockBelief
from .env import SAMPLE, RockSampleEnv
from .policies import Policy


@dataclass
class EpisodeResult:
    discounted_return: float
    total_reward: float
    steps: int
    good_samples: int
    bad_samples: int
    returned: bool  # exited the grid (returned to base)


def run_episode(
    env: RockSampleEnv,
    policy: Policy,
    rng: np.random.Generator,
    policy_rng: np.random.Generator | None = None,
) -> EpisodeResult:
    s = env.sample_initial_state(rng)
    b = RockBelief.initial(env)
    policy.reset(env, policy_rng)
    disc, total, good, bad, t, returned = 0.0, 0.0, 0, 0, 0, False
    while not s.terminal and t < env.cfg.max_steps:
        a = policy.act(b)
        if a == SAMPLE and (i := env.rock_at(s.pos)) is not None:
            good += s.rocks[i]
            bad += not s.rocks[i]
        returned |= env.is_exit(s, a)
        sp, o, r = env.step(s, a, rng)
        disc += env.cfg.discount**t * r
        total += r
        b = b.update(env, a, o)
        s, t = sp, t + 1
    return EpisodeResult(disc, total, t, good, bad, returned)


def _run_episodes(env: RockSampleEnv, policy: Policy, seed: int, episodes: range) -> list[EpisodeResult]:
    return [
        run_episode(env, policy, np.random.default_rng([seed, i, 0]), np.random.default_rng([seed, i, 1]))
        for i in episodes
    ]


def evaluate(
    env: RockSampleEnv, policy: Policy, episodes: int = 1000, seed: int = 0, jobs: int = 1
) -> dict:
    """Monte Carlo evaluation. `jobs` > 1 splits episodes across processes
    (`jobs` = 0 uses every core); the results don't depend on `jobs`."""
    jobs = jobs or os.cpu_count() or 1
    if jobs == 1 or episodes < 2:
        results = _run_episodes(env, policy, seed, range(episodes))
    else:
        bounds = np.linspace(0, episodes, min(jobs, episodes) + 1).astype(int)
        chunks = [range(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
        with ProcessPoolExecutor(len(chunks)) as pool:
            parts = pool.map(_run_episodes, *zip(*[(env, policy, seed, c) for c in chunks]))
        results = [r for part in parts for r in part]
    ret = np.array([r.discounted_return for r in results])
    return {
        "policy": policy.name,
        "mean_return": ret.mean(),
        "stderr": ret.std(ddof=1) / np.sqrt(len(ret)) if len(ret) > 1 else 0.0,
        "return_rate": np.mean([r.returned for r in results]),
        "good_samples": np.mean([r.good_samples for r in results]),
        "bad_samples": np.mean([r.bad_samples for r in results]),
        "steps": np.mean([r.steps for r in results]),
    }
