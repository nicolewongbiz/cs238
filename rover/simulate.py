"""Episode rollout and Monte Carlo evaluation."""

from __future__ import annotations

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


def run_episode(env: RockSampleEnv, policy: Policy, rng: np.random.Generator) -> EpisodeResult:
    s = env.sample_initial_state(rng)
    b = RockBelief.initial(env)
    policy.reset(env)
    disc, total, good, bad, t = 0.0, 0.0, 0, 0, 0
    while not s.terminal and t < env.cfg.max_steps:
        a = policy.act(b)
        if a == SAMPLE and (i := env.rock_at(s.pos)) is not None:
            good += s.rocks[i]
            bad += not s.rocks[i]
        sp, o, r = env.step(s, a, rng)
        disc += env.cfg.discount**t * r
        total += r
        b = b.update(env, a, o)
        s, t = sp, t + 1
    return EpisodeResult(disc, total, t, good, bad, s.terminal)


def evaluate(env: RockSampleEnv, policy: Policy, episodes: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    results = [run_episode(env, policy, rng) for _ in range(episodes)]
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
