"""Baseline policies. Every policy maps a belief to an action.

Proposal mapping:
  (1) fixed / nearest-target      -> NearestTargetPolicy
  (2) greedy expected-science/cost -> GreedyExpectedValuePolicy
  (3) risk-aware belief planner    -> rover/planners.py (RolloutPlanner)
"""

from __future__ import annotations

import numpy as np

from .belief import RockBelief
from .env import EAST, NORTH, SAMPLE, SOUTH, WEST, RockSampleEnv


class Policy:
    name = "policy"

    def reset(self, env: RockSampleEnv) -> None:
        """Called at the start of every episode."""

    def act(self, belief: RockBelief) -> int:
        raise NotImplementedError


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_toward(pos: tuple[int, int], target: tuple[int, int]) -> int:
    if pos[0] != target[0]:
        return EAST if target[0] > pos[0] else WEST
    return NORTH if target[1] > pos[1] else SOUTH


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        self.n = env.num_actions

    def act(self, belief):
        return int(self.rng.integers(self.n))


class NearestTargetPolicy(Policy):
    """Belief-agnostic: visit and sample every rock in nearest-first order, then exit."""

    name = "nearest"

    def reset(self, env):
        self.env = env
        self.visited: set[int] = set()

    def act(self, belief):
        i = self.env.rock_at(belief.pos)
        if i is not None and i not in self.visited:
            self.visited.add(i)
            return SAMPLE
        remaining = [j for j in range(self.env.cfg.num_rocks) if j not in self.visited]
        if not remaining:
            return EAST
        target = min(remaining, key=lambda j: manhattan(belief.pos, self.env.cfg.rocks[j]))
        return move_toward(belief.pos, self.env.cfg.rocks[target])


class GreedyExpectedValuePolicy(Policy):
    """Check uncertain rocks, then chase the rock with the best discounted expected value.

    1. If standing on a rock with positive expected value, sample it.
    2. If some rock's belief is uncertain (and we haven't checked it too often), check it.
    3. Otherwise move toward argmax_i gamma^dist_i * E[reward of sampling i], if positive.
    4. Otherwise head east to exit.
    """

    name = "greedy"

    def __init__(self, uncertain: tuple[float, float] = (0.1, 0.9), max_checks: int = 3):
        self.lo, self.hi = uncertain
        self.max_checks = max_checks

    def reset(self, env):
        self.env = env
        self.checks = [0] * env.cfg.num_rocks

    def _expected_sample_value(self, p: float) -> float:
        cfg = self.env.cfg
        return p * cfg.good_reward + (1 - p) * cfg.bad_penalty

    def act(self, belief):
        # Plain Python lists: this runs inside every planner rollout, where numpy
        # scalar overhead dominates for k = 8 rocks.
        cfg, pos = self.env.cfg, belief.pos
        p_good = belief.p_good.tolist()
        i = self.env.rock_at(pos)
        if i is not None and self._expected_sample_value(p_good[i]) > 0:
            return SAMPLE

        uncertain = [
            j for j, p in enumerate(p_good) if self.lo < p < self.hi and self.checks[j] < self.max_checks
        ]
        if uncertain:
            j = min(uncertain, key=lambda j: manhattan(pos, cfg.rocks[j]))
            self.checks[j] += 1
            return self.env.check_action(j)

        scores = [
            cfg.discount ** manhattan(pos, cfg.rocks[j]) * self._expected_sample_value(p)
            for j, p in enumerate(p_good)
        ]
        best = max(range(len(scores)), key=scores.__getitem__)  # first max, like np.argmax
        if scores[best] > 0:
            return move_toward(pos, cfg.rocks[best])
        return EAST
