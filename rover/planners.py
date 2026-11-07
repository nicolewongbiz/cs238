"""Risk-aware, belief-informed planners (proposal policy 3).

RolloutPlanner: one-step lookahead with Monte Carlo rollouts (Kochenderfer et
al. 2022, sec. 22.3 / Bertsekas' rollout algorithm). For every candidate action
it samples states from the belief, applies the action, updates a copy of the
belief exactly, and then follows a base policy (by default the greedy
baseline) to the end of the episode. The action with the best *score* of the
sampled discounted returns is chosen.

Risk awareness: with `risk_alpha < 1`, actions are scored by CVaR_alpha (the
mean of the worst alpha-fraction of rollout returns) instead of the mean, so the
planner prefers actions whose bad outcomes are less bad. On plain RockSample
the only catastrophic outcome is an illegal action, so this matters once
battery depletion and terrain hazards are added.

Rollout improvement guarantees the planner does no worse than its base policy
in expectation (up to Monte Carlo error), so it should land between greedy and
the SARSOP reference.

Next: POMCP (Silver & Veness, 2010) for deeper lookahead.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .belief import RockBelief
from .env import EAST, NORTH, NUM_BASIC_ACTIONS, SAMPLE, SOUTH, WEST, RockSampleEnv, State
from .policies import GreedyExpectedValuePolicy, Policy


def cvar(returns: np.ndarray, alpha: float) -> float:
    """Mean of the worst ceil(alpha * n) returns; alpha = 1 gives the plain mean."""
    if alpha >= 1.0:
        return float(np.mean(returns))
    m = max(1, math.ceil(alpha * len(returns)))
    return float(np.mean(np.partition(returns, m - 1)[:m]))


def candidate_actions(env: RockSampleEnv, belief: RockBelief) -> list[int]:
    """Actions worth evaluating: drop moves into a wall, sampling off a rock or an
    already-sampled rock, and checking rocks whose value is already known."""
    x, y = belief.pos
    n = env.cfg.grid_size
    acts = [EAST]  # always legal (exits at the east edge)
    if y < n - 1:
        acts.append(NORTH)
    if y > 0:
        acts.append(SOUTH)
    if x > 0:
        acts.append(WEST)
    i = env.rock_at(belief.pos)
    if i is not None and belief.p_good[i] > 0:
        acts.append(SAMPLE)
    acts += [env.check_action(j) for j, p in enumerate(belief.p_good) if 0.0 < p < 1.0]
    return acts


class RolloutPlanner(Policy):
    name = "rollout"

    def __init__(
        self,
        num_rollouts: int = 20,
        horizon: int = 60,
        risk_alpha: float = 1.0,
        base_policy: Callable[[], Policy] = GreedyExpectedValuePolicy,
        seed: int = 0,
    ):
        self.num_rollouts = num_rollouts
        self.horizon = horizon
        self.risk_alpha = risk_alpha
        self.make_base = base_policy
        self.rng = np.random.default_rng(seed)
        if risk_alpha < 1.0:
            self.name = f"rollout-cvar{risk_alpha:g}"

    def reset(self, env):
        self.env = env
        self.base = self.make_base()

    def act(self, belief):
        if belief.terminal:
            return EAST
        actions = candidate_actions(self.env, belief)
        # Common random numbers: every action is scored on the same sampled
        # states and random streams, so differences between actions aren't noise.
        states = [belief.sample_state(self.rng) for _ in range(self.num_rollouts)]
        seeds = self.rng.integers(2**63, size=self.num_rollouts)
        best_a, best_score = actions[0], -math.inf
        for a in actions:
            returns = np.array(
                [self._simulate(s, belief, a, np.random.default_rng(sd)) for s, sd in zip(states, seeds)]
            )
            score = cvar(returns, self.risk_alpha)
            if score > best_score:
                best_a, best_score = a, score
        return best_a

    def _simulate(self, s: State, b: RockBelief, a: int, rng: np.random.Generator) -> float:
        """Discounted return of taking `a` from (s, b), then following the base policy."""
        env, gamma = self.env, self.env.cfg.discount
        self.base.reset(env)
        ret, disc = 0.0, 1.0
        for _ in range(self.horizon):
            s, o, r = env.step(s, a, rng)
            ret += disc * r
            if s.terminal:
                break
            disc *= gamma
            b = b.update(env, a, o)
            a = self.base.act(b)
        return ret
