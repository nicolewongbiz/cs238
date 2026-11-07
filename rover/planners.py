"""Risk-aware, belief-informed planners (proposal policy 3).

RolloutPlanner: one-step lookahead with Monte Carlo rollouts (Kochenderfer et
al. 2022, sec. 22.3; Bertsekas' rollout algorithm). For every candidate action
it samples states from the belief, applies the action, updates a copy of the
belief exactly, and then follows a base policy (by default the greedy
baseline) to the end of the episode. The action with the best *score* of the
sampled discounted returns is chosen. With `risk_alpha < 1`, actions are scored
by CVaR_alpha (the mean of the worst alpha-fraction of rollout returns) instead
of the mean, so the planner prefers actions whose bad outcomes are less bad.

POMCPPlanner: Partially Observable Monte Carlo Planning (Silver & Veness,
2010; Kochenderfer et al. 2022, sec. 22.6). Monte Carlo tree search over
action/observation histories with UCB1 action selection. Each simulation
samples a root state from the exact belief, descends the tree, and estimates
new leaves with one rollout of the base policy. Because the root belief is
exact here, a fresh tree is built every step and no particle filter is needed.

On plain RockSample the only catastrophic outcome is an illegal action, so risk
scoring matters once battery depletion and terrain hazards are added.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .belief import RockBelief
from .env import EAST, NORTH, SAMPLE, SOUTH, WEST, RockSampleEnv, State
from .policies import GreedyExpectedValuePolicy, Policy


def cvar(returns: np.ndarray, alpha: float) -> float:
    """Mean of the worst ceil(alpha * n) returns; alpha = 1 gives the plain mean."""
    if alpha >= 1.0:
        return float(np.mean(returns))
    m = max(1, math.ceil(alpha * len(returns)))
    return float(np.mean(np.partition(returns, m - 1)[:m]))


def candidate_actions(env: RockSampleEnv, belief: RockBelief, certain: float = 0.01) -> list[int]:
    """Actions worth evaluating: drop moves into a wall, sampling off a rock or an
    already-sampled rock, and checking rocks whose value is already (nearly)
    known, i.e. P(good) outside (certain, 1 - certain).

    Pruning near-certain checks matters for tree search: re-checking a rock that
    is surely bad wastes one discount step, a loss smaller than the Monte Carlo
    noise in Q, so with few simulations POMCP can otherwise check forever."""
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
    acts += [env.check_action(j) for j, p in enumerate(belief.p_good) if certain < p < 1.0 - certain]
    return acts


def base_rollout(
    env: RockSampleEnv,
    base: Policy,
    s: State,
    b: RockBelief,
    rng: np.random.Generator,
    horizon: int,
    first_action: int | None = None,
) -> float:
    """Discounted return of following `base` from (s, b) for up to `horizon` steps,
    optionally taking `first_action` first. The belief is updated exactly along
    the way, so the base policy sees what the rover would actually know."""
    gamma = env.cfg.discount
    base.reset(env)
    a = first_action
    ret, disc = 0.0, 1.0
    for _ in range(horizon):
        if a is None:
            a = base.act(b)
        s, o, r = env.step(s, a, rng)
        ret += disc * r
        if s.terminal:
            break
        disc *= gamma
        b = b.update(env, a, o)
        a = None
    return ret


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

    def reset(self, env, rng=None):
        self.env = env
        self.base = self.make_base()
        if rng is not None:
            self.rng = rng

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
                [
                    base_rollout(self.env, self.base, s, belief, np.random.default_rng(sd), self.horizon, a)
                    for s, sd in zip(states, seeds)
                ]
            )
            score = cvar(returns, self.risk_alpha)
            if score > best_score:
                best_a, best_score = a, score
        return best_a


class _Node:
    """History node: visit counts and running-mean Q for each candidate action."""

    __slots__ = ("actions", "N", "n", "q", "children")

    def __init__(self, actions: list[int]):
        self.actions = actions
        self.N = 0
        self.n = [0] * len(actions)
        self.q = [0.0] * len(actions)
        self.children: dict[tuple[int, int], _Node] = {}  # (action index, obs) -> node


class POMCPPlanner(Policy):
    name = "pomcp"

    def __init__(
        self,
        num_sims: int = 1000,
        exploration: float = 10.0,
        horizon: int = 60,
        base_policy: Callable[[], Policy] = GreedyExpectedValuePolicy,
        seed: int = 0,
    ):
        self.num_sims = num_sims
        self.c = exploration
        self.horizon = horizon
        self.make_base = base_policy
        self.rng = np.random.default_rng(seed)

    def reset(self, env, rng=None):
        self.env = env
        self.base = self.make_base()
        if rng is not None:
            self.rng = rng

    def act(self, belief):
        if belief.terminal:
            return EAST
        root = _Node(candidate_actions(self.env, belief))
        for _ in range(self.num_sims):
            self._simulate(belief.sample_state(self.rng), belief, root, 0)
        best = max((i for i in range(len(root.actions)) if root.n[i] > 0), key=root.q.__getitem__)
        self.last_root = root  # kept for inspection / debugging
        return root.actions[best]

    def _select(self, node: _Node) -> int:
        """UCB1, trying every action once first."""
        for i, k in enumerate(node.n):
            if k == 0:
                return i
        log_n = math.log(node.N)
        return max(
            range(len(node.actions)),
            key=lambda i: node.q[i] + self.c * math.sqrt(log_n / node.n[i]),
        )

    def _simulate(self, s: State, b: RockBelief, node: _Node, depth: int) -> float:
        env = self.env
        if depth >= self.horizon:
            return 0.0
        i = self._select(node)
        a = node.actions[i]
        sp, o, r = env.step(s, a, self.rng)
        if sp.terminal:
            q = r
        else:
            bp = b.update(env, a, o)
            child = node.children.get((i, o))
            if child is None:
                node.children[(i, o)] = _Node(candidate_actions(env, bp))
                future = base_rollout(env, self.base, sp, bp, self.rng, self.horizon - depth - 1)
            else:
                future = self._simulate(sp, bp, child, depth + 1)
            q = r + env.cfg.discount * future
        node.N += 1
        node.n[i] += 1
        node.q[i] += (q - node.q[i]) / node.n[i]
        return q
