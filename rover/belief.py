"""Exact belief for RockSample.

The rover's position is fully observed and rocks are independent a priori, and
each CHECK_i only informs rock i, so the belief factorizes into one Bernoulli
P(good) per rock. Updates are exact Bayes.

When hazards/battery are added, anything hidden and non-factorizable should
move to a particle belief (e.g. for a POMCP-style planner).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import NUM_BASIC_ACTIONS, OBS_GOOD, OBS_NONE, SAMPLE, RockSampleEnv, State


@dataclass
class RockBelief:
    pos: tuple[int, int]
    p_good: np.ndarray  # shape (num_rocks,)
    terminal: bool = False

    @classmethod
    def initial(cls, env: RockSampleEnv) -> "RockBelief":
        return cls(env.cfg.start, np.full(env.cfg.num_rocks, env.cfg.p_good))

    def copy(self) -> "RockBelief":
        return RockBelief(self.pos, self.p_good.copy(), self.terminal)

    def sample_state(self, rng: np.random.Generator) -> State:
        rocks = tuple(bool(g) for g in rng.random(len(self.p_good)) < self.p_good)
        return State(self.pos, rocks, self.terminal)

    def update(self, env: RockSampleEnv, a: int, o: int) -> "RockBelief":
        """Return the posterior belief after taking action `a` and observing `o`."""
        b = self.copy()
        if b.terminal:
            return b
        # Position dynamics don't depend on rock values, so propagate any state.
        sp, _ = env.transition(State(b.pos, (False,) * len(b.p_good)), a)
        b.pos, b.terminal = sp.pos, sp.terminal
        if a == SAMPLE:
            i = env.rock_at(b.pos)
            if i is not None:
                b.p_good[i] = 0.0  # sampled rocks become bad
        elif a >= NUM_BASIC_ACTIONS and o != OBS_NONE:
            i = a - NUM_BASIC_ACTIONS
            acc = env.sensor_accuracy(b.pos, i)
            like_good = acc if o == OBS_GOOD else 1.0 - acc
            like_bad = 1.0 - like_good
            p = b.p_good[i]
            denom = like_good * p + like_bad * (1.0 - p)
            if denom > 0:
                b.p_good[i] = like_good * p / denom
        return b
