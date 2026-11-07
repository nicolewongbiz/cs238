"""RockSample-style Mars rover POMDP (Smith & Simmons, 2004).

The core model is classic RockSample(n, k):
  * an n x n grid, k rocks at known positions, each rock hidden "good" or "bad";
  * actions: move N/S/E/W, SAMPLE the rock under the rover, CHECK_i any rock i;
  * CHECK_i returns a noisy good/bad reading whose accuracy decays with distance:
        eta = 2^(-d / d0),  P(correct reading) = (1 + eta) / 2;
  * sampling a good rock gives +10 (and the rock becomes bad), a bad rock -10;
  * moving east off the grid ends the episode with +10 ("return to base");
  * illegal actions (bumping a N/S/W wall, sampling an empty cell) give -100 and
    end the episode, as in the RockSample_7_8 model distributed with SARSOP.

The default config is the published RockSample(7,8) benchmark: 7x7 grid,
8 rocks, start (0, 3), d0 = 20, discount 0.95.

Project extensions (battery budget, terrain hazards, sensing costs) should be
added on top of this model via `RoverConfig` fields so the plain RockSample
configuration stays available as a benchmark that SARSOP can solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

NORTH, SOUTH, EAST, WEST, SAMPLE = range(5)
NUM_BASIC_ACTIONS = 5
MOVES = {NORTH: (0, 1), SOUTH: (0, -1), EAST: (1, 0), WEST: (-1, 0)}

OBS_NONE, OBS_GOOD, OBS_BAD = range(3)
NUM_OBS = 3

# Published RockSample(7,8) layout (Smith & Simmons, 2004), (x, y) with north = +y.
# Matches third_party/sarsop/examples/POMDP/RockSample_7_8.pomdp.
ROCKSAMPLE_7_8_ROCKS = ((2, 0), (0, 1), (3, 1), (6, 3), (2, 4), (3, 4), (5, 5), (1, 6))
ROCKSAMPLE_7_8_START = (0, 3)


@dataclass(frozen=True)
class RoverConfig:
    grid_size: int = 7
    rocks: tuple[tuple[int, int], ...] = ROCKSAMPLE_7_8_ROCKS
    start: tuple[int, int] = ROCKSAMPLE_7_8_START
    sensor_half_eff_dist: float = 20.0  # d0 in eta = 2^(-d/d0)
    p_good: float = 0.5  # prior probability each rock is good
    good_reward: float = 10.0
    bad_penalty: float = -10.0  # sampling a bad rock
    exit_reward: float = 10.0
    illegal_penalty: float = -100.0  # bumping a wall or sampling an empty cell
    illegal_terminates: bool = True
    step_cost: float = 0.0  # applied to every action
    check_cost: float = 0.0  # extra cost per CHECK action
    discount: float = 0.95
    max_steps: int = 200

    @property
    def num_rocks(self) -> int:
        return len(self.rocks)


def random_layout(grid_size: int, num_rocks: int, seed: int = 0) -> tuple[tuple[int, int], ...]:
    rng = np.random.default_rng(seed)
    cells = rng.choice(grid_size * grid_size, size=num_rocks, replace=False)
    return tuple((int(c % grid_size), int(c // grid_size)) for c in cells)


class State(NamedTuple):
    pos: tuple[int, int]
    rocks: tuple[bool, ...]  # True = good
    terminal: bool = False


class RockSampleEnv:
    def __init__(self, config: RoverConfig | None = None):
        self.cfg = config or RoverConfig()
        n = self.cfg.grid_size
        for x, y in (*self.cfg.rocks, self.cfg.start):
            if not (0 <= x < n and 0 <= y < n):
                raise ValueError(f"position {(x, y)} outside {n}x{n} grid")
        self._rock_at = {pos: i for i, pos in enumerate(self.cfg.rocks)}

    # ----- spaces -----------------------------------------------------------
    @property
    def num_actions(self) -> int:
        return NUM_BASIC_ACTIONS + self.cfg.num_rocks

    @staticmethod
    def check_action(rock: int) -> int:
        return NUM_BASIC_ACTIONS + rock

    def action_name(self, a: int) -> str:
        if a < NUM_BASIC_ACTIONS:
            return ["north", "south", "east", "west", "sample"][a]
        return f"check{a - NUM_BASIC_ACTIONS}"

    def rock_at(self, pos: tuple[int, int]) -> int | None:
        return self._rock_at.get(pos)

    # ----- model ------------------------------------------------------------
    def sample_initial_state(self, rng: np.random.Generator) -> State:
        rocks = tuple(bool(g) for g in rng.random(self.cfg.num_rocks) < self.cfg.p_good)
        return State(self.cfg.start, rocks)

    def sensor_accuracy(self, pos: tuple[int, int], rock: int) -> float:
        """P(CHECK_rock reports the true rock value) from position `pos`."""
        d = float(np.hypot(pos[0] - self.cfg.rocks[rock][0], pos[1] - self.cfg.rocks[rock][1]))
        eta = 2.0 ** (-d / self.cfg.sensor_half_eff_dist)
        return 0.5 * (1.0 + eta)

    def transition(self, s: State, a: int) -> tuple[State, float]:
        """Deterministic next state and reward for plain RockSample."""
        if s.terminal:
            return s, 0.0
        r = -self.cfg.step_cost
        if a in MOVES:
            dx, dy = MOVES[a]
            x, y = s.pos[0] + dx, s.pos[1] + dy
            n = self.cfg.grid_size
            if x >= n:  # exit to the east = return to base
                return State(s.pos, s.rocks, terminal=True), r + self.cfg.exit_reward
            if 0 <= x < n and 0 <= y < n:
                return State((x, y), s.rocks), r
            return self._illegal(s, r)  # bumped into a wall
        if a == SAMPLE:
            i = self.rock_at(s.pos)
            if i is None:
                return self._illegal(s, r)
            reward = self.cfg.good_reward if s.rocks[i] else self.cfg.bad_penalty
            rocks = s.rocks[:i] + (False,) + s.rocks[i + 1 :]
            return State(s.pos, rocks), r + reward
        return s, r - self.cfg.check_cost  # CHECK_i

    def _illegal(self, s: State, r: float) -> tuple[State, float]:
        return State(s.pos, s.rocks, terminal=self.cfg.illegal_terminates), r + self.cfg.illegal_penalty

    def is_exit(self, s: State, a: int) -> bool:
        """True if taking `a` in `s` leaves the grid to the east (returns to base)."""
        return not s.terminal and a == EAST and s.pos[0] == self.cfg.grid_size - 1

    def observation_probs(self, a: int, sp: State) -> np.ndarray:
        """Distribution over observations after taking `a` and landing in `sp`."""
        probs = np.zeros(NUM_OBS)
        if a < NUM_BASIC_ACTIONS or sp.terminal:
            probs[OBS_NONE] = 1.0
            return probs
        i = a - NUM_BASIC_ACTIONS
        acc = self.sensor_accuracy(sp.pos, i)
        probs[OBS_GOOD if sp.rocks[i] else OBS_BAD] = acc
        probs[OBS_BAD if sp.rocks[i] else OBS_GOOD] = 1.0 - acc
        return probs

    def step(self, s: State, a: int, rng: np.random.Generator) -> tuple[State, int, float]:
        """Generative model: returns (next_state, observation, reward)."""
        sp, r = self.transition(s, a)
        o = int(rng.choice(NUM_OBS, p=self.observation_probs(a, sp)))
        return sp, o, r
