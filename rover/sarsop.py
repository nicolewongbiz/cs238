"""SARSOP near-optimal reference policy (Kurniawati, Hsu & Lee, 2008).

Pipeline:
  1. `write_pomdp(env, path)` exports the RockSample model in Cassandra .pomdp format.
  2. `solve(...)` runs the `pomdpsol` binary (see scripts/install_sarsop.sh).
  3. `SarsopPolicy.load(path, env)` reads the alpha vectors and acts greedily on them.

States are (cell, rock values) plus, with a battery, the battery level:
|S| = (capacity + 1) * n^2 * 2^k + 1. The battery multiplies |S| (and every
alpha vector's length), so with a battery use a small map, e.g. 5x5 with 4
rocks; `write_pomdp` refuses models above `MAX_STATES`. Hazard maps are not
exported.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .belief import RockBelief
from .env import NUM_BASIC_ACTIONS, NUM_OBS, RockSampleEnv, State
from .policies import Policy

MAX_STATES = 50_000  # beyond this the .pomdp file and alpha vectors get impractical

DEFAULT_POMDPSOL = Path(__file__).resolve().parent.parent / "third_party" / "sarsop" / "src" / "pomdpsol"


class StateIndex:
    """Bijection between `State`s and integers, terminal last:
    ((battery * n^2 + cell) * 2^k + rock_bits). Without a battery, battery = 0
    and this is (cell * 2^k + rock_bits), matching the published RockSample file."""

    def __init__(self, env: RockSampleEnv):
        self.n, self.k = env.cfg.grid_size, env.cfg.num_rocks
        self.num_configs = 2**self.k
        cap = env.cfg.battery_capacity
        self.battery_levels = 1 if cap is None else cap + 1
        self.terminal = self.battery_levels * self.n * self.n * self.num_configs
        self.size = self.terminal + 1

    def block(self, pos: tuple[int, int], battery: int = 0) -> int:
        """Index of the first of the 2^k states at this cell and battery level."""
        return (battery * self.n * self.n + pos[1] * self.n + pos[0]) * self.num_configs

    def index(self, s: State) -> int:
        if s.terminal:
            return self.terminal
        bits = sum(1 << i for i, good in enumerate(s.rocks) if good)
        return self.block(s.pos, s.battery) + bits

    def states(self):
        for battery, y, x in itertools.product(range(self.battery_levels), range(self.n), range(self.n)):
            for bits in range(self.num_configs):
                yield State((x, y), tuple(bool(bits >> i & 1) for i in range(self.k)), battery=battery)
        yield State((0, 0), (False,) * self.k, terminal=True)


def initial_distribution(env: RockSampleEnv, idx: StateIndex) -> np.ndarray:
    b = np.zeros(idx.size)
    belief = RockBelief.initial(env)
    base = idx.block(belief.pos, belief.battery)
    b[base : base + idx.num_configs] = config_probs(belief.p_good)
    return b


def config_probs(p_good: np.ndarray) -> np.ndarray:
    """Probability of each rock configuration (bit i = rock i good) under independent rocks."""
    k = len(p_good)
    bits = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
    return np.prod(np.where(bits == 1, p_good, 1.0 - p_good), axis=1)


def write_pomdp(env: RockSampleEnv, path: str | os.PathLike) -> StateIndex:
    idx = StateIndex(env)
    if idx.size > MAX_STATES:
        raise ValueError(
            f"{idx.size} states is too many for SARSOP (limit {MAX_STATES}); "
            "use a smaller map or battery capacity"
        )
    lines = [
        f"discount: {env.cfg.discount}",
        "values: reward",
        f"states: {idx.size}",
        f"actions: {env.num_actions}",
        f"observations: {NUM_OBS}",
        "start: " + " ".join(f"{p:.10g}" for p in initial_distribution(env, idx)),
        "",
    ]
    for s in idx.states():
        si = idx.index(s)
        for a in range(env.num_actions):
            sp, r = env.transition(s, a)
            spi = idx.index(sp)
            lines.append(f"T: {a} : {si} : {spi} 1.0")
            if r != 0:
                lines.append(f"R: {a} : {si} : * : * {r:.10g}")
        # O(a, s', o) must be defined for every s', including states no check
        # action leads to (e.g. a full battery), so write it per state, not per transition.
        for a in range(NUM_BASIC_ACTIONS, env.num_actions):
            for o, p in enumerate(env.observation_probs(a, s)):
                if p > 0:
                    lines.append(f"O: {a} : {si} : {o} {p:.10g}")
    for a in range(NUM_BASIC_ACTIONS):
        lines.append(f"O: {a} : * : 0 1.0")
    Path(path).write_text("\n".join(lines) + "\n")
    return idx


def find_pomdpsol() -> str:
    exe = os.environ.get("POMDPSOL") or shutil.which("pomdpsol")
    if exe:
        return exe
    if DEFAULT_POMDPSOL.exists():
        return str(DEFAULT_POMDPSOL)
    raise FileNotFoundError("pomdpsol not found; run scripts/install_sarsop.sh or set $POMDPSOL")


def solve(pomdp_path, policy_path, timeout: float = 60.0, precision: float = 1e-3) -> None:
    subprocess.run(
        [
            find_pomdpsol(),
            str(pomdp_path),
            "--timeout", str(timeout),
            "--precision", str(precision),
            "--output", str(policy_path),
        ],
        check=True,
    )


def read_alpha_vectors(path, num_states: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse a SARSOP .policy file into (alphas [num_vectors, num_states], actions [num_vectors])."""
    root = ET.parse(path).getroot()
    alphas, actions = [], []
    for vec in root.iter():
        if vec.tag == "Vector":
            alpha = np.array(vec.text.split(), dtype=float)
        elif vec.tag == "SparseVector":
            alpha = np.zeros(num_states)
            for entry in vec.iter("Entry"):
                i, v = entry.text.split()
                alpha[int(i)] = float(v)
        else:
            continue
        alphas.append(alpha)
        actions.append(int(vec.attrib["action"]))
    return np.array(alphas), np.array(actions)


class SarsopPolicy(Policy):
    name = "sarsop"

    def __init__(self, alphas: np.ndarray, actions: np.ndarray, idx: StateIndex):
        self.alphas, self.actions, self.idx = alphas, actions, idx

    @classmethod
    def load(cls, policy_path, env: RockSampleEnv) -> "SarsopPolicy":
        idx = StateIndex(env)
        alphas, actions = read_alpha_vectors(policy_path, idx.size)
        return cls(alphas, actions, idx)

    def value(self, belief: RockBelief) -> float:
        return float(self._values(belief).max())

    def _values(self, belief: RockBelief) -> np.ndarray:
        if belief.terminal:
            return self.alphas[:, self.idx.terminal]
        base = self.idx.block(belief.pos, belief.battery)
        block = self.alphas[:, base : base + self.idx.num_configs]
        return block @ config_probs(belief.p_good)

    def act(self, belief):
        return int(self.actions[np.argmax(self._values(belief))])
