"""SARSOP near-optimal reference policy (Kurniawati, Hsu & Lee, 2008).

Pipeline:
  1. `write_pomdp(env, path)` exports the RockSample model in Cassandra .pomdp format.
  2. `solve(...)` runs the `pomdpsol` binary (see scripts/install_sarsop.sh).
  3. `SarsopPolicy.load(path, env)` reads the alpha vectors and acts greedily on them.

Only the plain RockSample model is exported (|S| = n^2 * 2^k + 1). Adding
battery levels or hazard maps multiplies |S|, so keep extensions small or use
SARSOP only on the base model as a reference point.
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

DEFAULT_POMDPSOL = Path(__file__).resolve().parent.parent / "third_party" / "sarsop" / "src" / "pomdpsol"


class StateIndex:
    """Bijection between `State`s and integers: (cell * 2^k + rock_bits), terminal last."""

    def __init__(self, env: RockSampleEnv):
        self.n, self.k = env.cfg.grid_size, env.cfg.num_rocks
        self.num_configs = 2**self.k
        self.terminal = self.n * self.n * self.num_configs
        self.size = self.terminal + 1

    def index(self, s: State) -> int:
        if s.terminal:
            return self.terminal
        bits = sum(1 << i for i, good in enumerate(s.rocks) if good)
        return (s.pos[1] * self.n + s.pos[0]) * self.num_configs + bits

    def states(self):
        for y, x in itertools.product(range(self.n), range(self.n)):
            for bits in range(self.num_configs):
                yield State((x, y), tuple(bool(bits >> i & 1) for i in range(self.k)))
        yield State((0, 0), (False,) * self.k, terminal=True)


def initial_distribution(env: RockSampleEnv, idx: StateIndex) -> np.ndarray:
    b = np.zeros(idx.size)
    belief = RockBelief.initial(env)
    base = idx.index(State(belief.pos, (False,) * idx.k))
    b[base : base + idx.num_configs] = config_probs(belief.p_good)
    return b


def config_probs(p_good: np.ndarray) -> np.ndarray:
    """Probability of each rock configuration (bit i = rock i good) under independent rocks."""
    k = len(p_good)
    bits = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
    return np.prod(np.where(bits == 1, p_good, 1.0 - p_good), axis=1)


def write_pomdp(env: RockSampleEnv, path: str | os.PathLike) -> StateIndex:
    idx = StateIndex(env)
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
            if a < NUM_BASIC_ACTIONS:
                continue
            for o, p in enumerate(env.observation_probs(a, sp)):
                if p > 0:
                    lines.append(f"O: {a} : {spi} : {o} {p:.10g}")
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
        base = self.idx.index(State(belief.pos, (False,) * self.idx.k))
        block = self.alphas[:, base : base + self.idx.num_configs]
        return block @ config_probs(belief.p_good)

    def act(self, belief):
        return int(self.actions[np.argmax(self._values(belief))])
