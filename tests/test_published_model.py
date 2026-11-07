"""Check our RockSample(7,8) against the model file shipped with SARSOP.

Requires third_party/sarsop (run scripts/install_sarsop.sh); skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pytest

from rover import RockSampleEnv
from rover.env import EAST, NORTH, OBS_BAD, OBS_GOOD, SAMPLE, SOUTH, WEST
from rover.sarsop import StateIndex

PUBLISHED = Path(__file__).resolve().parent.parent / "third_party/sarsop/examples/POMDP/RockSample_7_8.pomdp"

# Published action order: amn ame ams amw ac0..ac7 as
OUR_ACTION = {0: NORTH, 1: EAST, 2: SOUTH, 3: WEST, 12: SAMPLE, **{4 + i: 5 + i for i in range(8)}}


def parse(path):
    T, R, O = {}, {}, {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(":")]
        if parts[0] == "T":
            sp, _ = parts[3].split()
            T[int(parts[1]), int(parts[2])] = int(sp)
        elif parts[0] == "R":
            R[int(parts[1]), int(parts[2])] = float(parts[4].split()[1])
        elif parts[0] == "O":
            o, p = parts[3].split()
            O.setdefault((int(parts[1]), int(parts[2])), {})[int(o)] = float(p)
    return T, R, O


@pytest.mark.skipif(not PUBLISHED.exists(), reason="SARSOP examples not downloaded")
def test_matches_published_rocksample_7_8():
    env = RockSampleEnv()
    idx = StateIndex(env)
    T, R, O = parse(PUBLISHED)

    def published_index(s):
        # s<x><y><rock0>..<rock7>: y varies fastest, rock0 is the most significant bit
        if s.terminal:
            return 12544
        bits = sum(1 << (7 - i) for i, good in enumerate(s.rocks) if good)
        return (s.pos[0] * 7 + s.pos[1]) * 256 + bits

    states = list(idx.states())
    assert len(states) == 12545
    for s in states:
        ps = published_index(s)
        for pa, a in OUR_ACTION.items():
            sp, r = env.transition(s, a)
            assert published_index(sp) == T[pa, ps], (s, a)
            assert r == pytest.approx(R.get((pa, ps), 0.0)), (s, a)
            if pa in range(4, 12) and not sp.terminal:
                ours = env.observation_probs(a, sp)
                theirs = O[pa, published_index(sp)]
                assert ours[OBS_GOOD] == pytest.approx(theirs[0], abs=1e-6)
                assert ours[OBS_BAD] == pytest.approx(theirs[1], abs=1e-6)
