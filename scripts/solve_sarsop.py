"""Export the RockSample model and solve it with SARSOP.

    python scripts/solve_sarsop.py --timeout 300
    python scripts/evaluate.py --policies greedy sarsop
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rover import RockBelief, RockSampleEnv
from rover.sarsop import SarsopPolicy, solve, write_pomdp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--precision", type=float, default=1e-3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = RockSampleEnv()
    pomdp, policy = out / "rocksample.pomdp", out / "rocksample.policy"

    idx = write_pomdp(env, pomdp)
    print(f"wrote {pomdp} ({idx.size} states, {env.num_actions} actions)")
    solve(pomdp, policy, timeout=args.timeout, precision=args.precision)
    p = SarsopPolicy.load(policy, env)
    print(f"{len(p.actions)} alpha vectors; V(b0) lower bound = {p.value(RockBelief.initial(env)):.3f}")


if __name__ == "__main__":
    main()
