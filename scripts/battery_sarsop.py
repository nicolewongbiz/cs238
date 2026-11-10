"""Compare policies with SARSOP under a battery constraint, on a map small enough
for SARSOP to solve with the battery in the state (RockSample(5,4)).

    python scripts/battery_sarsop.py --capacities 8 12 16 24 --timeout 120

For every capacity this exports and solves the battery-constrained POMDP, then
evaluates all policies on the same missions. Results go to
results/battery_sarsop.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rover import BatteryAwareGreedyPolicy, GreedyExpectedValuePolicy, RockBelief, RockSampleEnv, RoverConfig, evaluate
from rover.planners import RolloutPlanner
from rover.sarsop import SarsopPolicy, solve, write_pomdp

# RockSample(5,4): 5x5 grid, 4 rocks. 401 states without a battery.
SMALL_MAP = dict(grid_size=5, rocks=((1, 1), (3, 3), (2, 4), (4, 0)), start=(0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacities", type=int, nargs="+", default=[8, 12, 16, 24])
    ap.add_argument("--skip-reference", action="store_true", help="skip the no-battery reference run")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=120.0, help="SARSOP time per capacity (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capacities = ([] if args.skip_reference else [None]) + args.capacities
    rows = []
    for cap in capacities:
        env = RockSampleEnv(RoverConfig(**SMALL_MAP, battery_capacity=cap))
        tag = "none" if cap is None else str(cap)
        pomdp, policy_file = out / f"battery_{tag}.pomdp", out / f"battery_{tag}.policy"
        idx = write_pomdp(env, pomdp)
        print(f"\n== battery {tag}: {idx.size} states, solving SARSOP for {args.timeout:g}s ...", flush=True)
        solve(pomdp, policy_file, timeout=args.timeout)
        sarsop = SarsopPolicy.load(policy_file, env)
        v0 = sarsop.value(RockBelief.initial(env))

        policies = [
            (GreedyExpectedValuePolicy(), args.jobs),
            (BatteryAwareGreedyPolicy(), args.jobs),
            (RolloutPlanner(base_policy=BatteryAwareGreedyPolicy, seed=args.seed), args.jobs),
            (RolloutPlanner(base_policy=BatteryAwareGreedyPolicy, risk_alpha=0.2, seed=args.seed), args.jobs),
            (sarsop, 1),  # cheap to run; avoid copying its alpha vectors to every worker
        ]
        print(f"{'policy':<20} {'return':>16} {'failed':>7} {'good':>6} {'bad':>6} {'steps':>6}")
        for policy, jobs in policies:
            r = evaluate(env, policy, episodes=args.episodes, seed=args.seed, jobs=jobs)
            print(
                f"{policy.name:<20} {r['mean_return']:>8.2f} ± {r['stderr']:<5.2f} {r['failure_rate']:>7.2f} "
                f"{r['good_samples']:>6.2f} {r['bad_samples']:>6.2f} {r['steps']:>6.1f}",
                flush=True,
            )
            rows.append({"capacity": tag, "states": idx.size, "sarsop_lower_bound": v0, **r})
        print(f"SARSOP lower bound V(b0) = {v0:.2f}")

    path = out / "battery_sarsop.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (float(v) if hasattr(v, "item") else v) for k, v in row.items()} for row in rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
