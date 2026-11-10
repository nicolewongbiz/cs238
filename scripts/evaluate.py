"""Compare policies on the rover POMDP via Monte Carlo simulation.

    python scripts/evaluate.py --episodes 1000
    python scripts/evaluate.py --policies greedy sarsop --sarsop-policy results/rocksample.policy
    python scripts/evaluate.py --policies greedy rollout --episodes 200 --rollouts 20 --risk-alpha 0.2
    python scripts/evaluate.py --policies greedy pomcp --episodes 200 --sims 1000 --jobs 0
    python scripts/evaluate.py --policies greedy rollout --battery 30 --risk-alpha 0.2 --jobs 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rover import GreedyExpectedValuePolicy, NearestTargetPolicy, RandomPolicy, RockSampleEnv, RoverConfig, evaluate
from rover.planners import POMCPPlanner, RolloutPlanner

POLICIES = {
    "random": RandomPolicy,
    "nearest": NearestTargetPolicy,
    "greedy": GreedyExpectedValuePolicy,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["random", "nearest", "greedy"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sarsop-policy", default="results/rocksample.policy")
    ap.add_argument("--rollouts", type=int, default=20, help="rollouts per action (rollout planner)")
    ap.add_argument("--risk-alpha", type=float, default=1.0, help="CVaR level; 1.0 = risk-neutral mean")
    ap.add_argument("--sims", type=int, default=1000, help="simulations per step (POMCP)")
    ap.add_argument("--exploration", type=float, default=10.0, help="UCB constant c (POMCP)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes; 0 = all cores")
    ap.add_argument("--battery", type=int, default=None, help="battery capacity (default: no battery)")
    args = ap.parse_args()

    env = RockSampleEnv(RoverConfig(battery_capacity=args.battery))
    print(
        f"{'policy':<16} {'return':>16} {'returned':>9} {'failed':>7} {'good':>6} {'bad':>6} "
        f"{'steps':>7} {'energy':>7}"
    )
    for name in args.policies:
        if name == "sarsop":
            from rover.sarsop import SarsopPolicy

            policy = SarsopPolicy.load(args.sarsop_policy, env)
        elif name == "rollout":
            policy = RolloutPlanner(num_rollouts=args.rollouts, risk_alpha=args.risk_alpha, seed=args.seed)
        elif name == "pomcp":
            policy = POMCPPlanner(num_sims=args.sims, exploration=args.exploration, seed=args.seed)
        else:
            policy = POLICIES[name]()
        r = evaluate(env, policy, episodes=args.episodes, seed=args.seed, jobs=args.jobs)
        print(
            f"{policy.name:<16} {r['mean_return']:>8.2f} ± {r['stderr']:<5.2f} {r['return_rate']:>9.2f} "
            f"{r['failure_rate']:>7.2f} {r['good_samples']:>6.2f} {r['bad_samples']:>6.2f} "
            f"{r['steps']:>7.1f} {r['energy_used']:>7.1f}"
        )


if __name__ == "__main__":
    main()
