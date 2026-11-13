# CS238 final project: risk-aware Mars rover (Nicole Wong, Tamir Ovadia)

RockSample(7,8) POMDP with baselines, a SARSOP reference, and online planners (rollout + CVaR, POMCP). We are now adding battery and terrain hazards. Read `docs/walkthrough.md` for how everything works and `README.md` for current results.

## Setup and checks

```bash
source .venv/bin/activate && pip install -e '.[dev]'
pytest                                   # must pass before every PR
./scripts/install_sarsop.sh              # once per machine; SARSOP lives in third_party/ (gitignored)
python scripts/solve_sarsop.py --timeout 300
python scripts/evaluate.py --policies greedy rollout --episodes 200 --jobs 0
```

## Who owns what (one branch each, merge to `main` by PR, the other person reviews)

| Owner | Branch | Scope |
|---|---|---|
| Nicole | `battery` (then `hazards`) | `rover/env.py`, `rover/belief.py`: battery, then terrain hazards and hazard scans; their tests |
| Tamir | `sweeps` | `scripts/sweep.py` + plotting: parameter sweeps saved to CSV, figures for the report. First sweep: `sensor_half_eff_dist` in {5, 10, 20, 40} for greedy, rollout, SARSOP |

Don't edit the other person's files without asking; coordinate through PRs.

## Interface contract (agreed; change only by agreement)

New features are off by default (`None`), so plain RockSample and every published number stay unchanged.

```python
# RoverConfig additions
battery_capacity: int | None = None    # None = no battery
move_energy: int = 1                   # per N/S/E/W, including the exit move
check_energy: int = 1
sample_energy: int = 2
depletion_penalty: float = -100.0

# State gains:          battery: int  (last field, default 0)
# RockBelief gains:     battery: int  (fully observed, no Bayes needed)
# EpisodeResult gains:  failed: bool, energy_used: int
# evaluate() gains:     "failure_rate", "energy_used"
```

Depletion rule: an action needs its energy cost. If the remaining battery is less than the cost, the rover is stranded: the episode ends with `depletion_penalty`, `failed = True`, and the action doesn't happen. Exiting with exactly enough energy succeeds.

Rules for code that consumes the model (sweeps, planners):

- Sweep scripts read metrics generically from `evaluate()` / `EpisodeResult` fields, so new metrics (failure rate, energy, later hazard metrics) appear without code changes.
- Planners use only `env.step`, `env.transition`, `RockBelief.update` and `candidate_actions`, so they work on extended models unchanged.
- `write_pomdp` exports the battery level as part of the state, which multiplies the state count by (capacity + 1), and refuses models above 50k states. So SARSOP with a battery runs only on small maps; `scripts/battery_sarsop.py` uses RockSample(5,4). On the 7x7 map SARSOP is the reference without a battery.
- With the battery on, the baseline is `greedy-battery` (greedy plus a return-home rule), not battery-blind `greedy`, and the planners should use it as their base (`--base greedy-battery`).
- Hazard fields will be added to this contract by Nicole before hazard work starts.

## Conventions

- `evaluate(env, policy, episodes, seed, jobs)` seeds each episode separately, so episode *i* has the same hidden rocks for every policy (paired comparisons), and results don't depend on `jobs`. Keep it that way.
- `env.step` draws exactly one uniform per step; don't change its random draws without re-running every number in the README.
- Scripts that use `jobs > 1` need an `if __name__ == "__main__":` guard (macOS spawns worker processes).
- Course policy: AI-written code is fine if we review, test and can explain it. The status update and final report must be in our own words, so don't draft report prose.
