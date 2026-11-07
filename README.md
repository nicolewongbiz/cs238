# Risk-Aware Scientific Exploration on Mars

CS238 / AA228 final project — Nicole Wong, Tamir Ovadia.

An autonomous rover must decide where to explore, when extra measurements are worth their cost, and when exploring further is too risky. We model this as a POMDP and compare baseline policies against a risk-aware, belief-informed planner. The full proposal is in [report/proposal.tex](report/proposal.tex).

## Starting point: RockSample

Following instructor feedback, the core model is **RockSample** (Smith & Simmons, 2004): a rover on an *n×n* grid with *k* rocks of hidden value (good/bad), a distance-dependent noisy sensor, and an exit on the east edge that stands for returning to base. The default `RoverConfig()` is the published **RockSample(7,8)** benchmark (7×7 grid, 8 rocks, start (0,3), 12,545 states), small enough for **SARSOP** to give a near-optimal reference policy. `tests/test_published_model.py` checks that our model matches the `RockSample_7_8.pomdp` file shipped with SARSOP on every transition, reward and observation probability. We then add the project's own elements (battery budget, terrain hazards, sensing costs) on top of it.

Rules: sampling a good rock gives +10 (it then turns bad), sampling a bad rock −10, exiting east +10. Illegal actions (bumping a north/south/west wall, sampling an empty cell) give −100 and end the episode.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/evaluate.py --episodes 1000
```

Optional SARSOP reference (builds [AdaCompNUS/sarsop](https://github.com/AdaCompNUS/sarsop) into `third_party/`):

```bash
./scripts/install_sarsop.sh
python scripts/solve_sarsop.py --timeout 300
python scripts/evaluate.py --policies nearest greedy sarsop
```

## Layout

| Path | Contents |
|---|---|
| `rover/env.py` | RockSample generative model and config (`RoverConfig`) |
| `rover/belief.py` | Exact factored belief (per-rock P(good)) with Bayes updates |
| `rover/policies.py` | Baselines: random, nearest-target, greedy expected value |
| `rover/planners.py` | **TODO:** risk-aware planner (rollout / POMCP) |
| `rover/simulate.py` | Episode runner and Monte Carlo evaluation metrics |
| `rover/sarsop.py` | `.pomdp` export, `pomdpsol` wrapper, alpha-vector policy |
| `scripts/` | Experiment entry points |
| `tests/` | pytest suite |
| `report/` | LaTeX proposal and final report |

## Baseline results (RockSample(7,8), γ = 0.95, 2000 episodes)

| Policy | Discounted return | Returned to base | Good / bad samples |
|---|---|---|---|
| random | −74.16 ± 0.52 | 0% | 0.03 / 0.03 |
| nearest-target | 1.89 ± 0.35 | 100% | 3.97 / 4.03 |
| greedy expected value | 17.07 ± 0.12 | 100% | 3.78 / 0.24 |
| SARSOP | *pending build* | | |

## Roadmap

- [x] RockSample environment, exact belief, baseline policies, evaluation harness
- [x] SARSOP export / policy loader
- [x] Default to the published RockSample(7,8) layout, verified against SARSOP's model file
- [ ] Build SARSOP and record its value on RockSample(7,8) as the near-optimal reference
- [ ] Risk-aware planner (rollout lookahead → POMCP) in `rover/planners.py`
- [ ] Battery budget: per-action energy use, failure if depleted before returning
- [ ] Terrain hazards: hidden hazard classes per cell, stochastic delay or immobilization, noisy hazard scans
- [ ] Experiments: sweeps over sensor noise (`sensor_half_eff_dist`) and hazard probability, science–safety tradeoff plots
- [ ] *Extension:* continuous dynamics with a CBF safety filter

## References

- T. Smith and R. Simmons. Heuristic Search Value Iteration for POMDPs. UAI, 2004.
- H. Kurniawati, D. Hsu, and W. S. Lee. SARSOP: Efficient Point-Based POMDP Planning by Approximating Optimally Reachable Belief Spaces. RSS, 2008.
- D. Silver and J. Veness. Monte-Carlo Planning in Large POMDPs. NeurIPS, 2010.
- M. J. Kochenderfer, T. A. Wheeler, and K. H. Wray. *Algorithms for Decision Making*. MIT Press, 2022.
