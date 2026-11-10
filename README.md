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

Optional SARSOP reference (builds [AdaCompNUS/sarsop](https://github.com/AdaCompNUS/sarsop) into `third_party/`; the script patches the build for Apple Silicon and recent clang):

```bash
./scripts/install_sarsop.sh
python scripts/solve_sarsop.py --timeout 300
python scripts/evaluate.py --policies nearest greedy sarsop
python scripts/evaluate.py --policies greedy rollout pomcp --episodes 200 --jobs 0
```

## Layout

| Path | Contents |
|---|---|
| `rover/env.py` | RockSample generative model and config (`RoverConfig`) |
| `rover/belief.py` | Exact factored belief (per-rock P(good)) with Bayes updates |
| `rover/policies.py` | Baselines: random, nearest-target, greedy expected value |
| `rover/planners.py` | Planners: Monte Carlo rollout with optional CVaR risk scoring, POMCP |
| `rover/simulate.py` | Episode runner and Monte Carlo evaluation metrics |
| `rover/sarsop.py` | `.pomdp` export, `pomdpsol` wrapper, alpha-vector policy |
| `scripts/` | Experiment entry points |
| `tests/` | pytest suite |
| `report/` | LaTeX proposal and final report |

## Results (RockSample(7,8), γ = 0.95)

Episode *i* uses the same hidden rock values for every policy (`evaluate(..., seed)` seeds each episode separately), so policies can be compared episode by episode.

| Policy | Episodes | Discounted return | Returned to base | Good / bad samples | Steps |
|---|---|---|---|---|---|
| random | 2000 | −74.28 ± 0.53 | 0% | 0.03 / 0.03 | 8.4 |
| nearest-target | 2000 | 2.10 ± 0.35 | 100% | 3.99 / 4.01 | 31.0 |
| greedy expected value | 2000 | 17.00 ± 0.12 | 100% | 3.76 / 0.24 | 29.2 |
| rollout (20 per action, greedy base) | 200 | 17.22 ± 0.36 | 100% | 3.85 / 0.05 | 33.0 |
| POMCP (1000 sims, c = 10, greedy rollouts) | 200 | 16.14 ± 0.36 | 100% | 3.85 / 0.12 | 33.6 |
| SARSOP (300 s) | 2000 | 21.26 ± 0.14 | 100% | 3.83 / 0.10 | 28.9 |

Paired against greedy on the same 200 episodes: rollout +0.44 ± 0.24, POMCP −0.64 ± 0.26.

- **SARSOP** bounds on V(b₀) after 300 s are [20.92, 24.76]; Kurniawati et al. (2008) report ≈ 21.27, so SARSOP is the near-optimal reference.
- **Rollout** is limited by its greedy base policy: 50 rollouts per action does no better than 20.
- **POMCP** is short of simulations, not wrong: on a 5×5, 4-rock model it improves from 15.30 (1k sims) to 16.41 (5k) toward SARSOP's 17.34. Pure Python manages about 1000 sims in 0.3 s per decision; published POMCP results on RockSample(7,8) use far more.
- Cost per episode (one core): rollout ≈ 0.9 s, POMCP ≈ 10 s. Use `--jobs 0` to spread episodes over all cores.

## Battery results (RockSample(7,8) + battery, 200 episodes)

Energy per action: move 1, check 1, sample 2. Running out before exiting strands the rover (−100, mission fails). Greedy uses about 33 energy per mission with no battery limit.

| Capacity | Policy | Discounted return | Failed | Returned | Good / bad samples |
|---|---|---|---|---|---|
| 25 | greedy | −16.44 ± 0.81 | 87% | 13% | 2.46 / 0.14 |
| 25 | rollout | −5.18 ± 1.19 | 55% | 45% | 2.27 / 0.09 |
| 25 | rollout, CVaR 0.2 | **1.74 ± 1.11** | **30%** | 70% | 1.74 / 0.02 |
| 35 | greedy | 7.93 ± 0.77 | 37% | 63% | 3.47 / 0.21 |
| 35 | rollout | **9.73 ± 0.69** | 24% | 76% | 3.34 / 0.14 |
| 35 | rollout, CVaR 0.2 | 8.33 ± 0.64 | 23% | 77% | 2.87 / 0.04 |
| 50 | greedy | 16.78 ± 0.39 | 0% | 100% | 3.73 / 0.23 |
| 50 | rollout | **17.18 ± 0.36** | 0% | 100% | 3.84 / 0.04 |
| 50 | rollout, CVaR 0.2 | 13.78 ± 0.28 | 0% | 100% | 3.63 / 0.03 |

- With a tight battery, greedy (battery-blind) is stranded most of the time. Rollout cuts failures, and CVaR scoring cuts them further and gives the best return.
- With an ample battery, CVaR's caution costs about 3.4 points of return: the price of risk aversion when nothing can go wrong.
- Energy use is deterministic and fully observed, so every failure comes from short-sighted planning. Rollout still fails often at capacity 25 because its greedy base policy ignores the battery.

## Roadmap

- [x] RockSample environment, exact belief, baseline policies, evaluation harness
- [x] SARSOP export / policy loader
- [x] Default to the published RockSample(7,8) layout, verified against SARSOP's model file
- [x] Build SARSOP and record its value on RockSample(7,8) as the near-optimal reference
- [x] Rollout-lookahead planner with CVaR risk scoring in `rover/planners.py`
- [x] POMCP planner (needs more simulations or speed to beat greedy)
- [x] Battery budget: per-action energy use, failure if depleted before returning
- [ ] Terrain hazards: hidden hazard classes per cell, stochastic delay or immobilization, noisy hazard scans
- [ ] Experiments: sweeps over sensor noise (`sensor_half_eff_dist`) and hazard probability, science–safety tradeoff plots
- [ ] *Extension:* continuous dynamics with a CBF safety filter

## References

- T. Smith and R. Simmons. Heuristic Search Value Iteration for POMDPs. UAI, 2004.
- H. Kurniawati, D. Hsu, and W. S. Lee. SARSOP: Efficient Point-Based POMDP Planning by Approximating Optimally Reachable Belief Spaces. RSS, 2008.
- D. Silver and J. Veness. Monte-Carlo Planning in Large POMDPs. NeurIPS, 2010.
- M. J. Kochenderfer, T. A. Wheeler, and K. H. Wray. *Algorithms for Decision Making*. MIT Press, 2022.
