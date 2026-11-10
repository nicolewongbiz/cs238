# Project Walkthrough: Risk-Aware Mars Rover Exploration

A guide to everything built so far: what each piece does, the ideas behind it, and what the results mean. It is meant for the two of us to understand and explain the code. It is not report text; the course requires the report to be in our own words.

Contents

1. [The big picture](#1-the-big-picture)
2. [Background: POMDPs in one page](#2-background-pomdps-in-one-page)
3. [The model: `rover/env.py`](#3-the-model-roverenvpy)
4. [The belief: `rover/belief.py`](#4-the-belief-roverbeliefpy)
5. [Baseline policies: `rover/policies.py`](#5-baseline-policies-roverpoliciespy)
6. [Evaluation: `rover/simulate.py`](#6-evaluation-roversimulatepy)
7. [SARSOP, the near-optimal reference: `rover/sarsop.py`](#7-sarsop-the-near-optimal-reference-roversarsoppy)
8. [Our planners: `rover/planners.py`](#8-our-planners-roverplannerspy)
9. [Tests](#9-tests)
10. [Results and what they mean](#10-results-and-what-they-mean)
11. [How to run everything](#11-how-to-run-everything)
12. [What's next](#12-whats-next)
13. [Glossary](#13-glossary)

---

## 1. The big picture

**Goal (from the proposal).** An autonomous Mars rover must decide where to explore, when an extra measurement is worth its cost, and when exploring further is too risky. Some things it cares about can't be seen directly, such as which sites are scientifically valuable. So we model the problem as a **POMDP** and compare simple baselines against a risk-aware planner that reasons about its beliefs.

**Plan.**

1. Start from the standard benchmark **RockSample** (Smith & Simmons, 2004), as the instructor suggested.
2. Build baseline policies: *nearest-target* and *greedy*.
3. Get a **near-optimal reference** policy from the SARSOP solver, so we know how good "good" is.
4. Build our own **planner**: rollout lookahead and POMCP, with a risk-aware scoring option.
5. *(Not done yet.)* Add **battery** and **terrain hazards** so that risk actually matters, then run experiments.

**Git history so far.**

| Commit | What it did |
|---|---|
| `c9a06d4` | Initial setup: rover model, baselines, SARSOP bridge |
| `0b64fc8` | Switched to the published RockSample(7,8) map; verified against SARSOP's own model file |
| `8dac1df` | Got SARSOP building on Apple Silicon; recorded its score |
| `0da62f5` | Rollout planner with CVaR risk scoring |
| `79494a0` | POMCP planner; reproducible parallel evaluation |

**Code layout.**

```
rover/env.py        the world: states, actions, rewards, noisy sensor
rover/belief.py     what the rover believes about the hidden rocks
rover/policies.py   baselines: random, nearest-target, greedy
rover/simulate.py   run missions, compute metrics
rover/sarsop.py     export model to SARSOP, load its solution as a policy
rover/planners.py   our planners: rollout (+ CVaR) and POMCP
scripts/            install_sarsop.sh, solve_sarsop.py, evaluate.py
tests/              pytest suite (23 tests)
```

---

## 2. Background: POMDPs in one page

A **Markov decision process (MDP)** has states *S*, actions *A*, transitions *P(s' | s, a)*, rewards *R(s, a)* and a discount factor γ. The agent always knows the state.

A **partially observable MDP (POMDP)** adds **observations** *O* with a model *P(o | s', a)*. The agent never sees the state directly, only noisy observations. So it keeps a **belief** *b(s)*, a probability distribution over states, and updates it with Bayes' rule after every action and observation.

Key consequences:

- **A policy maps beliefs to actions**, not states to actions. The right move depends on what you *know*.
- **Information-gathering actions have value.** Checking a rock earns nothing directly, but it improves the belief and therefore later decisions. A good POMDP policy trades the cost of looking (here, time) against the value of what it learns.
- **Solving exactly is intractable** for anything but tiny problems, because the belief space is continuous. Practical methods either approximate the value function over beliefs (SARSOP) or plan online from the current belief by simulation (rollout, POMCP).

**Discounting.** A reward *t* steps in the future is worth γᵗ times as much. With γ = 0.95, +10 reached 10 steps from now is worth 10 × 0.95¹⁰ ≈ 6.0. So wasted steps cost value, and policies are pushed to be efficient.

---

## 3. The model: `rover/env.py`

### RockSample(7,8)

| Element | In our model |
|---|---|
| Map | 7 × 7 grid; rover starts at (0, 3) |
| Hidden state | 8 rocks at known positions; each is secretly **good** or **bad** (50/50 prior) |
| Actions (13) | `north`, `south`, `east`, `west`, `sample`, `check0` … `check7` |
| Observations | `none` (after moves/samples), `good` or `bad` (after a check) |
| Discount | γ = 0.95 |

**Rewards.**

| Event | Reward |
|---|---|
| Sample a good rock | +10 (the rock then becomes bad, so it can't be farmed) |
| Sample a bad rock | −10 |
| Drive off the **east** edge ("return to base") | +10, mission ends |
| Bump a north/south/west wall, or sample empty ground | −100, mission ends |

**The noisy sensor.** `check i` reports rock *i*'s value correctly with probability

```
P(correct) = ½ (1 + η),   η = 2^(−d / d₀),   d₀ = 20
```

where *d* is the distance to the rock. At distance 0, η = 1 and the reading is perfect. As *d* grows, η shrinks and the reading approaches a coin flip. So checking from far away is cheap in movement but less informative. That creates the move-closer-or-check-now tradeoff.

### Key code

- **`RoverConfig`** holds every setting: grid size, rock positions, rewards, sensor range, discount and step limit. Battery and hazard settings will be added here.
- **`transition(s, a)`** returns the next state and reward. Movement and sampling are deterministic; only the sensor is random.
- **`observation_probs(a, s')`** gives the distribution over observations.
- **`step(s, a, rng)`** is the **generative model**: it returns next state, observation and reward. Every simulation (evaluation, rollouts, POMCP) goes through it. It is optimized to draw exactly one random number per step, which keeps runs reproducible and fast.

The default map is the **published** RockSample(7,8) layout, and a test checks it against the `.pomdp` file distributed with SARSOP (see [Tests](#9-tests)). So our numbers are directly comparable to the literature.

---

## 4. The belief: `rover/belief.py`

The full state is (position, 8 rock values), and there are 2⁸ = 256 possible rock combinations. A general belief would need a probability for each combination. Two facts make it much simpler:

1. **Position is fully observed**: the rover knows where it is.
2. **Rocks are independent**, and `check i` only tells you about rock *i*.

So the belief **factors** into one number per rock, **P(rock i is good)**. Eight numbers represent the belief *exactly*.

**Bayes update after `check i`** (`RockBelief.update`):

```
                     P(reading | good) · P(good)
P(good | reading) = ──────────────────────────────
                             P(reading)
```

*Worked example.* Rock 4 squares away: η = 2^(−4/20) ≈ 0.871, so P(correct) ≈ 0.935.
- Prior 0.5, reading "good": posterior = 0.935.
- A second "good" reading: posterior = 0.935² / (0.935² + 0.065²) ≈ 0.995.

**After `sample`**, the rock's probability becomes 0, because sampled rocks turn bad.

When hazards are added, hidden quantities that don't factor this neatly may need a **particle belief** (a set of sampled states) instead.

---

## 5. Baseline policies: `rover/policies.py`

Every policy has `reset(env, rng)` (called at the start of each mission) and `act(belief) → action`.

| Policy | Rule | Proposal role |
|---|---|---|
| `RandomPolicy` | Uniformly random action | Sanity floor |
| `NearestTargetPolicy` | Ignore the sensor; go to the nearest unvisited rock, sample it, repeat; then exit east | (1) fixed / nearest-target |
| `GreedyExpectedValuePolicy` | See below | (2) greedy expected-science |

**Greedy, step by step:**

1. If standing on a rock whose expected sample value is positive (P(good) > 0.5), **sample**.
2. Else, if some rock is uncertain (0.1 < P(good) < 0.9) and hasn't been checked 3 times yet, **check** the nearest such rock.
3. Else, score each rock as γ^distance × E[sample reward] and **move toward** the best one, if its score is positive.
4. Else **exit east**.

Greedy is a strong, sensible heuristic. Its blind spots: it doesn't weigh *how much* a check would help, it checks in a fixed order, and it has no notion of risk.

---

## 6. Evaluation: `rover/simulate.py`

- **`run_episode`** plays one mission. A hidden true state (actual rock values) is drawn at random. The policy only ever sees the belief, which is updated from the observations the simulator produces. The episode ends on exit, an illegal action, or after 200 steps.
- **`evaluate`** runs many missions and reports:
  - **mean discounted return** ± standard error (the main score);
  - **return-to-base rate**;
  - average **good / bad samples**;
  - average **steps**.

**Reproducibility and paired comparisons.** Episode *i* uses its own random seed for the environment and another for the policy. Two consequences:

- **Every policy faces the same hidden rock values in episode *i***. So we can compare policies *per mission* (a "paired" comparison), which removes most of the luck. That's how we got "rollout − greedy = +0.44 ± 0.24".
- **`jobs`** splits episodes across CPU cores and gives *identical* results to a serial run (there's a test for this). Use `--jobs 0` for all cores.

---

## 7. SARSOP, the near-optimal reference: `rover/sarsop.py`

### What SARSOP is

SARSOP (Kurniawati, Hsu & Lee, 2008) is an **offline POMDP solver**. Before the mission starts, it computes a policy for *every* belief the rover might reach. We use it as a **reference line**: an estimate of the best achievable score, so our own planner can be judged against both the baselines and near-optimal behavior.

### How it works

**1. The value function is made of "alpha vectors."** A POMDP's optimal value as a function of the belief, *V(b)*, has a special shape: piecewise-linear and convex. It can be written as the maximum over a set of vectors:

```
V(b) = max over α of  Σₛ α(s) · b(s)
```

Each **alpha vector** α is a list of values, one per state. Each one corresponds to a conditional plan ("take action *a*, then do this or that depending on what you see"), and α(s) is that plan's value if the true state were *s*. The policy is: compute α·b for every vector and take the action of the best one.

**2. It is point-based.** Computing every alpha vector is hopeless. Point-based methods instead pick a set of sample beliefs and improve the value function only at those points, using a **Bellman backup** that adds or replaces alpha vectors. Each new vector also improves the estimate at nearby beliefs.

**3. Its key idea is to focus on the beliefs that matter.** Most of belief space is never visited by a good policy. SARSOP samples beliefs by simulating forward from the starting belief *b₀*, steering toward beliefs the *optimal* policy would plausibly reach (the "optimally reachable belief space"). To steer, it keeps two bounds on the optimal value:
   - a **lower bound**: the alpha vectors, i.e. the value of an actual policy;
   - an **upper bound**: an optimistic estimate.

   It explores where the gap between them is large, because that's where more work can still change the decision.

**4. It stops** when the gap at *b₀* is below the requested precision, or at the timeout. The output `.policy` file is the set of alpha vectors, which is the lower-bound policy.

**Reading the solver output.** After 300 s on RockSample(7,8), SARSOP reported bounds **[20.92, 24.76]**. The true optimal value lies somewhere in between, and the policy we use is guaranteed to be worth at least about 20.92 in expectation. In simulation it scored **21.26**, matching the published ≈ 21.27.

### Our pipeline

1. **Export** (`write_pomdp`). Writes our model in the standard Cassandra `.pomdp` text format: every transition, reward and observation probability.
   - **States:** 49 cells × 256 rock combinations + 1 "done" state = **12,545**.
   - **`StateIndex`** numbers them as `cell × 256 + rock_bits`, where bit *i* = 1 means rock *i* is good.
2. **Solve** (`solve`, `scripts/solve_sarsop.py`). Runs the `pomdpsol` program, here for 300 s.
3. **Load and act** (`SarsopPolicy`). Reads the alpha vectors from the XML policy file. To act from our factored belief:
   - position is known, so only the 256 entries of each α for the current cell matter;
   - `config_probs` turns the 8 per-rock probabilities into probabilities for all 256 combinations (they're independent, so multiply);
   - the value of each vector is a dot product, and the policy takes the action of the best vector.

### Building it

`scripts/install_sarsop.sh` downloads SARSOP into `third_party/` and builds it. It patches three things for macOS on Apple Silicon:
- removes Intel-only (SSE) compiler flags;
- fixes a C++ standard-library incompatibility in `SparseMatrix.h`;
- tells the newer compiler to accept the old-style C in SARSOP's `.pomdp` parser.

### Limitation

The state count multiplies with every hidden variable we add. Ten battery levels would mean about 125k states; a per-cell hazard map would be astronomically more. So **SARSOP is only usable on the base model** or very small extensions. Online planners have to handle the full risk-aware model.

---

## 8. Our planners: `rover/planners.py`

Both are **online** planners. At every step they plan from the *current* belief by simulating possible futures with the generative model. That scales to models far too large for SARSOP.

### Shared helpers

- **`candidate_actions`** removes pointless actions before planning:
  - moves into walls;
  - sampling where there's no rock (or the rock was already sampled);
  - checking rocks we're already more than 99% sure about.

  The last rule fixed a real bug. POMCP's value estimates were noisy enough that it sometimes re-checked known-bad rocks forever, since each wasted check only costs one discount step.
- **`base_rollout`** simulates the rest of a mission from a given (state, belief) by following the greedy policy, updating the belief along the way exactly as the real rover would.

### Rollout planner (`RolloutPlanner`)

Rollout is a one-step lookahead with Monte Carlo evaluation (Bertsekas' rollout algorithm; *Algorithms for Decision Making* §22.3). At every step:

1. Sample 20 possible hidden rock configurations from the current belief.
2. For each candidate action and each sample: take the action, then let greedy finish the mission. Record the discounted return.
3. Score each action from its 20 returns and take the best.

Details that matter:

- **Common random numbers.** Every action is tested on the *same* 20 samples and random streams, so differences between actions reflect the actions, not luck.
- **Theory.** Rollout is guaranteed to do at least as well as its base policy (greedy), up to sampling noise. It "fixes" greedy's mistakes one step at a time.
- **Risk awareness via CVaR.** With `risk_alpha = 1` an action is scored by its *mean* return. With `risk_alpha < 1` it's scored by **CVaR_α**, the average of the worst α-fraction of outcomes. For example, α = 0.2 judges each action by its worst 20% of simulated futures. That makes the planner prefer actions whose bad cases are less bad, which is the point once battery depletion and getting stuck are possible. On plain RockSample the only catastrophe is an illegal action, which pruning already prevents, so CVaR has little to act on yet.

### POMCP (`POMCPPlanner`)

POMCP, Partially Observable Monte Carlo Planning (Silver & Veness, 2010; *Algorithms for Decision Making* §22.6), applies Monte Carlo tree search to POMDPs. It builds a search tree whose nodes are **histories**: action → observation → action → … At every step it runs 1000 simulations, and each one does the following:

1. **Sample** a hidden state from the current belief.
2. **Descend** the tree. At each node, choose an action by **UCB1**:

   ```
   Q(a) + c · √( ln N / n(a) )
   ```

   This is the average return so far plus an exploration bonus that's large for rarely tried actions. *c* (default 10) sets the balance. The simulator decides which observation branch to follow.
3. **Expand and estimate.** On reaching a history not yet in the tree, add it and estimate its value with one greedy rollout.
4. **Back up.** Update the visit counts and running-average Q along the path.

After all simulations, take the root action with the highest average.

Unlike rollout, POMCP looks many steps ahead, so with enough simulations it converges toward optimal behavior. Because our belief is exact, each step samples directly from it and builds a fresh tree, so no particle filter is needed.

---

## 9. Tests

`pytest` runs 23 tests in about 7 s.

| File | What it checks |
|---|---|
| `test_env.py` | Movement, walls, exit, sampling, sensor accuracy, observation model; belief Bayes updates and position/sampling tracking |
| `test_published_model.py` | Our model matches SARSOP's `RockSample_7_8.pomdp` on **every** transition, reward and observation probability |
| `test_sarsop.py` | State numbering is a one-to-one mapping; start distribution; rock-combination probabilities; `.pomdp` export; **end to end**: solve a tiny model with `pomdpsol`, load the policy, and check its simulated return matches SARSOP's value |
| `test_policies.py` | Baselines and planners finish and return to base; greedy beats nearest; CVaR math; action pruning; POMCP tree counts are consistent; parallel and serial evaluation give identical results |

---

## 10. Results and what they mean

RockSample(7,8), γ = 0.95.

| Policy | Episodes | Discounted return | Returned | Good / bad samples | Steps |
|---|---|---|---|---|---|
| random | 2000 | −74.28 ± 0.53 | 0% | 0.03 / 0.03 | 8.4 |
| nearest-target | 2000 | 2.10 ± 0.35 | 100% | 3.99 / 4.01 | 31.0 |
| greedy | 2000 | 17.00 ± 0.12 | 100% | 3.76 / 0.24 | 29.2 |
| rollout (20 per action) | 200 | 17.22 ± 0.36 | 100% | 3.85 / 0.05 | 33.0 |
| POMCP (1000 sims) | 200 | 16.14 ± 0.36 | 100% | 3.85 / 0.12 | 33.6 |
| **SARSOP (300 s)** | 2000 | **21.26 ± 0.14** | 100% | 3.83 / 0.10 | 28.9 |

Paired against greedy on the same 200 missions: **rollout +0.44 ± 0.24**, **POMCP −0.64 ± 0.26**.

**What to take from this.**

- **Using the sensor matters most.** Nearest-target samples every rock blindly (about 4 bad samples per mission); greedy checks first and gains about 15 points.
- **There's room above greedy.** SARSOP is about 4 points higher. It wastes fewer steps and makes almost no bad samples.
- **Rollout helps a little.** It nearly eliminates bad samples (0.05 vs 0.24) but spends more steps. Going from 20 to 50 rollouts per action didn't help, so it's limited by its greedy base policy.
- **POMCP is short of simulations, not wrong.** On a smaller 5×5, 4-rock map it improves steadily (15.30 at 1k → 16.41 at 5k simulations) toward SARSOP's 17.34. Pure Python affords about 1000 simulations per decision (about 0.3 s); published POMCP results use far more.
- **The real test hasn't happened yet.** On plain RockSample nothing truly risky can happen, so a risk-aware planner has nothing to show. Its advantage should appear once battery and hazards create real failure modes that greedy ignores.

**Speed** (per episode, one core): greedy is instant, rollout about 0.9 s, POMCP about 10 s. Use `--jobs 0`.

---

## 11. How to run everything

```bash
# one-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest

# baselines
python scripts/evaluate.py --episodes 1000

# SARSOP reference (build once, ~5 min solve)
./scripts/install_sarsop.sh
python scripts/solve_sarsop.py --timeout 300
python scripts/evaluate.py --policies nearest greedy sarsop --episodes 2000

# planners (use all cores)
python scripts/evaluate.py --policies greedy rollout pomcp --episodes 200 --jobs 0

# planner knobs
#   --rollouts N      rollouts per action (rollout planner)
#   --risk-alpha A    CVaR level; 1.0 = mean, 0.2 = worst 20%
#   --sims N          simulations per step (POMCP)
#   --exploration C   UCB constant (POMCP)
```

`results/` (SARSOP files) and `third_party/` (the SARSOP source) are not in git; each machine builds its own.

---

## 12. What's next

1. **Battery and terrain hazards (the main contribution).**
   - **Battery:** every action uses energy; the mission fails if the battery runs out before returning.
   - **Hazards:** hidden hazard classes per cell, with stochastic delays or immobilization, and noisy hazard scans.
   - These go into `RoverConfig`, `State`, `step` and the belief.
2. **Planners on the extended model.** Greedy vs rollout with CVaR α ∈ {1.0, 0.5, 0.2}. Metrics: science return, **mission failure rate**, energy used.
3. **Experiment sweeps and plots.** Vary sensor noise (`sensor_half_eff_dist`) and hazard probability; plot science-vs-safety tradeoff curves.
4. **Optional.** Faster POMCP: reuse the search tree between steps and speed up the simulator.
5. **Status update** (1 page) and **final report** (4–6 pages, own words, with a group contributions section). Cite SARSOP, POMCP and RockSample.

**Suggested split.** Agree first on the extended model's interface (new state fields, config fields, metrics). Then work in parallel:

| | Person A: model | Person B: planners and experiments |
|---|---|---|
| Code | `env.py`, `belief.py` (battery, hazards, scan action) | `planners.py`, sweep scripts, plots |
| Report | Problem and POMDP formulation | Method, experiments, results figures |

Use one branch per person and merge by pull request, reviewing each other's PRs so both of us can explain all the code.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **POMDP** | MDP where the state is hidden and the agent gets noisy observations |
| **Belief** | Probability distribution over hidden states; here, P(good) for each rock |
| **Discounted return** | Σ γᵗ rₜ, the total reward with future rewards shrunk by γ per step |
| **Generative model** | A simulator that returns (next state, observation, reward) given (state, action) |
| **Alpha vector** | A linear function of the belief representing one conditional plan's value; the POMDP value function is the max over them |
| **Point-based solver** | Approximates the value function by backing up only at sampled beliefs |
| **SARSOP** | Point-based offline solver that samples beliefs reachable under near-optimal policies, guided by upper and lower bounds |
| **Rollout** | Evaluate an action by simulating the future under a base policy |
| **Common random numbers** | Evaluating alternatives on the same random samples, so comparisons aren't swamped by noise |
| **MCTS / POMCP** | Tree search that grows the tree where simulations look promising; POMCP is the POMDP version |
| **UCB1** | Action-selection rule: average value + exploration bonus √(ln N / n) |
| **CVaR_α** | Conditional value at risk: the mean of the worst α-fraction of outcomes; a risk-averse score |
| **Paired comparison** | Comparing two policies on the same random missions and analyzing the per-mission differences |

## References

- T. Smith and R. Simmons. Heuristic Search Value Iteration for POMDPs. UAI, 2004. (RockSample)
- H. Kurniawati, D. Hsu, and W. S. Lee. SARSOP: Efficient Point-Based POMDP Planning by Approximating Optimally Reachable Belief Spaces. RSS, 2008.
- D. Silver and J. Veness. Monte-Carlo Planning in Large POMDPs. NeurIPS, 2010. (POMCP)
- M. J. Kochenderfer, T. A. Wheeler, and K. H. Wray. *Algorithms for Decision Making*. MIT Press, 2022.
