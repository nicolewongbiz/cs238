"""Risk-aware, belief-informed planners (proposal policy 3).

TODO: implement one or more of
  * Rollout / lookahead: for each action, simulate N trajectories from states
    sampled from the belief (`RockBelief.sample_state`) with a rollout policy
    (e.g. GreedyExpectedValuePolicy), pick the best mean discounted return.
  * POMCP (Silver & Veness, 2010): MCTS over action/observation histories with
    a particle belief. Works directly with `RockSampleEnv.step`.
  * Risk awareness: score actions by a risk-adjusted return, e.g. penalize
    P(battery depleted / immobilized) or optimize CVaR instead of the mean.

Each planner should subclass `rover.policies.Policy` so it plugs into
`rover.simulate.evaluate` and `scripts/evaluate.py` unchanged.
"""
