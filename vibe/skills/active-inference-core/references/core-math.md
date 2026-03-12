# Active Inference Math

Use this reference when you need the objective functions or need to check whether a code path matches the theory.

## Core Objects

- `A`: Likelihood model mapping hidden states to observations.
- `B`: Transition model mapping current hidden states and actions to next hidden states.
- `C`: Preferences over observations.
- `D`: Prior beliefs over initial hidden states.
- `q(s)`: Approximate posterior over hidden states.

## Variational Free Energy

Use variational free energy for perception and belief updating.

`F[q] = E_q[ln q(s) - ln p(o, s)]`

Minimize `F` with respect to `q(s)` so posterior beliefs explain the current observation `o` under the generative model.

## Expected Free Energy

Use expected free energy for action or policy selection.

For a candidate action or policy:

`G = risk + ambiguity`

Keep the exact decomposition explicit even if the implementation uses a library helper or a different but equivalent notation.

- Risk: Penalize predicted outcomes that are inconsistent with preferences `C`.
- Ambiguity: Penalize actions that are expected to produce observations with high uncertainty under the likelihood model `A`.

Different texts expand these terms differently. Preserve the operational meaning:

- Predict hidden states with `B`.
- Predict observations with `A`.
- Score preferences with `C`.
- Prefer the action with the lowest `G`.

## Review Heuristics

- If a model updates beliefs without conditioning on observations, it is not doing the perception step correctly.
- If an action rule ignores ambiguity entirely, the expected free energy story is incomplete.
- If preferences are stored over hidden states while the write-up claims a standard `C` over observations, either fix the code or explain the deviation.
