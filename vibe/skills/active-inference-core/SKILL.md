---
name: active-inference-core
description: Build, review, or explain active inference agents that perform perception by minimizing variational free energy and choose actions by minimizing expected free energy. Use when Codex needs to define or debug A/B/C/D models, map observations to hidden-state beliefs, compare candidate actions or policies with explicit risk and ambiguity terms, explain Markov blanket or viability concepts in active-inference code, or implement an end-to-end active-inference loop in code or pseudocode.
---

# Active Inference Core

Keep active inference implementations conceptually consistent. Define the generative model clearly, infer hidden states from observations by minimizing variational free energy, and select actions by minimizing expected free energy.

## Work in a Fixed Order

Use the same inspection order before editing theory-heavy code.

1. Check whether the runtime is available with `python3 project/main.py --help` or a short simulation run.
2. Inspect `project/agent/generative_model.py` to confirm the intended `A`, `B`, `C`, `D`, policy horizon, and `pymdp.Agent` flags.
3. Trace one complete loop through `project/main.py`, `project/agent/inference.py`, and `project/agent/policy.py`.
4. Verify that the world and blanket mapping in `project/env/world.py` and `project/env/blanket.py` match the mathematical story.
5. Use `project/analysis/metrics.py` and `project/analysis/ness.py` to judge whether a change improved viability or recovery rather than only changing internal beliefs.

If the runtime fails because `pymdp` is missing, treat that as an environment issue first. The code already documents the expected install command: `python3 -m pip install inferactively-pymdp`.

## Define the Generative Model

Identify the observation modalities, hidden-state factors, control factors, and time horizon before changing code.

- Encode the likelihood model `A` so it maps hidden states to predicted observations.
- Encode the transition model `B` so it maps current hidden states and actions to next hidden states.
- Encode the preference model `C` over observations, not hidden states.
- Encode the prior `D` over initial hidden states.
- Keep array shapes explicit for each modality and factor. Most implementation mistakes come from shape drift or mixing observation labels with state labels.
- Distinguish controllable from uncontrollable factors. In this repository, position is action-controlled and energy evolves as an uncontrollable factor.

For this repository, the concrete model is:

- Two hidden-state factors: position and energy.
- Two observation modalities: position observations and energy observations.
- One control factor: movement along position with actions `move_left`, `stay`, `move_right`.
- A short policy horizon set by `policy_len`.

## Respect the Repository Invariants

Keep these implementation-specific choices intact unless the task is explicitly to change the formulation.

- `project/env/world.py` exposes direct observations. Do not assume hidden noisy sensors just because `A` is probabilistic.
- `project/agent/generative_model.py` sets `use_utility=False` and `use_states_info_gain=False`. The policy code therefore computes explicit `risk + ambiguity` manually instead of relying on the default library EFE helpers.
- `C` is stored as log-preference scores and converted to probabilities with `softmax` inside `project/agent/policy.py`. Do not compare predicted outcomes directly against raw `C` values.
- The empirical prior in `project/agent/inference.py` depends on the cached previous action in `agent.action`. If that action disappears or changes shape, the prior-prediction step becomes wrong.
- Viability in `project/env/viability.py` depends only on energy staying above the boundary. Position matters instrumentally because it changes future energy, not because it is itself a survival criterion.

## Infer Hidden States

Treat perception as approximate Bayesian inference over hidden states.

- Update posterior beliefs `q(s)` so they best explain the current observation under `A` and prior predictive beliefs from `B`.
- Minimize variational free energy during belief updating, whether the implementation does this explicitly or through a library call.
- Preserve the order: observe, infer posterior states, evaluate policies, then act.
- When learning from transitions, align the next observation with the previous posterior state estimate before updating `A` or `B`.
- Preserve the empirical prior. If the implementation predicts a prior from the last action through `B`, do not silently replace it with `D` except at the initial step.

Read [references/core-math.md](references/core-math.md) when you need the core objective or notation.

## Evaluate Candidate Actions

Evaluate each candidate action or policy by minimizing expected free energy `G`.

For each candidate action:

1. Predict future hidden states with `B`.
2. Predict future observations with `A`.
3. Compare predicted outcomes against `C`.
4. Compute expected free energy with explicit risk and ambiguity terms.
5. Select `argmin_a G(a)` unless the design intentionally samples stochastically from a posterior induced by lower `G`.

Do not treat action selection as a separate heuristic. It must be downstream of the expected free energy calculation.

When an implementation bypasses a library helper and computes `G` manually, keep the decomposition inspectable:

- Risk should compare predicted outcomes against preference probabilities derived from `C`.
- Ambiguity should come from the expected conditional entropy of the likelihood model `A`.
- Policy posterior and chosen action should both be explainable from the same `G` values.

## Inspect Risk and Ambiguity

Keep both terms visible in explanations, reviews, and implementations.

- Risk penalizes predicted outcomes that conflict with preferences `C`.
- Ambiguity penalizes actions that lead to observations that are hard to interpret under `A`.
- If an implementation claims to minimize expected free energy but never represents ambiguity, flag it.
- If preferences are encoded over hidden states rather than observations, call that out unless the formulation explicitly justifies the change.
- If a codebase disables library-level utility or epistemic-value terms and replaces them with a custom `risk + ambiguity` calculation, explain that deviation instead of calling it a standard built-in EFE path.

## Review the Implementation

When reviewing code or pseudocode, check these failure modes first.

- Check that `A`, `B`, `C`, and `D` are all present and attached to the correct variables.
- Check that posterior state inference happens before action selection.
- Check that policy evaluation returns or implies expected free energy values.
- Check that the selected action is consistent with minimizing `G`.
- Check that learning updates use correctly aligned observations and previous posteriors.
- Check that comments and documentation do not collapse variational free energy and expected free energy into the same quantity.
- Check that each modality/factor uses the intended axis order before patching tensor code.
- Check that the previous action is still available when constructing the empirical prior for the next inference step.
- Check that learning updates are evaluated against downstream viability or recovery metrics rather than only local reductions in free energy.

## Map the Concepts to This Repository

Use this mapping when working in the current workspace.

- `project/agent/generative_model.py` defines `A`, `B`, `C`, `D`, Dirichlet concentration parameters, and the `pymdp.Agent` configuration.
- `project/agent/inference.py` performs state inference, computes variational free energy from the posterior and empirical prior, and applies learning updates to `A` and `B`.
- `project/agent/policy.py` computes risk, ambiguity, expected free energy, the policy posterior, and the selected action.
- `project/env/blanket.py` holds the sensory, internal, and active state dataclasses plus the Markov blanket transformations.
- `project/env/world.py` defines the external dynamics, labels, perturbation logic, and the survival task structure.
- `project/env/viability.py` defines the viability boundary used to decide whether the system survives.
- `project/analysis/metrics.py` and `project/analysis/ness.py` summarize survival, occupancy, and post-perturbation recovery.
- `project/main.py` shows the end-to-end order: observe, infer, evaluate policies, select action, transition, learn, then analyze outcomes.

When debugging behavior, inspect these files in this order:

1. `project/main.py` for loop order and intervention points.
2. `project/agent/generative_model.py` for model shapes and preferences.
3. `project/agent/inference.py` for empirical prior construction and free-energy reporting.
4. `project/agent/policy.py` for policy rollout, risk, ambiguity, and action selection.
5. `project/env/world.py` and `project/env/viability.py` for whether the environment dynamics actually support the claimed theory.

Use `python3 project/main.py --help` or run the simulation before large refactors so theory changes are checked against the actual loop and reported metrics.

Read [references/core-math.md](references/core-math.md) before rewriting the agent logic or explaining the theory in more formal terms.
