# Active Inference Agent Design

## Proposed Agent

Design a viability-seeking forager that keeps its energy inside viable bounds while exploiting resource patches and recovering after perturbations.

The current codebase already supports a minimal two-factor agent:

- hidden position: `danger`, `home`, `resource`
- hidden energy: `depleted`, `low`, `medium`, `high`
- actions: `move_left`, `stay`, `move_right`
- perception: minimize variational free energy `F`
- action: minimize expected free energy `G = risk + ambiguity`

This document keeps that structure and recommends one stronger next-step design.

## Recommended Architecture

Use three hidden-state factors:

1. `position`
   Values: `danger`, `home`, `resource`
2. `energy`
   Values: `depleted`, `low`, `medium`, `high`
3. `resource_context`
   Values: `scarce`, `stable`, `abundant`

The first two factors match the existing implementation. The third factor lets the agent infer whether the world is currently generous or hostile, which matters after perturbations or repeated low-yield visits to the resource location.

## Observations

Use three observation modalities:

1. `position_obs`
   Direct but slightly noisy observation of location.
2. `energy_obs`
   Noisy interoceptive observation of energy.
3. `cue_obs`
   A coarse exteroceptive cue about the current resource regime.

For the current codebase, the first two modalities already exist. `cue_obs` is the cleanest extension because it adds epistemic value without changing the Markov blanket structure.

## Generative Model

### `A`: likelihood

- `A[position_obs]` should remain near-deterministic given hidden position.
- `A[energy_obs]` should remain moderately noisy given hidden energy.
- `A[cue_obs]` should depend mainly on `resource_context`, with enough noise that ambiguity matters.

Operationally:

- when `resource_context=abundant`, the cue should weakly predict that moving to `resource` will replenish energy well
- when `resource_context=scarce`, the cue should weakly predict poor replenishment and make `home` safer

### `B`: transitions

- `B[position]` remains action-controlled and deterministic
- `B[energy]` depends on both position and context indirectly through the world dynamics
- `B[resource_context]` is uncontrollable and slowly changing

Recommended assumptions:

- `move_right` shifts toward `resource`
- `move_left` shifts toward `danger`
- `stay` preserves position
- energy drops in `danger`, decays slowly at `home`, and tends to recover at `resource`
- context usually persists, but can drift between `scarce`, `stable`, and `abundant`

### `C`: preferences

Keep `C` defined over observations, not hidden states.

Recommended preference ranking:

- strongest preference: `high` energy observations
- acceptable: `medium` energy at `home`
- weakly positive: `resource` position observations
- strongly aversive: `depleted` energy and `danger` observations

This preserves the repo's current interpretation where policy evaluation compares predicted observations against softmaxed observation preferences.

### `D`: priors

Recommended initial prior:

- start near `home`
- start around `medium` energy
- assume `resource_context=stable`

This mirrors the current initialization and avoids overcommitting the agent to optimism or threat at time zero.

## Policy Design

Use a policy horizon of `2` or `3`.

Reason:

- horizon `1` is enough for reflexive regulation
- horizon `2` or `3` lets the agent trade off short-term safety against delayed recovery
- the context factor only becomes useful when the agent can look ahead more than one step

Policy scoring should remain explicit:

- `risk`: mismatch between predicted observations and preferred observations
- `ambiguity`: expected uncertainty in observations under the likelihood model

Interpretation:

- high-energy, low-danger futures reduce risk
- information-seeking actions reduce ambiguity when cue observations can resolve context uncertainty

## Inference And Control Loop

At each step:

1. build the empirical prior from the previous posterior and previous action
2. infer hidden states from the latest observations by minimizing `F`
3. roll out each candidate policy through `B`
4. predict observations through `A`
5. compute `G = risk + ambiguity`
6. form a posterior over policies
7. execute the first action of the best policy
8. update `A` and the learnable parts of `B`

This exactly matches the existing code structure in:

- `/Users/momo/Documents/vibe/project/agent/inference.py`
- `/Users/momo/Documents/vibe/project/agent/policy.py`

## Learning Strategy

Learn selectively.

- keep `B[position]` fixed because movement is known and deterministic
- allow `A[cue_obs]` to learn slowly
- allow `B[resource_context]` to learn slowly
- optionally allow `B[energy]` to adapt if the world is perturbed often

This avoids degrading a stable control model while still letting the agent adapt to changes in environmental productivity.

## Behavioral Target

The designed agent should show these behaviors:

- move to `resource` when energy is falling and cues suggest replenishment
- remain at `home` when uncertainty is high and the expected value of exploration is poor
- avoid `danger` except when model uncertainty makes brief sampling informative enough to justify it
- recover after perturbations by shifting toward safer or more replenishing policies

## Minimal Implementation Plan

Phase 1: current-model tuning

- keep two hidden factors and two observation modalities
- raise `policy_len` default from `1` to `2` or `3`
- keep explicit `risk + ambiguity`
- tune `C` so low energy is more costly than missing the resource briefly

Phase 2: full agent

- extend `world.py` with a latent `resource_context`
- extend `blanket.py` with `cue_obs`
- add the third factor and cue modality in `generative_model.py`
- keep `inference.py` structure unchanged except for larger object arrays
- reuse the same policy evaluation logic in `policy.py`

## Recommended Default Agent

If implementing only one version, implement this one:

- factors: `position`, `energy`, `resource_context`
- modalities: `position_obs`, `energy_obs`, `cue_obs`
- actions: `move_left`, `stay`, `move_right`
- policy horizon: `3`
- control objective: minimize `F` for inference and `G = risk + ambiguity` for action
- learning: fixed movement model, adaptive cue and context dynamics

This design stays consistent with the current repository while making the agent materially more capable than the existing reflexive survival agent.
