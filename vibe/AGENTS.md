# AGENTS.md

## Project Overview

This project implements a high-concept survival simulation game based on:

- Karl Friston's Active Inference framework
- the Free Energy Principle
- explicit Markov Blanket structure
- Nonequilibrium Steady State (NESS) / viability maintenance

The game should feel like an artificial life experiment rather than a conventional reward-maximization game.

The core objective is to build an agent that maintains its own viability under partial observation in an 11x11 world.


## Core Design Goals

The implementation must:

1. build an Active Inference agent using an explicit generative model
2. perform perception by minimizing variational free energy
3. perform action selection by minimizing expected free energy
4. explicitly separate external, sensory, internal, and active states
5. model survival as viability maintenance rather than standard reward optimization
6. allow long-horizon simulation and NESS-style analysis
7. remain interpretable, small, and inspectable


## Hard Constraints

Use:

- Python
- numpy
- pymdp

Do not use:

- PyTorch
- TensorFlow
- JAX
- neural networks
- gradient descent
- backpropagation
- reinforcement learning
- policy gradients
- Q-learning
- black-box end-to-end models

This is not a deep learning project.
This is not a reinforcement learning project.
All inference and control should be explicit and interpretable.


## World Concept

The game world is an 11x11 survival gridworld.

The agent is not given full access to the true world state.

It must survive by inferring hidden causes of sensory observations and selecting actions that keep it within viable states over time.

The game should be framed as an artificial organism maintaining itself in a partially observed environment.


## Minimal Game Specification

World size:

- 11x11 grid

Entities:

- 1 agent
- 3 food sources
- 2 hazards

Agent state variables:

- position
- energy

Energy rules:

- energy decreases by 1 every step
- eating food increases energy by 4
- stepping on a hazard decreases energy by 5
- max energy should be bounded, e.g. 10
- if energy reaches 0, viability fails and the episode resets

Suggested initial values:

- initial energy: 7
- max energy: 10

Food behavior:

- food respawns after being consumed
- respawn location should be sampled from valid empty cells

Hazard behavior:

- hazards may move stochastically or follow a simple local movement rule
- hazard dynamics should remain interpretable and simple

The game loop must support long survival runs and repeated resets for statistical analysis.


## Partial Observation

The agent must not observe the full world state.

Use partial and local observations only.

Recommended observation channels:

1. local vision:
   - a 5x5 local view centered on the agent

2. food smell:
   - none / weak / strong
   - intensity based on distance to nearest food

3. danger signal:
   - none / weak / strong
   - intensity based on distance to nearest hazard

4. internal energy signal:
   - okay / low / critical

The agent should only update beliefs from these observations.
The environment should maintain true state privately.


## Markov Blanket Requirements

The implementation must explicitly represent the following four state classes:

- external_state
- sensory_state
- internal_state
- active_state

Interpretation:

- external_state = true world state (food, hazards, map, true energy dynamics)
- sensory_state = local view, smell, danger signal, energy signal
- internal_state = beliefs about food, hazard proximity, viability, hidden state factors
- active_state = selected actions / motor output

The dependency structure must be explicit:

external -> sensory  
sensory -> internal  
internal -> active  
active -> external

Direct external <-> internal coupling is not allowed.

The Markov Blanket consists of:

- sensory_state
- active_state

The code should make this split concrete rather than treating it as a loose metaphor.


## Active Inference Requirements

The agent must be implemented as an Active Inference agent.

It must:

- infer hidden states from observations
- maintain beliefs over hidden state factors
- evaluate candidate actions or policies using expected free energy
- choose actions that minimize expected free energy
- maintain preferred and viable states over time

Do not replace Active Inference with heuristic action rules unless they are clearly marked as temporary scaffolding.


## Free Energy Principle

Perception must be implemented as variational free energy minimization.

Belief updates should infer hidden states that best explain current observations.

Variational free energy should be treated as the objective for perception / inference.

The implementation does not need overly abstract symbolic derivations, but it must preserve the correct role of free energy in the inference loop.


## Expected Free Energy

Action selection must minimize expected free energy.

For each candidate action or short-horizon policy:

1. predict future hidden states using the transition model (B)
2. predict expected observations using the likelihood model (A)
3. evaluate predicted outcomes under preferences (C)
4. estimate expected free energy
5. select the action / policy with minimal expected free energy

Expected free energy should explicitly include:

- risk: divergence from preferred outcomes
- ambiguity: uncertainty in predicted observations

The implementation should keep this interpretable.
If simplifications are made, they must preserve the risk + ambiguity structure.


## Generative Model Structure

Use an explicit discrete generative model with:

- A : likelihood mapping, observations given hidden states
- B : state transition model, next hidden states given current states and action
- C : preferences over observations / outcomes
- D : priors over initial hidden states

All matrices must be documented with:

- dimensions
- meaning of each axis
- interpretation of factors and modalities

Do not leave opaque arrays unexplained.

Keep the hidden-state space small enough to inspect manually.

Prefer factored discrete hidden states over large uninterpretable tables.


## Hidden State Factors

At minimum, hidden state factors should capture:

- agent position
- energy level

Optional but recommended hidden factors:

- nearest food direction
- hazard proximity
- local viability regime

These factors should remain explicit and interpretable.


## Observations and Preferences

Observations should represent partial sensory evidence rather than full state access.

Preferences should reflect viability-oriented outcomes, for example:

preferred:

- adequate energy
- low danger
- food availability or food-relevant cues
- remaining in viable states

non-preferred:

- critical energy
- strong danger
- death / reset states

The project should emphasize viability maintenance over conventional score maximization.


## Viability and NESS

The world must support viability analysis and NESS-style measurement.

Viability means the agent remains within a bounded region of acceptable states over time.

NESS-style behavior should be evaluated empirically through long simulations, not claimed abstractly.

The code should support measuring:

- survival time
- reset frequency
- occupancy distribution over positions
- occupancy distribution over energy levels
- fraction of time spent in viable states
- recovery after perturbation

The agent does not need to converge to a literal fixed point.
It should exhibit bounded long-run dynamics with stable statistical structure.


## Attractor / Basin Interpretation

The game should support an attractor-like interpretation of survival dynamics.

Define viable / good states such as:

- energy is not critical
- danger is low or manageable
- food is reachable
- the agent is not trapped in immediate collapse

The agent should tend to return toward these viable regions after perturbation.

This attractor interpretation should emerge from the dynamics and policy selection, not from ad hoc scripted teleportation.


## Perturbation Support

Include support for perturbation experiments.

Possible perturbations:

- sudden energy drop
- temporary sensory noise increase
- additional hazard spawn
- removal of a food source
- random displacement of the agent

The analysis code should measure whether and how quickly the agent returns to a viable regime after perturbation.

This is important for demonstrating resilience and NESS-like behavior.


## Project Structure

Use this project layout:

project/
  AGENTS.md
  README.md
  main.py
  config.py

  env/
    world.py
    blanket.py
    observations.py
    viability.py
    perturbations.py
    renderer.py

  agent/
    generative_model.py
    inference.py
    policy.py
    preferences.py
    state_factors.py

  analysis/
    metrics.py
    ness.py
    plots.py

  tests/
    test_blanket.py
    test_inference.py
    test_policy.py
    test_viability.py
    test_ness.py


## File Responsibilities

Expected responsibilities:

env/world.py
- true world state
- entity placement
- movement rules
- food respawn
- hazard dynamics
- reset logic

env/blanket.py
- explicit Markov Blanket interfaces
- mapping from external state to sensory state
- mapping from active state to external state transitions

env/observations.py
- local 5x5 view
- smell / danger / energy signal construction

env/viability.py
- define viable states
- detect viability failure
- track reset conditions

env/perturbations.py
- perturbation functions for experiments

env/renderer.py
- simple rendering for grid and overlays
- optional belief / stats display hooks

agent/generative_model.py
- construct A, B, C, D
- document dimensions and semantics

agent/inference.py
- variational free energy style belief update
- hidden-state posterior update

agent/policy.py
- expected free energy evaluation
- risk / ambiguity decomposition
- action selection

agent/preferences.py
- preference definitions used to build C

agent/state_factors.py
- explicit description of hidden-state factors and observation modalities

analysis/metrics.py
- survival time
- reset count
- viable-state fraction
- occupancy summaries

analysis/ness.py
- long-horizon simulation
- empirical steady-state statistics
- perturbation recovery analysis

analysis/plots.py
- energy over time
- occupancy heatmaps
- recovery curves

tests/
- unit tests for blanket split, inference behavior, policy scoring, viability rules, and NESS analysis


## Rendering / Presentation

Keep visuals simple and interpretable.

Recommended display:

- 11x11 grid
- agent, food, hazards
- current energy
- current observations
- inferred food-related belief
- inferred danger-related belief
- selected action
- survival time
- reset count

Optional analysis views:

- energy over time
- occupancy heatmap
- perturbation recovery plots

Do not overinvest in flashy graphics.
Prioritize conceptual clarity and research-demo quality.


## Coding Style

Write small, readable, modular code.

Requirements:

- every file should have a short module docstring
- all major functions should have docstrings
- avoid clever abstractions
- prefer explicit names over compressed code
- add comments for probability tables and matrix shapes
- keep world rules deterministic where possible, stochastic only where useful
- keep implementation inspectable and debuggable

Do not hide core logic behind heavy frameworks.


## Implementation Priorities

Build in this order:

1. basic 11x11 world and reset loop
2. partial observation pipeline
3. explicit Markov Blanket split
4. simple discrete generative model (A, B, C, D)
5. belief update / inference loop
6. expected free energy based action selection
7. viability metrics
8. NESS / long-horizon analysis
9. perturbation analysis
10. simple rendering and plots


## Output Expectations for Codex

When generating code:

1. explain the role of each file first
2. then generate code in small, coherent steps
3. keep code runnable at every step
4. do not invent unnecessary infrastructure
5. preserve the Active Inference / Free Energy interpretation throughout
6. do not silently substitute reinforcement learning or heuristic search for expected free energy minimization

If a simplification is introduced, state clearly:

- what is simplified
- why it is simplified
- how it still maps onto the intended theory


## Forbidden Moves

Do not:

- replace Active Inference with reinforcement learning
- replace expected free energy with reward maximization language
- replace explicit beliefs with hidden neural embeddings
- use black-box policies
- claim Markov Blanket structure unless the split is explicit in code
- claim NESS without long-run simulation evidence
- claim free energy minimization if the code is only heuristic scoring without clear mapping


## First Task

First generate:

1. the full project scaffold
2. the minimal 11x11 world implementation
3. the explicit Markov Blanket module
4. a small discrete generative model
5. a basic inference loop
6. expected free energy based action selection
7. viability tracking
8. a simple long-horizon NESS analysis script
9. minimal tests

Before writing files, explain how each part maps to:

- external states
- sensory states
- internal states
- active states
- free energy based inference
- expected free energy based action
- viability
- NESS