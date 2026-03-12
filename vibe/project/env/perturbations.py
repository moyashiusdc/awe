from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from project.env.world import ExternalState, GridPosition, SurvivalWorld


@dataclass(frozen=True)
class PerturbationEvent:
    """Description of an exogenous perturbation applied to the world."""

    kind: str
    details: dict[str, Any]


def _rng(generator: np.random.Generator | None = None) -> np.random.Generator:
    return generator if generator is not None else np.random.default_rng()


def _empty_positions(world: SurvivalWorld, state: ExternalState) -> list[GridPosition]:
    return world.empty_positions(state)


def sudden_energy_drop(
    world: SurvivalWorld,
    state: ExternalState,
    amount: int,
) -> tuple[ExternalState, PerturbationEvent]:
    next_energy = int(np.clip(state.energy - amount, 0, world.config.max_energy))
    if next_energy <= 0:
        return (
            world.reset_state(
                reset_count=state.reset_count + 1,
                reference_state=state,
            ),
            PerturbationEvent(
                kind="sudden_energy_shock",
                details={"amount": int(amount), "triggered_reset": True},
            ),
        )

    return (
        replace(state, energy=next_energy, was_reset=False),
        PerturbationEvent(
            kind="sudden_energy_shock",
            details={"amount": int(amount), "triggered_reset": False},
        ),
    )


def hazard_spawn(
    world: SurvivalWorld,
    state: ExternalState,
    position: GridPosition | None = None,
    generator: np.random.Generator | None = None,
) -> tuple[ExternalState, PerturbationEvent]:
    available_positions = _empty_positions(world, state)
    if not available_positions:
        return (
            state,
            PerturbationEvent(kind="spawn_extra_hazard", details={"spawned": False}),
        )

    if position is None:
        spawn_position = available_positions[int(_rng(generator).integers(len(available_positions)))]
    else:
        spawn_position = world.clamp_position(position)
        if spawn_position not in available_positions:
            raise ValueError(f"Cannot spawn a hazard on occupied tile {spawn_position}.")

    next_state = replace(
        state,
        hazard_positions=tuple(sorted(state.hazard_positions + (spawn_position,))),
        was_reset=False,
    )
    return (
        next_state,
        PerturbationEvent(
            kind="spawn_extra_hazard",
            details={"spawned": True, "position": spawn_position},
        ),
    )


def food_removal(
    world: SurvivalWorld,
    state: ExternalState,
    position: GridPosition | None = None,
    generator: np.random.Generator | None = None,
) -> tuple[ExternalState, PerturbationEvent]:
    if not state.food_positions:
        return (
            state,
            PerturbationEvent(kind="remove_one_food_source", details={"removed": False}),
        )

    if position is None:
        removal_index = int(_rng(generator).integers(len(state.food_positions)))
        removal_position = state.food_positions[removal_index]
    else:
        removal_position = world.clamp_position(position)
        if removal_position not in state.food_positions:
            raise ValueError(f"No food source exists at {removal_position}.")

    next_food_positions = tuple(
        food_position for food_position in state.food_positions if food_position != removal_position
    )
    next_state = replace(state, food_positions=next_food_positions, was_reset=False)
    return (
        next_state,
        PerturbationEvent(
            kind="remove_one_food_source",
            details={"removed": True, "position": removal_position},
        ),
    )


def random_displacement(
    world: SurvivalWorld,
    state: ExternalState,
    position: GridPosition | None = None,
    generator: np.random.Generator | None = None,
) -> tuple[ExternalState, PerturbationEvent]:
    available_positions = _empty_positions(world, state)
    if not available_positions:
        return (
            state,
            PerturbationEvent(kind="random_displacement", details={"displaced": False}),
        )

    if position is None:
        target_position = available_positions[int(_rng(generator).integers(len(available_positions)))]
    else:
        target_position = world.clamp_position(position)
        if target_position not in available_positions:
            raise ValueError(f"Cannot displace the agent onto occupied tile {target_position}.")

    next_state = replace(
        state,
        agent_row=target_position[0],
        agent_col=target_position[1],
        was_reset=False,
    )
    return (
        next_state,
        PerturbationEvent(
            kind="random_displacement",
            details={"displaced": True, "position": target_position},
        ),
    )


def temporary_sensory_noise_increase(
    world: SurvivalWorld,
    state: ExternalState,
    amount: float,
    duration_steps: int,
) -> tuple[ExternalState, PerturbationEvent]:
    world.increase_sensory_noise(amount=amount, duration_steps=duration_steps)
    return (
        state,
        PerturbationEvent(
            kind="temporary_sensory_noise_increase",
            details={"amount": float(amount), "duration_steps": int(duration_steps)},
        ),
    )


def apply_perturbation(
    world: SurvivalWorld,
    state: ExternalState,
    kind: str,
    **kwargs,
) -> tuple[ExternalState, PerturbationEvent]:
    if kind in {"sudden_energy_drop", "sudden_energy_shock"}:
        return sudden_energy_drop(world, state, amount=int(kwargs.get("amount", 1)))
    if kind in {"hazard_spawn", "spawn_extra_hazard"}:
        return hazard_spawn(
            world,
            state,
            position=kwargs.get("position"),
            generator=kwargs.get("generator"),
        )
    if kind in {"food_removal", "remove_one_food_source"}:
        return food_removal(
            world,
            state,
            position=kwargs.get("position"),
            generator=kwargs.get("generator"),
        )
    if kind == "random_displacement":
        return random_displacement(
            world,
            state,
            position=kwargs.get("position"),
            generator=kwargs.get("generator"),
        )
    if kind in {"temporary_sensory_noise_increase", "sensory_noise_increase"}:
        return temporary_sensory_noise_increase(
            world,
            state,
            amount=float(kwargs.get("amount", 0.2)),
            duration_steps=int(kwargs.get("duration_steps", 5)),
        )
    raise ValueError(f"Unknown perturbation kind: {kind}")
