from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from project.env.world import ExternalState, GridPosition


LOCAL_VIEW_SIZE = 5
LOCAL_VIEW_RADIUS = LOCAL_VIEW_SIZE // 2

LOCAL_TILE_LABELS = ("out_of_bounds", "empty", "food", "hazard", "agent")
FOOD_SMELL_LABELS = ("none", "weak", "strong")
DANGER_SIGNAL_LABELS = ("none", "weak", "strong")
ENERGY_SIGNAL_LABELS = ("okay", "low", "critical")

OUT_OF_BOUNDS_TILE = 0
EMPTY_TILE = 1
FOOD_TILE = 2
HAZARD_TILE = 3
AGENT_TILE = 4


@dataclass(frozen=True)
class PartialObservation:
    """Partial sensory state exposed to the agent."""

    local_view: np.ndarray
    food_smell: int
    danger_signal: int
    energy_signal: int


def _manhattan_distance(source: GridPosition, target: GridPosition) -> int:
    return abs(source[0] - target[0]) + abs(source[1] - target[1])


def _nearest_distance(origin: GridPosition, positions: tuple[GridPosition, ...]) -> int | None:
    if not positions:
        return None
    return min(_manhattan_distance(origin, position) for position in positions)


def _sample_noisy_index(
    value: int,
    cardinality: int,
    noise: float,
    generator: np.random.Generator | None,
) -> int:
    if cardinality <= 1 or noise <= 0.0:
        return value
    rng = generator if generator is not None else np.random.default_rng()
    if float(rng.random()) >= noise:
        return value
    alternatives = [index for index in range(cardinality) if index != value]
    return int(alternatives[int(rng.integers(len(alternatives)))])


def classify_food_smell(state: ExternalState) -> int:
    distance = _nearest_distance(state.agent_position, state.food_positions)
    if distance is None or distance > 4:
        return 0
    if distance <= 2:
        return 2
    return 1


def classify_danger_signal(state: ExternalState) -> int:
    distance = _nearest_distance(state.agent_position, state.hazard_positions)
    if distance is None or distance > 4:
        return 0
    if distance <= 2:
        return 2
    return 1


def classify_energy_signal(energy: int) -> int:
    if energy <= 2:
        return 2
    if energy <= 4:
        return 1
    return 0


def build_local_view(
    state: ExternalState,
    grid_size: int,
    radius: int = LOCAL_VIEW_RADIUS,
    noise: float = 0.0,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    size = (radius * 2) + 1
    local_view = np.full((size, size), OUT_OF_BOUNDS_TILE, dtype=int)

    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            world_row = state.agent_row + row_offset
            world_col = state.agent_col + col_offset
            local_row = row_offset + radius
            local_col = col_offset + radius

            if not (0 <= world_row < grid_size and 0 <= world_col < grid_size):
                continue

            position = (world_row, world_col)
            if position == state.agent_position:
                tile_value = AGENT_TILE
            elif position in state.hazard_positions:
                tile_value = HAZARD_TILE
            elif position in state.food_positions:
                tile_value = FOOD_TILE
            else:
                tile_value = EMPTY_TILE

            if tile_value != AGENT_TILE:
                tile_value = _sample_noisy_index(
                    tile_value - 1,
                    len(LOCAL_TILE_LABELS) - 1,
                    noise,
                    generator,
                ) + 1
            local_view[local_row, local_col] = tile_value

    return local_view


def build_observation(
    state: ExternalState,
    grid_size: int,
    noise: float = 0.0,
    generator: np.random.Generator | None = None,
) -> PartialObservation:
    return PartialObservation(
        local_view=build_local_view(
            state=state,
            grid_size=grid_size,
            noise=noise,
            generator=generator,
        ),
        food_smell=_sample_noisy_index(
            classify_food_smell(state),
            len(FOOD_SMELL_LABELS),
            noise,
            generator,
        ),
        danger_signal=_sample_noisy_index(
            classify_danger_signal(state),
            len(DANGER_SIGNAL_LABELS),
            noise,
            generator,
        ),
        energy_signal=_sample_noisy_index(
            classify_energy_signal(state.energy),
            len(ENERGY_SIGNAL_LABELS),
            noise,
            generator,
        ),
    )
