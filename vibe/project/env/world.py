from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


GridPosition = tuple[int, int]

POSITION_LABELS: Tuple[str, ...] = ("danger", "home", "resource")
ENERGY_LABELS: Tuple[str, ...] = ("depleted", "low", "medium", "high")

ACTION_LABELS: Tuple[str, ...] = (
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "stay",
)
GRID_ACTION_DELTAS: tuple[GridPosition, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (0, 0),
)

DEFAULT_FOOD_POSITIONS: tuple[GridPosition, ...] = ((2, 2), (2, 8), (8, 5))
DEFAULT_HAZARD_POSITIONS: tuple[GridPosition, ...] = ((5, 2), (7, 8))


@dataclass(frozen=True)
class ExternalState:
    """True external world state behind the Markov Blanket."""

    agent_row: int
    agent_col: int
    food_positions: tuple[GridPosition, ...]
    hazard_positions: tuple[GridPosition, ...]
    energy: int
    step_count: int = 0
    reset_count: int = 0
    was_reset: bool = False

    @property
    def agent_position(self) -> GridPosition:
        return (self.agent_row, self.agent_col)

    @property
    def position(self) -> int:
        """
        Compatibility projection for the current toy agent.

        0 = danger tile, 1 = ordinary safe tile, 2 = food tile.
        """

        if self.agent_position in self.hazard_positions:
            return 0
        if self.agent_position in self.food_positions:
            return 2
        return 1


@dataclass(frozen=True)
class WorldConfig:
    """Parameters for the 11x11 survival gridworld."""

    grid_size: int = 11
    start_position: GridPosition = (5, 5)
    start_energy: int = 7
    max_energy: int = 10
    step_cost: int = 1
    food_reward: int = 4
    hazard_penalty: int = 5
    num_food_sources: int = 3
    num_hazards: int = 2
    local_view_size: int = 5
    food_respawn_rate: float = 0.15
    sensory_noise: float = 0.0
    random_seed: int = 7
    food_positions: tuple[GridPosition, ...] | None = None
    hazard_positions: tuple[GridPosition, ...] | None = None


class SurvivalWorld:
    """11x11 survival world with food, hazards, energy decay, and reset dynamics."""

    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()
        self.grid_size = self.config.grid_size
        self.num_positions = len(POSITION_LABELS)
        self.num_energy_levels = len(ENERGY_LABELS)
        self.rng = np.random.default_rng(self.config.random_seed)
        self.base_sensory_noise = float(np.clip(self.config.sensory_noise, 0.0, 1.0))
        self.sensory_noise_boost = 0.0
        self.sensory_noise_boost_steps_remaining = 0
        self.food_positions = self._resolve_food_positions(self.config.food_positions)
        self.hazard_positions = self._resolve_hazard_positions(self.config.hazard_positions)
        self._validate_layout()

    def _preferred_positions(
        self,
        anchors: tuple[GridPosition, ...],
        count: int,
        excluded: set[GridPosition],
    ) -> tuple[GridPosition, ...]:
        selected = [position for position in anchors if position not in excluded]
        if len(selected) >= count:
            return tuple(selected[:count])

        candidates = [
            (row, col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)
            if (row, col) not in excluded and (row, col) not in selected
        ]
        self.rng.shuffle(candidates)
        selected.extend(candidates[: max(0, count - len(selected))])
        return tuple(selected[:count])

    def _resolve_food_positions(
        self,
        food_positions: tuple[GridPosition, ...] | None,
    ) -> tuple[GridPosition, ...]:
        if food_positions is not None:
            return tuple(food_positions)
        excluded = {self.config.start_position}
        return self._preferred_positions(
            anchors=DEFAULT_FOOD_POSITIONS,
            count=self.config.num_food_sources,
            excluded=excluded,
        )

    def _resolve_hazard_positions(
        self,
        hazard_positions: tuple[GridPosition, ...] | None,
    ) -> tuple[GridPosition, ...]:
        if hazard_positions is not None:
            return tuple(hazard_positions)
        excluded = {self.config.start_position} | set(self.food_positions)
        return self._preferred_positions(
            anchors=DEFAULT_HAZARD_POSITIONS,
            count=self.config.num_hazards,
            excluded=excluded,
        )

    def _validate_layout(self) -> None:
        self._validate_positions(
            self.food_positions,
            expected_count=self.config.num_food_sources,
        )
        self._validate_positions(
            self.hazard_positions,
            expected_count=self.config.num_hazards,
        )
        if set(self.food_positions) & set(self.hazard_positions):
            raise ValueError("Food and hazard positions must not overlap.")
        if self.config.start_position in self.food_positions or self.config.start_position in self.hazard_positions:
            raise ValueError("The starting position must be an empty tile.")

    def _validate_positions(
        self,
        positions: Sequence[GridPosition],
        expected_count: int,
    ) -> None:
        if len(positions) != expected_count:
            raise ValueError(f"Expected {expected_count} positions, got {len(positions)}.")
        if len(set(positions)) != len(positions):
            raise ValueError("World positions must be unique.")
        for row, col in positions:
            if not self.in_bounds((row, col)):
                raise ValueError(f"Out-of-bounds world position {(row, col)}.")

    def in_bounds(self, position: GridPosition) -> bool:
        row, col = position
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def clamp_position(self, position: GridPosition) -> GridPosition:
        row, col = position
        return (
            int(np.clip(row, 0, self.grid_size - 1)),
            int(np.clip(col, 0, self.grid_size - 1)),
        )

    def empty_positions(self, state: ExternalState) -> list[GridPosition]:
        occupied = set(state.food_positions) | set(state.hazard_positions) | {state.agent_position}
        return [
            (row, col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)
            if (row, col) not in occupied
        ]

    def effective_sensory_noise(self) -> float:
        return float(np.clip(self.base_sensory_noise + self.sensory_noise_boost, 0.0, 1.0))

    def increase_sensory_noise(self, amount: float, duration_steps: int) -> None:
        self.sensory_noise_boost = float(np.clip(self.sensory_noise_boost + amount, 0.0, 1.0))
        self.sensory_noise_boost_steps_remaining = max(
            self.sensory_noise_boost_steps_remaining,
            int(duration_steps),
        )

    def _advance_noise_decay(self) -> None:
        if self.sensory_noise_boost_steps_remaining <= 0:
            return
        self.sensory_noise_boost_steps_remaining -= 1
        if self.sensory_noise_boost_steps_remaining == 0:
            self.sensory_noise_boost = 0.0

    def initial_state(self) -> ExternalState:
        return ExternalState(
            agent_row=self.config.start_position[0],
            agent_col=self.config.start_position[1],
            food_positions=self.food_positions,
            hazard_positions=self.hazard_positions,
            energy=self.config.start_energy,
        )

    def reset_state(
        self,
        reset_count: int,
        reference_state: ExternalState | None = None,
    ) -> ExternalState:
        state = self.initial_state()
        food_positions = (
            state.food_positions if reference_state is None else reference_state.food_positions
        )
        hazard_positions = (
            state.hazard_positions if reference_state is None else reference_state.hazard_positions
        )
        return ExternalState(
            agent_row=state.agent_row,
            agent_col=state.agent_col,
            food_positions=food_positions,
            hazard_positions=hazard_positions,
            energy=state.energy,
            step_count=0,
            reset_count=reset_count,
            was_reset=True,
        )

    def is_food(self, position: GridPosition, state: ExternalState | None = None) -> bool:
        food_positions = self.food_positions if state is None else state.food_positions
        return position in food_positions

    def is_hazard(self, position: GridPosition, state: ExternalState | None = None) -> bool:
        hazard_positions = self.hazard_positions if state is None else state.hazard_positions
        return position in hazard_positions

    def energy_level_index(self, energy: int) -> int:
        if energy <= 0:
            return 0
        if energy <= 3:
            return 1
        if energy <= 7:
            return 2
        return 3

    def position_to_index(self, position: GridPosition) -> int:
        row, col = position
        return (row * self.grid_size) + col

    def index_to_position(self, position_index: int) -> GridPosition:
        bounded_index = int(np.clip(position_index, 0, (self.grid_size * self.grid_size) - 1))
        return divmod(bounded_index, self.grid_size)

    def tile_class_index(self, position: GridPosition, state: ExternalState | None = None) -> int:
        if self.is_hazard(position, state):
            return 0
        if self.is_food(position, state):
            return 2
        return 1

    def _sample_noisy_index(self, value: int, cardinality: int) -> int:
        noise = self.effective_sensory_noise()
        if cardinality <= 1 or noise <= 0.0:
            return value
        if float(self.rng.random()) >= noise:
            return value
        alternatives = [index for index in range(cardinality) if index != value]
        return int(alternatives[int(self.rng.integers(len(alternatives)))])

    def observe(self, state: ExternalState) -> tuple[int, int, int]:
        """
        Observation interface for the active-inference model.

        Modalities:
        - exact grid position index
        - tile class (danger / home / resource)
        - energy level
        """

        position_obs = self._sample_noisy_index(
            self.position_to_index(state.agent_position),
            self.grid_size * self.grid_size,
        )
        tile_obs = self._sample_noisy_index(
            self.tile_class_index(state.agent_position, state),
            self.num_positions,
        )
        energy_obs = self._sample_noisy_index(
            self.energy_level_index(state.energy),
            self.num_energy_levels,
        )
        return position_obs, tile_obs, energy_obs

    def observe_partial(self, state: ExternalState):
        from project.env.observations import build_observation

        return build_observation(
            state=state,
            grid_size=self.grid_size,
            noise=self.effective_sensory_noise(),
            generator=self.rng,
        )

    def _action_delta(self, action_index: int) -> GridPosition:
        bounded_action = int(np.clip(action_index, 0, len(ACTION_LABELS) - 1))
        return GRID_ACTION_DELTAS[bounded_action]

    def _maybe_respawn_food(self, state: ExternalState) -> tuple[GridPosition, ...]:
        food_positions = list(state.food_positions)
        while len(food_positions) < self.config.num_food_sources:
            if float(self.rng.random()) > self.config.food_respawn_rate:
                break
            occupied = set(food_positions) | set(state.hazard_positions) | {state.agent_position}
            candidates = [
                (row, col)
                for row in range(self.grid_size)
                for col in range(self.grid_size)
                if (row, col) not in occupied
            ]
            if not candidates:
                break
            respawn_position = candidates[int(self.rng.integers(len(candidates)))]
            food_positions.append(respawn_position)
        return tuple(sorted(food_positions))

    def step(self, state: ExternalState, action_index: int) -> ExternalState:
        """
        Apply a 2D action to the grid world.
        """

        delta_row, delta_col = self._action_delta(action_index)
        next_position = self.clamp_position(
            (state.agent_row + delta_row, state.agent_col + delta_col)
        )

        energy = state.energy - self.config.step_cost
        next_food_positions = tuple(state.food_positions)

        if next_position in state.food_positions:
            energy += self.config.food_reward
            next_food_positions = tuple(
                food_position for food_position in state.food_positions if food_position != next_position
            )
        elif next_position in state.hazard_positions:
            energy -= self.config.hazard_penalty

        next_energy = int(np.clip(energy, 0, self.config.max_energy))
        if next_energy <= 0:
            self._advance_noise_decay()
            return self.reset_state(
                reset_count=state.reset_count + 1,
                reference_state=state,
            )

        next_state = ExternalState(
            agent_row=next_position[0],
            agent_col=next_position[1],
            food_positions=next_food_positions,
            hazard_positions=state.hazard_positions,
            energy=next_energy,
            step_count=state.step_count + 1,
            reset_count=state.reset_count,
            was_reset=False,
        )
        respawned_food_positions = self._maybe_respawn_food(next_state)
        self._advance_noise_decay()
        return ExternalState(
            agent_row=next_state.agent_row,
            agent_col=next_state.agent_col,
            food_positions=respawned_food_positions,
            hazard_positions=next_state.hazard_positions,
            energy=next_state.energy,
            step_count=next_state.step_count,
            reset_count=next_state.reset_count,
            was_reset=False,
        )

    def _forced_position(self, forced_position: int | GridPosition | None) -> GridPosition:
        if forced_position is None:
            return self.config.start_position
        if isinstance(forced_position, tuple):
            return self.clamp_position(forced_position)
        if forced_position == 0 and self.hazard_positions:
            return self.hazard_positions[0]
        if forced_position == 1:
            return self.config.start_position
        if forced_position == 2 and self.food_positions:
            return self.food_positions[0]

        flat_index = int(np.clip(forced_position, 0, (self.grid_size * self.grid_size) - 1))
        return divmod(flat_index, self.grid_size)

    def perturb(
        self,
        state: ExternalState,
        energy_drop: int = 2,
        forced_position: int | GridPosition | None = None,
    ) -> ExternalState:
        """Apply an exogenous perturbation while preserving the world layout."""

        position = self._forced_position(forced_position)
        next_energy = int(np.clip(state.energy - energy_drop, 0, self.config.max_energy))
        if next_energy <= 0:
            return self.reset_state(
                reset_count=state.reset_count + 1,
                reference_state=state,
            )

        return ExternalState(
            agent_row=position[0],
            agent_col=position[1],
            food_positions=state.food_positions,
            hazard_positions=state.hazard_positions,
            energy=next_energy,
            step_count=state.step_count,
            reset_count=state.reset_count,
            was_reset=False,
        )
