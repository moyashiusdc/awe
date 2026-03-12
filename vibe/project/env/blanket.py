from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from project.env.world import ACTION_LABELS, ExternalState, SurvivalWorld


@dataclass(frozen=True)
class SensoryState:
    """Sensory states sit on the Markov Blanket and mediate observation."""

    position_obs: int
    tile_obs: int
    energy_obs: int

    def as_observation(self) -> list[int]:
        return [self.position_obs, self.tile_obs, self.energy_obs]


@dataclass
class InternalState:
    """Internal states contain posterior beliefs and policy evaluations."""

    posterior_states: np.ndarray
    empirical_prior: np.ndarray
    variational_free_energy: float
    policy_posterior: np.ndarray | None = None
    expected_free_energy: np.ndarray | None = None
    risk: np.ndarray | None = None
    ambiguity: np.ndarray | None = None


@dataclass(frozen=True)
class ActiveState:
    """Active states sit on the Markov Blanket and mediate intervention."""

    action_index: int
    action_label: str


class MarkovBlanket:
    """
    Explicit blanket implementation.

    Dependencies are constrained to:
    external -> sensory
    sensory -> internal
    internal -> active
    active -> external
    """

    def external_to_sensory(
        self,
        world: SurvivalWorld,
        external_state: ExternalState,
    ) -> SensoryState:
        position_obs, tile_obs, energy_obs = world.observe(external_state)
        return SensoryState(
            position_obs=position_obs,
            tile_obs=tile_obs,
            energy_obs=energy_obs,
        )

    def sensory_to_internal(
        self,
        posterior_states: np.ndarray,
        empirical_prior: np.ndarray,
        variational_free_energy: float,
        policy_posterior: np.ndarray | None = None,
        expected_free_energy: np.ndarray | None = None,
        risk: np.ndarray | None = None,
        ambiguity: np.ndarray | None = None,
    ) -> InternalState:
        return InternalState(
            posterior_states=posterior_states,
            empirical_prior=empirical_prior,
            variational_free_energy=variational_free_energy,
            policy_posterior=policy_posterior,
            expected_free_energy=expected_free_energy,
            risk=risk,
            ambiguity=ambiguity,
        )

    def internal_to_active(self, action_payload: Any) -> ActiveState:
        action_array = np.asarray(action_payload, dtype=float).reshape(-1)
        action_index = int(action_array[0])
        return ActiveState(
            action_index=action_index,
            action_label=ACTION_LABELS[action_index],
        )

    def active_to_external(
        self,
        world: SurvivalWorld,
        external_state: ExternalState,
        active_state: ActiveState,
    ) -> ExternalState:
        return world.step(external_state, active_state.action_index)
