from __future__ import annotations

from dataclasses import dataclass

from project.env.world import ExternalState


@dataclass(frozen=True)
class ViabilityBounds:
    """Bounded region the agent must remain in to survive."""

    min_viable_energy: int = 1
    max_viable_energy: int = 3

    def contains_energy(self, energy: int) -> bool:
        return self.min_viable_energy <= energy <= self.max_viable_energy

    def contains_state(self, state: ExternalState) -> bool:
        return self.contains_energy(state.energy)


def viability_failed(state: ExternalState) -> bool:
    return state.energy <= 0
