from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from project.analysis.metrics import (
    BeliefVisualization,
    SimulationRecord,
    average_energy,
    belief_visualization,
    energy_histogram,
    fraction_time_viable,
    perturbation_counts,
    position_occupancy_heatmap,
    recovery_after_perturbation,
    reset_count,
    reset_frequency,
    state_occupancy_distribution,
    survival_time,
)


@dataclass(frozen=True)
class NESSReport:
    survival_time: int
    average_energy: float
    fraction_time_viable: float
    state_occupancy_distribution: dict[str, float]
    long_run_state_distribution: dict[str, float]
    recovery_time_after_perturbation: int | None
    reset_count: int
    reset_frequency: float
    energy_histogram: np.ndarray
    position_occupancy_heatmap: np.ndarray
    perturbation_counts: dict[str, int]
    belief_visualization: BeliefVisualization | None = None


def analyze_ness(
    records: Sequence[SimulationRecord],
    perturbation_time: int | None = None,
    burn_in: int = 25,
    grid_size: int = 11,
    max_energy: int | None = None,
    normalize_energy_histogram: bool = False,
    normalize_occupancy_heatmap: bool = True,
) -> NESSReport:
    tail_records = records[burn_in:] if len(records) > burn_in else records
    return NESSReport(
        survival_time=survival_time(records),
        average_energy=average_energy(records),
        fraction_time_viable=fraction_time_viable(records),
        state_occupancy_distribution=state_occupancy_distribution(records),
        long_run_state_distribution=state_occupancy_distribution(tail_records),
        recovery_time_after_perturbation=recovery_after_perturbation(
            records,
            perturbation_time,
            recovered_position=2,
            min_recovered_energy=2,
        ),
        reset_count=reset_count(records),
        reset_frequency=reset_frequency(records),
        energy_histogram=energy_histogram(
            records,
            max_energy=max_energy,
            normalize=normalize_energy_histogram,
        ),
        position_occupancy_heatmap=position_occupancy_heatmap(
            records,
            grid_size=grid_size,
            normalize=normalize_occupancy_heatmap,
        ),
        perturbation_counts=perturbation_counts(records),
        belief_visualization=belief_visualization(records),
    )
