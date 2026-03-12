from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BeliefSnapshot:
    """Visualization-ready copy of the agent's internal-state summary."""

    posterior_states: tuple[np.ndarray, ...]
    variational_free_energy: float
    expected_free_energy: np.ndarray | None = None
    policy_posterior: np.ndarray | None = None
    risk: np.ndarray | None = None
    ambiguity: np.ndarray | None = None

    @classmethod
    def from_internal_state(cls, internal_state) -> BeliefSnapshot:
        posterior_states = tuple(
            np.asarray(factor_posterior, dtype=float)
            for factor_posterior in internal_state.posterior_states
        )
        return cls(
            posterior_states=posterior_states,
            variational_free_energy=float(internal_state.variational_free_energy),
            expected_free_energy=_optional_array(internal_state.expected_free_energy),
            policy_posterior=_optional_array(internal_state.policy_posterior),
            risk=_optional_array(internal_state.risk),
            ambiguity=_optional_array(internal_state.ambiguity),
        )


@dataclass(frozen=True)
class BeliefVisualization:
    """Time-aligned belief traces for plotting or inspection."""

    time: np.ndarray
    posterior_states: tuple[np.ndarray, ...]
    variational_free_energy: np.ndarray
    expected_free_energy: np.ndarray | None = None
    policy_posterior: np.ndarray | None = None
    risk: np.ndarray | None = None
    ambiguity: np.ndarray | None = None


@dataclass(frozen=True)
class SimulationRecord:
    time: int
    position: int
    energy: int
    viable: bool
    action: str | None
    perturbed: bool = False
    row: int | None = None
    col: int | None = None
    reset_count: int = 0
    perturbation_kind: str | None = None
    belief: BeliefSnapshot | None = None
    observation_summary: str | None = None
    food_positions: tuple[tuple[int, int], ...] | None = None
    hazard_positions: tuple[tuple[int, int], ...] | None = None


def _optional_array(values) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _stack_or_object_array(values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.stack(values)
    except ValueError:
        return np.asarray(values, dtype=object)


def survival_time(records: Sequence[SimulationRecord]) -> int:
    for record in records:
        if not record.viable:
            return record.time
    return records[-1].time if records else 0


def average_energy(records: Sequence[SimulationRecord]) -> float:
    if not records:
        return 0.0
    return float(np.mean([record.energy for record in records]))


def state_occupancy_distribution(
    records: Sequence[SimulationRecord],
) -> dict[str, float]:
    if not records:
        return {}

    counts = Counter((record.position, record.energy) for record in records)
    total = float(len(records))
    return {
        f"position={position}, energy={energy}": count / total
        for (position, energy), count in sorted(counts.items())
    }


def fraction_time_viable(records: Sequence[SimulationRecord]) -> float:
    if not records:
        return 0.0
    viable_count = sum(record.viable for record in records)
    return viable_count / float(len(records))


def reset_count(records: Sequence[SimulationRecord]) -> int:
    if not records:
        return 0
    return max(int(record.reset_count) for record in records)


def reset_frequency(records: Sequence[SimulationRecord]) -> float:
    if len(records) <= 1:
        return 0.0
    return reset_count(records) / float(len(records) - 1)


def energy_histogram(
    records: Sequence[SimulationRecord],
    max_energy: int | None = None,
    normalize: bool = False,
) -> np.ndarray:
    if not records:
        if max_energy is None:
            return np.zeros(0, dtype=float if normalize else int)
        return np.zeros(max_energy + 1, dtype=float if normalize else int)

    histogram_max = max(record.energy for record in records) if max_energy is None else max_energy
    histogram = np.zeros(histogram_max + 1, dtype=float)
    for record in records:
        if 0 <= record.energy <= histogram_max:
            histogram[record.energy] += 1.0
    if normalize and histogram.sum() > 0:
        histogram /= histogram.sum()
    if normalize:
        return histogram
    return histogram.astype(int)


def position_occupancy_heatmap(
    records: Sequence[SimulationRecord],
    grid_size: int = 11,
    normalize: bool = True,
) -> np.ndarray:
    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    for record in records:
        if record.row is None or record.col is None:
            continue
        if 0 <= record.row < grid_size and 0 <= record.col < grid_size:
            heatmap[record.row, record.col] += 1.0
    if normalize and heatmap.sum() > 0:
        heatmap /= heatmap.sum()
    return heatmap


def perturbation_counts(records: Sequence[SimulationRecord]) -> dict[str, int]:
    counts = Counter(
        record.perturbation_kind
        for record in records
        if record.perturbation_kind is not None
    )
    return dict(sorted(counts.items()))


def belief_visualization(records: Sequence[SimulationRecord]) -> BeliefVisualization | None:
    snapshots = [
        (record.time, record.belief)
        for record in records
        if record.belief is not None
    ]
    if not snapshots:
        return None

    time = np.asarray([time_index for time_index, _ in snapshots], dtype=int)
    beliefs = [belief for _, belief in snapshots if belief is not None]
    num_factors = len(beliefs[0].posterior_states)
    posterior_states = tuple(
        _stack_or_object_array(
            [np.asarray(belief.posterior_states[factor_index], dtype=float) for belief in beliefs]
        )
        for factor_index in range(num_factors)
    )

    variational_free_energy = np.asarray(
        [belief.variational_free_energy for belief in beliefs],
        dtype=float,
    )

    expected_free_energy = None
    if all(belief.expected_free_energy is not None for belief in beliefs):
        expected_free_energy = _stack_or_object_array(
            [np.asarray(belief.expected_free_energy, dtype=float) for belief in beliefs]
        )

    policy_posterior = None
    if all(belief.policy_posterior is not None for belief in beliefs):
        policy_posterior = _stack_or_object_array(
            [np.asarray(belief.policy_posterior, dtype=float) for belief in beliefs]
        )

    risk = None
    if all(belief.risk is not None for belief in beliefs):
        risk = _stack_or_object_array(
            [np.asarray(belief.risk, dtype=float) for belief in beliefs]
        )

    ambiguity = None
    if all(belief.ambiguity is not None for belief in beliefs):
        ambiguity = _stack_or_object_array(
            [np.asarray(belief.ambiguity, dtype=float) for belief in beliefs]
        )

    return BeliefVisualization(
        time=time,
        posterior_states=posterior_states,
        variational_free_energy=variational_free_energy,
        expected_free_energy=expected_free_energy,
        policy_posterior=policy_posterior,
        risk=risk,
        ambiguity=ambiguity,
    )


def recovery_after_perturbation(
    records: Sequence[SimulationRecord],
    perturbation_time: int | None,
    recovered_position: int = 2,
    min_recovered_energy: int = 2,
) -> int | None:
    if perturbation_time is None:
        return None

    perturbed_index = next(
        (index for index, record in enumerate(records) if record.time == perturbation_time),
        None,
    )
    if perturbed_index is None:
        return None

    pre_perturbation_energy = (
        records[perturbed_index - 1].energy
        if perturbed_index > 0
        else records[perturbed_index].energy
    )
    energy_threshold = max(pre_perturbation_energy, min_recovered_energy)

    for record in records[perturbed_index:]:
        if record.time == perturbation_time:
            if record.energy >= energy_threshold and record.viable:
                return 0
            continue
        if record.energy >= energy_threshold and record.viable:
            return record.time - perturbation_time

    for record in records[perturbed_index:]:
        if record.time <= perturbation_time:
            continue
        if record.position == recovered_position and record.energy >= min_recovered_energy:
            return record.time - perturbation_time
    return None
