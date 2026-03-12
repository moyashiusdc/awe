from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import textwrap
from typing import Sequence

_default_mpl_dir = Path.home() / ".matplotlib"
if "MPLCONFIGDIR" not in os.environ and not os.access(_default_mpl_dir, os.W_OK):
    fallback_mpl_dir = Path("/tmp/mpl")
    fallback_mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.analysis.ness import NESSReport
from project.env.observations import (
    DANGER_SIGNAL_LABELS,
    ENERGY_SIGNAL_LABELS,
    FOOD_SMELL_LABELS,
)
from project.env.world import POSITION_LABELS


PANEL_WIDTH = 58


@dataclass(frozen=True)
class RenderArtifacts:
    energy_plot_path: Path
    heatmap_plot_path: Path


def format_array(values: np.ndarray | None) -> str:
    if values is None:
        return "n/a"
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size <= 8:
        return np.array2string(array, precision=3, suppress_small=True)
    head = np.array2string(array[:4], precision=3, suppress_small=True)
    tail = np.array2string(array[-4:], precision=3, suppress_small=True)
    return f"{head} ... {tail} (n={array.size})"


def observation_summary(partial_observation) -> str:
    return (
        f"food={FOOD_SMELL_LABELS[partial_observation.food_smell]}, "
        f"danger={DANGER_SIGNAL_LABELS[partial_observation.danger_signal]}, "
        f"energy={ENERGY_SIGNAL_LABELS[partial_observation.energy_signal]}"
    )


def _grid_rows(world, state) -> list[str]:
    rows: list[str] = []
    for row in range(world.grid_size):
        chars: list[str] = []
        for col in range(world.grid_size):
            position = (row, col)
            if position == state.agent_position:
                chars.append("A")
            elif position in state.food_positions:
                chars.append("F")
            elif position in state.hazard_positions:
                chars.append("X")
            else:
                chars.append(".")
        rows.append(" ".join(chars))
    return rows


def boxed_grid(world, state) -> list[str]:
    grid_rows = _grid_rows(world, state)
    row_width = len(grid_rows[0]) + 2
    horizontal = "+" + ("-" * row_width) + "+"
    boxed = [horizontal]
    for row in grid_rows:
        boxed.append(f"| {row} |")
    boxed.append(horizontal)
    return boxed


def _posterior_lines(belief) -> list[str]:
    if belief is None:
        return ["Posterior states: n/a"]
    lines = []
    for factor_index, posterior in enumerate(belief.posterior_states):
        lines.append(f"Posterior {factor_index}: {format_array(np.asarray(posterior))}")
    if belief.policy_posterior is not None:
        lines.append(f"Policy q(pi): {format_array(belief.policy_posterior)}")
    return lines


def status_panel_lines(
    world,
    result,
    report: NESSReport,
    actions,
    state,
    record,
) -> list[str]:
    active_record = next(
        (trajectory_record for trajectory_record in reversed(result.trajectory) if trajectory_record.belief is not None),
        record,
    )
    belief = active_record.belief
    lines = [
        "STATUS PANEL",
        f"Energy           {record.energy:>2}/{world.config.max_energy}",
        f"Observation      {record.observation_summary or 'n/a'}",
        f"Chosen action    {active_record.action or 'n/a'}",
        f"Survival steps   {record.time}",
        f"Reset count      {record.reset_count}",
        f"Tile class       {POSITION_LABELS[record.position]}",
        f"Grid position    ({record.row}, {record.col})",
        "",
        "ACTIVE INFERENCE",
        f"Variational F    {('n/a' if belief is None else f'{belief.variational_free_energy:.3f}')}",
        f"Expected F       {format_array(None if belief is None else belief.expected_free_energy)}",
        f"Risk             {format_array(None if belief is None else belief.risk)}",
        f"Ambiguity        {format_array(None if belief is None else belief.ambiguity)}",
    ]
    lines.extend(_posterior_lines(belief))
    lines.extend(
        [
            "",
            "EXPERIMENT",
            f"Viable fraction  {report.fraction_time_viable:.3f}",
            f"Average energy   {report.average_energy:.3f}",
            f"Recovery time    {report.recovery_time_after_perturbation}",
            f"Events queued    {len(actions)}",
        ]
    )
    if result.event_log:
        event_time, event = result.event_log[-1]
        lines.append(f"Last event       t={event_time} {event.kind}")
    else:
        lines.append("Last event       none")
    return lines


def status_panel_lines_live(
    world,
    trajectory,
    actions,
    state,
    record,
) -> list[str]:
    active_record = next(
        (trajectory_record for trajectory_record in reversed(trajectory) if trajectory_record.belief is not None),
        record,
    )
    belief = active_record.belief
    viable_fraction = (
        sum(1 for trajectory_record in trajectory if trajectory_record.viable) / float(len(trajectory))
        if trajectory
        else 0.0
    )
    average_energy = (
        sum(trajectory_record.energy for trajectory_record in trajectory) / float(len(trajectory))
        if trajectory
        else 0.0
    )
    lines = [
        "STATUS PANEL",
        f"Energy           {record.energy:>2}/{world.config.max_energy}",
        f"Observation      {record.observation_summary or 'n/a'}",
        f"Chosen action    {active_record.action or 'n/a'}",
        f"Survival steps   {record.time}",
        f"Reset count      {record.reset_count}",
        f"Tile class       {POSITION_LABELS[record.position]}",
        f"Grid position    ({record.row}, {record.col})",
        "",
        "ACTIVE INFERENCE",
        f"Variational F    {('n/a' if belief is None else f'{belief.variational_free_energy:.3f}')}",
        f"Expected F       {format_array(None if belief is None else belief.expected_free_energy)}",
        f"Risk             {format_array(None if belief is None else belief.risk)}",
        f"Ambiguity        {format_array(None if belief is None else belief.ambiguity)}",
    ]
    lines.extend(_posterior_lines(belief))
    lines.extend(
        [
            "",
            "EXPERIMENT",
            f"Viable fraction  {viable_fraction:.3f}",
            f"Average energy   {average_energy:.3f}",
            "Recovery time    running",
            f"Events queued    {len(actions)}",
        ]
    )
    return lines


def compose_terminal_frame(
    world,
    result,
    report: NESSReport,
    actions,
    experimenter_mode: bool,
    state,
    record,
) -> str:
    title = "ARTIFICIAL LIFE EXPERIMENT"
    subtitle = "A=agent  F=food  X=hazard  .=empty"
    grid_lines = boxed_grid(world, state)
    raw_panel_lines = status_panel_lines(
        world=world,
        result=result,
        report=report,
        actions=actions,
        state=state,
        record=record,
    )
    panel_lines: list[str] = []
    for line in raw_panel_lines:
        wrapped = textwrap.wrap(line, width=PANEL_WIDTH) or [""]
        panel_lines.extend(segment.ljust(PANEL_WIDTH) for segment in wrapped)
    width = max(len(line) for line in grid_lines)
    max_lines = max(len(grid_lines), len(panel_lines))
    combined: list[str] = [title, subtitle, ""]
    for index in range(max_lines):
        left = grid_lines[index] if index < len(grid_lines) else " " * width
        right = panel_lines[index] if index < len(panel_lines) else " " * PANEL_WIDTH
        combined.append(f"{left.ljust(width)}   {right}")
    if experimenter_mode:
        combined.append("")
        combined.append("Mode: experimenter controls the ecology; the agent controls itself.")
    return "\n".join(combined)


def compose_terminal_frame_live(
    world,
    trajectory,
    actions,
    experimenter_mode: bool,
    state,
    record,
) -> str:
    title = "ARTIFICIAL LIFE EXPERIMENT"
    subtitle = "A=agent  F=food  X=hazard  .=empty"
    grid_lines = boxed_grid(world, state)
    raw_panel_lines = status_panel_lines_live(
        world=world,
        trajectory=trajectory,
        actions=actions,
        state=state,
        record=record,
    )
    panel_lines: list[str] = []
    for line in raw_panel_lines:
        wrapped = textwrap.wrap(line, width=PANEL_WIDTH) or [""]
        panel_lines.extend(segment.ljust(PANEL_WIDTH) for segment in wrapped)
    width = max(len(line) for line in grid_lines)
    max_lines = max(len(grid_lines), len(panel_lines))
    combined: list[str] = [title, subtitle, ""]
    for index in range(max_lines):
        left = grid_lines[index] if index < len(grid_lines) else " " * width
        right = panel_lines[index] if index < len(panel_lines) else " " * PANEL_WIDTH
        combined.append(f"{left.ljust(width)}   {right}")
    if experimenter_mode:
        combined.append("")
        combined.append("Mode: experimenter controls the ecology; the agent controls itself.")
    return "\n".join(combined)


def render_heatmap_ascii(heatmap: np.ndarray) -> str:
    if heatmap.size == 0:
        return "<empty>"
    palette = " .:-=+*#%@"
    max_value = float(np.max(heatmap))
    if max_value <= 0.0:
        return "\n".join("".join(" " for _ in row) for row in heatmap)
    rows: list[str] = []
    for row in heatmap:
        chars = []
        for value in row:
            scaled = value / max_value
            index = min(int(round(scaled * (len(palette) - 1))), len(palette) - 1)
            chars.append(palette[index])
        rows.append("".join(chars))
    return "\n".join(rows)


def render_histogram_ascii(histogram: np.ndarray) -> str:
    if histogram.size == 0:
        return "<empty>"
    max_count = int(np.max(histogram))
    if max_count <= 0:
        return "\n".join(f"{energy:2d}: " for energy in range(len(histogram)))
    return "\n".join(
        f"{energy:2d}: {'#' * int(round((count / max_count) * 24))} ({int(count)})"
        for energy, count in enumerate(histogram)
    )


def animate_terminal_frames(
    world,
    result,
    report: NESSReport,
    actions,
    experimenter_mode: bool,
    frame_delay: float,
    sleep_fn,
) -> None:
    print("\033[?25l", end="")
    try:
        for record in result.trajectory:
            state = result.recorded_state(record.time)
            print("\033[2J\033[H", end="")
            print(
                compose_terminal_frame(
                    world=world,
                    result=result,
                    report=report,
                    actions=actions,
                    experimenter_mode=experimenter_mode,
                    state=state,
                    record=record,
                )
            )
            sleep_fn(frame_delay)
    finally:
        print("\033[?25h", end="")


def save_experiment_plots(result, report: NESSReport, output_dir: str | Path) -> RenderArtifacts:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    times = [record.time for record in result.trajectory]
    energies = [record.energy for record in result.trajectory]

    energy_plot_path = output_path / "energy_over_time.png"
    heatmap_plot_path = output_path / "position_occupancy_heatmap.png"

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(times, energies, color="#1f4e5f", linewidth=2.2)
    ax.set_title("Energy Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(energy_plot_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.imshow(report.position_occupancy_heatmap, cmap="YlGnBu", origin="upper")
    ax.set_title("Position Occupancy Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heatmap_plot_path, dpi=160)
    plt.close(fig)

    return RenderArtifacts(
        energy_plot_path=energy_plot_path,
        heatmap_plot_path=heatmap_plot_path,
    )
