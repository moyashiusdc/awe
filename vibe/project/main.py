from __future__ import annotations

import argparse
import os
import sys
import time as time_module
from dataclasses import dataclass
from pathlib import Path

_default_mpl_dir = Path.home() / ".matplotlib"
if "MPLCONFIGDIR" not in os.environ and not os.access(_default_mpl_dir, os.W_OK):
    fallback_mpl_dir = Path("/tmp/mpl")
    fallback_mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback_mpl_dir)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.agent.generative_model import create_survival_agent
from project.agent.inference import ActiveInferenceUpdater
from project.agent.policy import ActiveInferencePolicySelector
from project.analysis.metrics import BeliefSnapshot, SimulationRecord
from project.analysis.ness import NESSReport, analyze_ness
from project.env.blanket import MarkovBlanket
from project.env.perturbations import PerturbationEvent, apply_perturbation
from project.env.viability import ViabilityBounds
from project.env.world import (
    ENERGY_LABELS,
    ExternalState,
    POSITION_LABELS,
    SurvivalWorld,
    WorldConfig,
)
from project.renderer import (
    compose_terminal_frame,
    compose_terminal_frame_live,
    observation_summary,
    render_heatmap_ascii,
    render_histogram_ascii,
    save_experiment_plots,
)


@dataclass(frozen=True)
class ExperimentAction:
    time: int
    kind: str
    params: dict[str, object]


@dataclass(frozen=True)
class SimulationResult:
    trajectory: list[SimulationRecord]
    perturbation_time: int | None
    final_state: ExternalState
    event_log: list[tuple[int, PerturbationEvent]]
    world: SurvivalWorld

    def recorded_state(self, time_index: int) -> ExternalState:
        record = self.trajectory[min(max(time_index, 0), len(self.trajectory) - 1)]
        return ExternalState(
            agent_row=self.final_state.agent_row if record.row is None else record.row,
            agent_col=self.final_state.agent_col if record.col is None else record.col,
            food_positions=(
                self.final_state.food_positions
                if record.food_positions is None
                else record.food_positions
            ),
            hazard_positions=(
                self.final_state.hazard_positions
                if record.hazard_positions is None
                else record.hazard_positions
            ),
            energy=record.energy,
            step_count=record.time,
            reset_count=record.reset_count,
            was_reset=False,
        )


def _parse_value(raw_value: str) -> object:
    if "x" in raw_value:
        left, right = raw_value.split("x", maxsplit=1)
        if left.lstrip("-").isdigit() and right.lstrip("-").isdigit():
            return (int(left), int(right))
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def _parse_action_spec(spec: str) -> ExperimentAction:
    parts = spec.split(":", maxsplit=2)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid experiment action '{spec}'. Use time:kind[:key=value,key=value]."
        )
    time = int(parts[0])
    kind = parts[1]
    params: dict[str, object] = {}
    if len(parts) == 3 and parts[2]:
        for assignment in parts[2].split(","):
            key, value = assignment.split("=", maxsplit=1)
            params[key] = _parse_value(value)
    return ExperimentAction(time=time, kind=kind, params=params)


def _normalize_actions(
    action_specs: list[str] | None,
    perturb_step: int | None,
    energy_drop: int,
    forced_position: int | None,
) -> list[ExperimentAction]:
    actions = [_parse_action_spec(spec) for spec in (action_specs or [])]
    if perturb_step is not None:
        params: dict[str, object] = {"amount": energy_drop}
        if forced_position is not None:
            params["position"] = forced_position
        actions.append(
            ExperimentAction(
                time=perturb_step,
                kind="sudden_energy_shock",
                params=params,
            )
        )
    return sorted(actions, key=lambda action: action.time)


def _actions_by_time(actions: list[ExperimentAction]) -> dict[int, list[ExperimentAction]]:
    schedule: dict[int, list[ExperimentAction]] = {}
    for action in actions:
        schedule.setdefault(action.time, []).append(action)
    return schedule


def _format_distribution(distribution: dict[str, float]) -> str:
    if not distribution:
        return "  <empty>"
    return "\n".join(
        f"  {state_key}: {probability:.3f}"
        for state_key, probability in distribution.items()
    )


def _format_belief_summary(report: NESSReport) -> str:
    belief = report.belief_visualization
    if belief is None:
        return "No belief trace recorded."

    lines = [
        f"Mean variational free energy: {float(belief.variational_free_energy.mean()):.3f}",
        f"Final variational free energy: {float(belief.variational_free_energy[-1]):.3f}",
    ]
    if belief.expected_free_energy is not None:
        lines.append(f"Final expected free energy: {belief.expected_free_energy[-1]}")
    if belief.policy_posterior is not None:
        lines.append(f"Final policy posterior: {belief.policy_posterior[-1]}")
    if belief.risk is not None:
        lines.append(f"Final risk: {belief.risk[-1]}")
    if belief.ambiguity is not None:
        lines.append(f"Final ambiguity: {belief.ambiguity[-1]}")
    for factor_index, factor_trace in enumerate(belief.posterior_states):
        lines.append(f"Factor {factor_index} posterior: {factor_trace[-1]}")
    return "\n".join(lines)


def _print_report(
    report: NESSReport,
    result: SimulationResult,
    actions: list[ExperimentAction],
    experimenter_mode: bool,
    artifact_dir: Path,
) -> None:
    final_record = result.trajectory[-1]
    print(
        compose_terminal_frame(
            world=result.world,
            result=result,
            report=report,
            actions=actions,
            experimenter_mode=experimenter_mode,
            state=result.final_state,
            record=final_record,
        )
    )
    print("")
    print("Protocol:")
    print(f"  steps={len(result.trajectory) - 1}")
    print(f"  food_count={result.world.config.num_food_sources}")
    print(f"  hazard_count={result.world.config.num_hazards}")
    print(f"  energy_decay_per_step={result.world.config.step_cost}")
    print(f"  food_respawn_rate={result.world.config.food_respawn_rate:.2f}")
    print(f"  sensory_noise={result.world.base_sensory_noise:.2f}")
    if actions:
        for action in actions:
            print(f"  t={action.time} -> {action.kind} {action.params}")
    else:
        print("  no scheduled perturbations")
    print("")
    print("Results:")
    print(
        "  final_state="
        f"({final_record.row}, {final_record.col}) "
        f"{POSITION_LABELS[final_record.position]}/"
        f"{ENERGY_LABELS[result.world.energy_level_index(final_record.energy)]}"
    )
    print(f"  survival_time={report.survival_time}")
    print(f"  reset_count={report.reset_count}")
    print(f"  average_energy={report.average_energy:.3f}")
    print(f"  viable_state_fraction={report.fraction_time_viable:.3f}")
    print(f"  recovery_time_after_perturbation={report.recovery_time_after_perturbation}")
    print(f"  perturbation_counts={report.perturbation_counts}")
    print("")
    print("Energy Histogram:")
    print(render_histogram_ascii(report.energy_histogram))
    print("")
    print("Position Occupancy Heatmap:")
    print(render_heatmap_ascii(report.position_occupancy_heatmap))
    print("")
    print("Belief Summary:")
    print(_format_belief_summary(report))
    print("")
    print("State Occupancy Distribution:")
    print(_format_distribution(report.state_occupancy_distribution))
    print("Long-run State Distribution:")
    print(_format_distribution(report.long_run_state_distribution))
    print("")
    print("Saved Views:")
    print(f"  energy_over_time={artifact_dir / 'energy_over_time.png'}")
    print(f"  occupancy_heatmap={artifact_dir / 'position_occupancy_heatmap.png'}")
    if result.event_log:
        print("Event Log:")
        for time_index, event in result.event_log:
            print(f"  t={time_index} -> {event.kind} {event.details}")


def run_simulation(
    steps: int,
    policy_len: int,
    world_config: WorldConfig | None = None,
    scheduled_actions: list[ExperimentAction] | None = None,
    live_ui: bool = False,
    frame_delay: float = 0.08,
) -> SimulationResult:
    world = SurvivalWorld(config=world_config)
    blanket = MarkovBlanket()
    viability = ViabilityBounds(max_viable_energy=world.config.max_energy)
    agent = create_survival_agent(world=world, policy_len=policy_len)
    inference = ActiveInferenceUpdater(agent=agent, blanket=blanket)
    policy_selector = ActiveInferencePolicySelector(agent=agent, blanket=blanket)

    action_schedule = _actions_by_time(scheduled_actions or [])
    external_state = world.initial_state()
    sensory_state = blanket.external_to_sensory(world, external_state)
    perturbation_time: int | None = None
    trajectory: list[SimulationRecord] = []
    event_log: list[tuple[int, PerturbationEvent]] = []

    for time_index in range(steps):
        applied_event: PerturbationEvent | None = None
        if time_index in action_schedule:
            for action in action_schedule[time_index]:
                external_state, applied_event = apply_perturbation(
                    world,
                    external_state,
                    action.kind,
                    **action.params,
                )
                event_log.append((time_index, applied_event))
                if perturbation_time is None:
                    perturbation_time = time_index
            sensory_state = blanket.external_to_sensory(world, external_state)

        viable = viability.contains_state(external_state)
        internal_state = inference.infer(sensory_state)
        internal_state = policy_selector.evaluate_policies(internal_state)
        active_state = policy_selector.select_action(internal_state)
        belief = BeliefSnapshot.from_internal_state(internal_state)
        partial_observation = world.observe_partial(external_state)
        trajectory.append(
            SimulationRecord(
                time=time_index,
                position=external_state.position,
                energy=external_state.energy,
                viable=viable,
                action=active_state.action_label,
                perturbed=applied_event is not None,
                row=external_state.agent_row,
                col=external_state.agent_col,
                reset_count=external_state.reset_count,
                perturbation_kind=None if applied_event is None else applied_event.kind,
                belief=belief,
                observation_summary=observation_summary(partial_observation),
                food_positions=external_state.food_positions,
                hazard_positions=external_state.hazard_positions,
            )
        )

        if live_ui:
            print("\033[2J\033[H", end="")
            print(
                compose_terminal_frame_live(
                    world=world,
                    trajectory=trajectory,
                    actions=scheduled_actions or [],
                    experimenter_mode=True,
                    state=external_state,
                    record=trajectory[-1],
                )
            )
            time_module.sleep(frame_delay)

        next_external_state = blanket.active_to_external(world, external_state, active_state)
        next_sensory_state = blanket.external_to_sensory(world, next_external_state)
        inference.learn_from_transition(
            next_sensory_state=next_sensory_state,
            previous_posterior_states=internal_state.posterior_states,
        )
        external_state = next_external_state
        sensory_state = next_sensory_state

    final_viable = viability.contains_state(external_state)
    trajectory.append(
        SimulationRecord(
            time=steps,
            position=external_state.position,
            energy=external_state.energy,
            viable=final_viable,
            action=None,
            perturbed=False,
            row=external_state.agent_row,
            col=external_state.agent_col,
            reset_count=external_state.reset_count,
            perturbation_kind=None,
            belief=None,
            observation_summary=observation_summary(world.observe_partial(external_state)),
            food_positions=external_state.food_positions,
            hazard_positions=external_state.hazard_positions,
        )
    )

    return SimulationResult(
        trajectory=trajectory,
        perturbation_time=perturbation_time,
        final_state=external_state,
        event_log=event_log,
        world=world,
    )


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent / "output"
    parser = argparse.ArgumentParser(
        description="Artificial life experiment shell around an active-inference survival agent."
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps.")
    parser.add_argument(
        "--policy-len",
        type=int,
        default=3,
        help="Policy horizon used by pymdp policy evaluation.",
    )
    parser.add_argument(
        "--experimenter-mode",
        action="store_true",
        help="Print the simulation as an experiment report rather than a toy game summary.",
    )
    parser.add_argument(
        "--live-ui",
        action="store_true",
        help="Animate a stable terminal UI frame for each simulation step.",
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.01,
        help="Seconds to pause between live terminal UI frames.",
    )
    parser.add_argument("--food-count", type=int, default=3, help="Initial number of food sources.")
    parser.add_argument("--hazard-count", type=int, default=2, help="Initial number of hazards.")
    parser.add_argument(
        "--energy-decay",
        type=int,
        default=1,
        help="Metabolic cost paid on every step.",
    )
    parser.add_argument(
        "--food-respawn-rate",
        type=float,
        default=0.15,
        help="Probability of respawning a missing food source after each step.",
    )
    parser.add_argument(
        "--sensory-noise",
        type=float,
        default=0.0,
        help="Baseline sensory corruption rate in [0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for world layout, respawn, and perturbation placement.",
    )
    parser.add_argument(
        "--experiment-action",
        action="append",
        default=[],
        help=(
            "Scheduled environment intervention formatted as "
            "time:kind[:key=value,key=value]. "
            "Kinds: sudden_energy_shock, spawn_extra_hazard, remove_one_food_source, "
            "random_displacement, temporary_sensory_noise_increase."
        ),
    )
    parser.add_argument(
        "--perturb-step",
        type=int,
        default=None,
        help="Legacy single perturbation time for a sudden energy shock.",
    )
    parser.add_argument(
        "--energy-drop",
        type=int,
        default=2,
        help="Legacy sudden-energy-shock magnitude.",
    )
    parser.add_argument(
        "--forced-position",
        type=int,
        default=None,
        help="Legacy forced position for the legacy perturbation interface.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=25,
        help="Ignore the first burn-in states when computing long-run occupancy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Directory for saved experiment plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_config = WorldConfig(
        num_food_sources=args.food_count,
        num_hazards=args.hazard_count,
        step_cost=args.energy_decay,
        food_respawn_rate=args.food_respawn_rate,
        sensory_noise=args.sensory_noise,
        random_seed=args.seed,
    )
    actions = _normalize_actions(
        action_specs=args.experiment_action,
        perturb_step=args.perturb_step,
        energy_drop=args.energy_drop,
        forced_position=args.forced_position,
    )
    try:
        result = run_simulation(
            steps=args.steps,
            policy_len=args.policy_len,
            world_config=world_config,
            scheduled_actions=actions,
            live_ui=args.live_ui,
            frame_delay=args.frame_delay,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(str(exc)) from exc

    report = analyze_ness(
        result.trajectory,
        perturbation_time=result.perturbation_time,
        burn_in=args.burn_in,
        grid_size=result.world.grid_size,
        max_energy=result.world.config.max_energy,
    )
    artifact_dir = Path(args.output_dir)
    save_experiment_plots(result=result, report=report, output_dir=artifact_dir)

    if args.live_ui:
        print("")

    _print_report(
        report=report,
        result=result,
        actions=actions,
        experimenter_mode=args.experimenter_mode,
        artifact_dir=artifact_dir,
    )


if __name__ == "__main__":
    main()
