from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from project.env.world import (
    ACTION_LABELS,
    ENERGY_LABELS,
    GRID_ACTION_DELTAS,
    POSITION_LABELS,
    SurvivalWorld,
)

try:
    from pymdp.agent import Agent
    from pymdp import utils
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime
    Agent = None
    utils = None
    PYMDP_IMPORT_ERROR = exc
else:
    PYMDP_IMPORT_ERROR = None


def _dirichlet_counts(categorical: np.ndarray, concentration: float = 16.0) -> np.ndarray:
    return 1.0 + concentration * categorical


def _energy_delta(world: SurvivalWorld, next_position: tuple[int, int]) -> int:
    delta = -world.config.step_cost
    if next_position in world.food_positions:
        delta += world.config.food_reward
    elif next_position in world.hazard_positions:
        delta -= world.config.hazard_penalty
    return delta


def build_position_transition_tensor(world: SurvivalWorld) -> np.ndarray:
    """Build B[next_position, current_position, action] for an 11x11 grid."""

    num_positions = world.grid_size * world.grid_size
    num_actions = len(ACTION_LABELS)
    position_B = np.zeros((num_positions, num_positions, num_actions), dtype=float)

    for current_position_index in range(num_positions):
        row, col = world.index_to_position(current_position_index)
        for action_index, (delta_row, delta_col) in enumerate(GRID_ACTION_DELTAS):
            next_position = world.clamp_position((row + delta_row, col + delta_col))
            next_position_index = world.position_to_index(next_position)
            position_B[next_position_index, current_position_index, action_index] = 1.0

    return position_B


def _validate_position_transition_tensor(position_B: np.ndarray, world: SurvivalWorld) -> None:
    if not np.allclose(position_B.sum(axis=0), 1.0):
        raise ValueError("Each (current_state, action) pair must map to exactly one next grid state.")

    center_index = world.position_to_index(world.config.start_position)
    reachable = {
        int(np.argmax(position_B[:, center_index, action_index]))
        for action_index in range(position_B.shape[2])
    }
    if len(reachable) < 5:
        raise ValueError("All five actions must reach distinct states from the interior of the grid.")


@dataclass(frozen=True)
class SurvivalGenerativeModel:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    pA: np.ndarray
    pB: np.ndarray
    control_fac_idx: list[int]
    policy_len: int


def build_survival_generative_model(
    world: SurvivalWorld | None = None,
    policy_len: int = 1,
) -> SurvivalGenerativeModel:
    """Build a true 2D active-inference model for the 11x11 survival world."""

    if utils is None:  # pragma: no cover - exercised at runtime
        raise ImportError(
            "pymdp is required. Install it with `python3 -m pip install inferactively-pymdp`."
        ) from PYMDP_IMPORT_ERROR

    world = world or SurvivalWorld()
    num_positions = world.grid_size * world.grid_size
    num_tile_classes = len(POSITION_LABELS)
    num_energy_observations = len(ENERGY_LABELS)
    num_exact_energy_states = world.config.max_energy + 1
    num_joint_states = num_positions * num_exact_energy_states
    num_actions = len(ACTION_LABELS)

    position_B = build_position_transition_tensor(world)
    _validate_position_transition_tensor(position_B, world)

    A = utils.obj_array(3)
    B = utils.obj_array(1)
    C = utils.obj_array(3)
    D = utils.obj_array(1)
    pA = utils.obj_array(3)
    pB = utils.obj_array(1)

    position_obs_noise = 0.02
    tile_obs_noise = 0.05
    energy_obs_noise = 0.08

    A[0] = np.full(
        (num_positions, num_joint_states),
        position_obs_noise / (num_positions - 1),
        dtype=float,
    )
    A[1] = np.full(
        (num_tile_classes, num_joint_states),
        tile_obs_noise / (num_tile_classes - 1),
        dtype=float,
    )
    A[2] = np.full(
        (num_energy_observations, num_joint_states),
        energy_obs_noise / (num_energy_observations - 1),
        dtype=float,
    )

    B[0] = np.zeros((num_joint_states, num_joint_states, num_actions), dtype=float)

    for position_index in range(num_positions):
        position = world.index_to_position(position_index)
        tile_class = world.tile_class_index(position)
        for energy in range(num_exact_energy_states):
            joint_index = (position_index * num_exact_energy_states) + energy
            A[0][position_index, joint_index] = 1.0 - position_obs_noise
            A[1][tile_class, joint_index] = 1.0 - tile_obs_noise
            A[2][world.energy_level_index(energy), joint_index] = 1.0 - energy_obs_noise

            for action_index in range(num_actions):
                next_position_index = int(np.argmax(position_B[:, position_index, action_index]))
                next_position = world.index_to_position(next_position_index)
                next_energy = int(
                    np.clip(
                        energy + _energy_delta(world, next_position),
                        0,
                        world.config.max_energy,
                    )
                )
                if next_energy <= 0:
                    next_position = world.config.start_position
                    next_position_index = world.position_to_index(next_position)
                    next_energy = world.config.start_energy

                next_joint_index = (next_position_index * num_exact_energy_states) + next_energy
                B[0][next_joint_index, joint_index, action_index] = 1.0

    # Position observations are informational but not directly preferred.
    # Survival value should come from actual tile class and energy outcomes.
    C[0] = np.zeros(num_positions, dtype=float)
    C[1] = np.array([-6.0, 0.0, 4.5], dtype=float)
    C[2] = np.array([-10.0, -3.0, 1.5, 4.0], dtype=float)

    start_position_index = world.position_to_index(world.config.start_position)
    start_joint_index = (start_position_index * num_exact_energy_states) + world.config.start_energy
    D[0] = utils.onehot(start_joint_index, num_joint_states).astype(float)

    pA[0] = _dirichlet_counts(A[0])
    pA[1] = _dirichlet_counts(A[1])
    pA[2] = _dirichlet_counts(A[2])
    pB[0] = _dirichlet_counts(B[0], concentration=8.0)

    return SurvivalGenerativeModel(
        A=A,
        B=B,
        C=C,
        D=D,
        pA=pA,
        pB=pB,
        control_fac_idx=[0],
        policy_len=policy_len,
    )


def create_survival_agent(
    world: SurvivalWorld | None = None,
    policy_len: int = 1,
) -> Agent:
    if Agent is None:  # pragma: no cover - exercised at runtime
        raise ImportError(
            "pymdp is required. Install it with `python3 -m pip install inferactively-pymdp`."
        ) from PYMDP_IMPORT_ERROR

    model = build_survival_generative_model(world=world, policy_len=policy_len)
    return Agent(
        A=model.A,
        B=model.B,
        C=model.C,
        D=model.D,
        pA=model.pA,
        pB=model.pB,
        policy_len=model.policy_len,
        control_fac_idx=model.control_fac_idx,
        action_selection="stochastic",
        use_utility=False,
        use_states_info_gain=False,
    )
