from __future__ import annotations

import numpy as np

try:
    from pymdp import control, maths, utils
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime
    control = None
    maths = None
    utils = None
    PYMDP_IMPORT_ERROR = exc
else:
    PYMDP_IMPORT_ERROR = None

from project.env.blanket import InternalState, MarkovBlanket, SensoryState


class ActiveInferenceUpdater:
    """Bayesian belief updates and Dirichlet learning around a pymdp agent."""

    def __init__(self, agent, blanket: MarkovBlanket) -> None:
        if control is None or maths is None or utils is None:  # pragma: no cover - runtime
            raise ImportError(
                "pymdp is required. Install it with `python3 -m pip install inferactively-pymdp`."
            ) from PYMDP_IMPORT_ERROR
        self.agent = agent
        self.blanket = blanket

    def _empirical_prior(self) -> np.ndarray:
        if hasattr(self.agent, "qs") and self.agent.action is not None:
            return control.get_expected_states(
                self.agent.qs,
                self.agent.B,
                self.agent.action.reshape(1, -1),
            )[0]
        return self.agent.D

    def infer(self, sensory_state: SensoryState) -> InternalState:
        empirical_prior = self._empirical_prior()
        posterior_states = self.agent.infer_states(sensory_state.as_observation())
        processed_observation = utils.process_observation(
            sensory_state.as_observation(),
            self.agent.num_modalities,
            self.agent.num_obs,
        )
        log_likelihood = maths.spm_log_single(
            maths.get_joint_likelihood(
                self.agent.A,
                processed_observation,
                self.agent.num_states,
            )
        )
        variational_free_energy = float(
            maths.calc_free_energy(
                posterior_states,
                maths.spm_log_obj_array(empirical_prior),
                len(self.agent.num_states),
                log_likelihood,
            )
        )
        return self.blanket.sensory_to_internal(
            posterior_states=posterior_states,
            empirical_prior=empirical_prior,
            variational_free_energy=variational_free_energy,
        )

    def learn_from_transition(
        self,
        next_sensory_state: SensoryState,
        previous_posterior_states,
    ) -> None:
        self.agent.update_A(next_sensory_state.as_observation())
        if previous_posterior_states is not None:
            self.agent.update_B(previous_posterior_states)
