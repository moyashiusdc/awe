from __future__ import annotations

import copy

import numpy as np

try:
    from pymdp import control, maths
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime
    control = None
    maths = None
    PYMDP_IMPORT_ERROR = exc
else:
    PYMDP_IMPORT_ERROR = None

from project.env.blanket import ActiveState, InternalState, MarkovBlanket


class ActiveInferencePolicySelector:
    """Active state selection through explicit expected free energy minimization."""

    def __init__(self, agent, blanket: MarkovBlanket) -> None:
        if control is None or maths is None:  # pragma: no cover - exercised at runtime
            raise ImportError(
                "pymdp is required. Install it with `python3 -m pip install inferactively-pymdp`."
            ) from PYMDP_IMPORT_ERROR
        self.agent = agent
        self.blanket = blanket

    def _preference_probabilities(self, horizon: int) -> np.ndarray:
        tiled_preferences = copy.deepcopy(self.agent.C)
        for modality, preference in enumerate(tiled_preferences):
            if preference.ndim == 1:
                tiled_preferences[modality] = np.tile(preference[:, None], (1, horizon))
        return maths.softmax_obj_arr(tiled_preferences)

    def _compute_risk(self, qo_pi, preference_probabilities: np.ndarray) -> float:
        risk = 0.0
        for t, qo_t in enumerate(qo_pi):
            for modality, qo_modality in enumerate(qo_t):
                pref = preference_probabilities[modality][:, t]
                risk += float(
                    qo_modality.dot(
                        maths.spm_log_single(qo_modality) - maths.spm_log_single(pref)
                    )
                )
        return risk

    def _compute_ambiguity(self, qs_pi) -> float:
        ambiguity = 0.0
        for qs_t in qs_pi:
            joint_qs = qs_t[0]
            for factor_index in range(1, len(qs_t)):
                joint_qs = joint_qs[..., None] * qs_t[factor_index]

            for likelihood in self.agent.A:
                conditional_entropy = -np.sum(
                    likelihood * maths.spm_log_single(likelihood),
                    axis=0,
                )
                ambiguity += float(np.sum(conditional_entropy * joint_qs))
        return ambiguity

    def evaluate_policies(self, internal_state: InternalState) -> InternalState:
        horizon = self.agent.policies[0].shape[0]
        preference_probabilities = self._preference_probabilities(horizon)

        risk = np.zeros(len(self.agent.policies), dtype=float)
        ambiguity = np.zeros(len(self.agent.policies), dtype=float)
        expected_free_energy = np.zeros(len(self.agent.policies), dtype=float)

        for policy_index, policy in enumerate(self.agent.policies):
            qs_pi = control.get_expected_states(
                internal_state.posterior_states,
                self.agent.B,
                policy,
            )
            qo_pi = control.get_expected_obs(qs_pi, self.agent.A)
            risk[policy_index] = self._compute_risk(qo_pi, preference_probabilities)
            ambiguity[policy_index] = self._compute_ambiguity(qs_pi)
            expected_free_energy[policy_index] = (
                risk[policy_index] + ambiguity[policy_index]
            )

        if self.agent.E is None:
            lnE = maths.spm_log_single(
                np.ones(len(self.agent.policies), dtype=float) / len(self.agent.policies)
            )
        else:
            lnE = maths.spm_log_single(self.agent.E)

        policy_posterior = maths.softmax((-expected_free_energy * self.agent.gamma) + lnE)
        return self.blanket.sensory_to_internal(
            posterior_states=internal_state.posterior_states,
            empirical_prior=internal_state.empirical_prior,
            variational_free_energy=internal_state.variational_free_energy,
            policy_posterior=policy_posterior,
            expected_free_energy=expected_free_energy,
            risk=risk,
            ambiguity=ambiguity,
        )

    def select_action(self, internal_state: InternalState) -> ActiveState:
        if internal_state.expected_free_energy is None or internal_state.policy_posterior is None:
            raise ValueError("Policies must be evaluated before selecting an action.")

        if getattr(self.agent, "action_selection", "deterministic") == "stochastic":
            policy_index = int(
                np.random.choice(
                    len(self.agent.policies),
                    p=np.asarray(internal_state.policy_posterior, dtype=float),
                )
            )
        else:
            policy_index = int(np.argmax(internal_state.policy_posterior))

        selected_policy = self.agent.policies[policy_index]
        self.agent.q_pi = internal_state.policy_posterior
        self.agent.action = selected_policy[0].astype(float)
        return self.blanket.internal_to_active(self.agent.action)
