import numpy as np
from qutip import (
    Qobj,
    operator_to_vector,
    vector_to_operator,
)

from quantum_helpers import operator_basis_lidar
from utils import project_U


class Fidelity:
    """
    Parameters
    ----------
    basis_states: list
        list of the basis states with which to construct the Lidar basis (coherences)
        see 10.1103/PhysRevA.77.032322 for more detail (note typo in the paper, missing an i)
    label_list: list
        labels that apply to the basis states. If not provided, we provide
        one for you free of charge
    """

    def __init__(self, basis_states: list, label_list: list = None):
        self.basis_states = basis_states
        if label_list is None:
            self.label_list = range(len(basis_states))
        else:
            self.label_list = label_list

    @staticmethod
    def process_fidelity_nielsen(entanglement_fidelity, num_qubits=2):
        """standard formula for process fidelity"""
        dim = num_qubits**2
        return (dim * entanglement_fidelity + 1) / (dim + 1)

    def entanglement_fidelity_nielsen(
        self,
        prop_or_final_states_dict,
        U_ideal,
        measurement_op=None,
        ptrace_idxs=None,
        num_qubits=2,
    ) -> (float, float):
        """
        Parameters
        ----------
        prop_or_final_states_dict: Qobj or dict
            either the propogator (superoperator) corresponding to the real time evolution
            or a dictionary of how the basis states of interest evolve
        U_ideal: Qobj
            propogator of the ideal evolution
        measurement_op: Qobj
            measurement operator, if any
        ptrace_idxs: tuple
            indices to keep if we want to trace over a subsystem
        num_qubits: int
            number of qubits, usually here 2

        Returns
        -------
            returns a tuple of floats corresponding to the entanglement fidelity
            according the Nielsen's formula together with the success probability

        """
        dim = 2**num_qubits
        op_dict, unique_state_dict = operator_basis_lidar(
            self.basis_states, self.label_list
        )
        overall_contr = 0.0
        total_prob = 0.0
        # TODO want to change num_states indexing so that we only sum over unique states
        num_states = 0
        for op_key in op_dict.keys():
            op, coeffs, rhos, labels = op_dict[op_key]
            for (coeff, pauli_rho, label) in zip(coeffs, rhos, labels):
                # we've constructed the superoperator
                if type(prop_or_final_states_dict) == Qobj:
                    rho = operator_to_vector(pauli_rho)
                    propagated_rho = vector_to_operator(prop_or_final_states_dict * rho)
                # we've tracked individual states instead
                else:
                    propagated_rho = prop_or_final_states_dict[label]
                state_contr, prob = self._fidel_individual_state(
                    propagated_rho,
                    op,
                    U_ideal,
                    measurement_op,
                    ptrace_idxs,
                )
                overall_contr += coeff * state_contr
                total_prob += prob
                num_states += 1
        return overall_contr / dim**2, total_prob / num_states

    def _fidel_individual_state(
        self,
        propagated_rho,
        op,
        U_ideal,
        measurement_op=None,
        ptrace_idxs=None,
    ):
        """see above function for documentation"""
        if measurement_op is not None:
            propagated_rho = measurement_op * propagated_rho * measurement_op.dag()
            prob = np.trace(propagated_rho)
        else:
            prob = 0.0
        if ptrace_idxs is not None:
            propagated_rho = propagated_rho.ptrace(ptrace_idxs)
            op = op.ptrace(ptrace_idxs)
        projected_rho = project_U(propagated_rho, basis_states=self.basis_states)
        projected_op = project_U(op, basis_states=self.basis_states)
        # below formula straight out of Nielsen's paper
        state_contr = np.trace(
            U_ideal * projected_op.dag() * U_ideal.dag() * projected_rho
        )
        return state_contr, prob
