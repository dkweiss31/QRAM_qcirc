from qutip import (
    Qobj,
    operator_to_vector,
    vector_to_operator,
)
import numpy as np
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

    def operator_basis_lidar(self) -> (dict, dict):
        """

        Returns
        -------
            a tuple of dictionaries. The first dictionary contains information on the operators
            whose evolution we want to track. The keys correspond to the coherence, e.g. "12"
            for the coherence |1><2| or 11 for the "coherence" |1><1|. The values are tuples containing
            four pieces of information. first the operator in question, next a tuple of coefficients
            in the state decomposition (really density matrices, but call them "states" to differentiate from the
            operator coherences which are not density matrices) of the operator, next is those states,
            and finally a tuple of labels corresponding to the states.
        """
        op_dict = {}
        unique_state_dict = {}
        for i, ket_0 in enumerate(self.basis_states):
            for j, ket_1 in enumerate(self.basis_states):
                if i == j:
                    label = self.label_list[i]
                    op_dict[label + label] = (
                        ket_0 * ket_0.dag(),
                        (1.0,),
                        (ket_0 * ket_0.dag(),),
                        ((label,),),
                    )
                    if (label,) not in unique_state_dict:
                        unique_state_dict[(label,)] = ket_0 * ket_0.dag()
                else:
                    # slight inefficiency rn is that |ij> + |kl> and |ij> + |kl> get recorded as different states
                    pl_state = (ket_0 + ket_1).unit()
                    min_state = (ket_0 + 1j * ket_1).unit()
                    new_states = (
                        pl_state * pl_state.dag(),
                        min_state * min_state.dag(),
                        ket_0 * ket_0.dag(),
                        ket_1 * ket_1.dag(),
                    )
                    alpha_coeffs = (1, 1j, -0.5 * (1 + 1j), -0.5 * (1 + 1j))
                    label_0 = self.label_list[i]
                    label_1 = self.label_list[j]
                    new_labels = (
                        (
                            label_0,
                            1,
                            label_1,
                        ),
                        (
                            label_0,
                            1j,
                            label_1,
                        ),
                        (label_0,),
                        (label_1,),
                    )
                    op_dict[label_0 + label_1] = (
                        ket_0 * ket_1.dag(),
                        alpha_coeffs,
                        new_states,
                        new_labels,
                    )
                    unique_state_dict.update(
                        {
                            new_labels[k]: state
                            for k, state in enumerate(new_states)
                            if k not in unique_state_dict
                        }
                    )
                    assert ket_0 * ket_1.dag() == sum(
                        [
                            alpha_coeffs[i] * new_states[i]
                            for i in range(len(alpha_coeffs))
                        ]
                    )
        return op_dict, unique_state_dict

    @staticmethod
    def operators_from_states(op_dict, unique_state_dict):
        """
        reconstruct operators from the decomposition as obtained in operator_basis_lidar
        Parameters
        ----------
        op_dict: dict
            dictionary of operators as defined in operator_basis_lidar
        unique_state_dict: dict
            dictionary of states as defined in operator_basis_lidar. The use case in mind is that the
            initial states have been mapped to final states, and we now want to reconstruct how the
            operators evolve
        Returns
        -------
        dict

        """
        return {
            key: sum(
                coeffs[idx] * unique_state_dict[label]
                for idx, label in enumerate(labels)
            )
            for key, (op, coeffs, rhos, labels) in op_dict.items()
        }

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
        op_dict, unique_state_dict = self.operator_basis_lidar()
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
