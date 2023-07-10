from itertools import product
from typing import List, Callable, Optional, Union

import numpy as np
from qutip import (
    Qobj,
    liouvillian,
    mesolve,
    Options,
    qeye,
    tensor,
)
from utils import get_map, construct_basis_states_list


def _propagator(H: Qobj, t: float, c_ops: Optional[List[Qobj]] = None) -> Qobj:
    """
    Parameters
    ----------
    H: Qobj
        Hamiltonian
    t: float
        time for which the Hamiltonian acts
    c_ops: List[Qobj]
        list of collapse operators

    Returns
    -------
    propagator corresponding to the Hamiltonian acting for a time :math:`t`.
    If c_ops are not passed, we assume the Hamiltonian is time
    independent and exponentiate it. If c_ops are passed, we construct the
    Liouvillian and exponentiate that instead.
    """
    if c_ops is None:
        return (-1j * H * t).expm()
    return (liouvillian(H, c_ops) * t).expm()


def apply_gate_to_states(gate: Callable, states_dict: dict, num_cpus: int = 1) -> dict:
    """
    apply the specified gate to the collection of states.
    Parameters
    ----------
    gate: Callable
        function taking states (density matrices) as input
    states_dict: dict
        dictionary of the states to apply the gate to
    num_cpus: int
        number of cpus to use (this is the costliest function)
    Returns
    -------
    dictionary of the final states with the same keys as states_dict
    """
    mapped_states = list(get_map(num_cpus)(gate, states_dict.values()))
    return dict(zip(states_dict.keys(), mapped_states))


def operator_basis_lidar(basis_states, label_list=None) -> (dict, dict):
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
    if label_list is None:
        label_list = range(len(basis_states))
    for i, ket_0 in enumerate(basis_states):
        for j, ket_1 in enumerate(basis_states):
            if i == j:
                label = label_list[i]
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
                label_0 = label_list[i]
                label_1 = label_list[j]
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
                    [alpha_coeffs[i] * new_states[i] for i in range(len(alpha_coeffs))]
                )
    return op_dict, unique_state_dict


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
            coeffs[idx] * unique_state_dict[label] for idx, label in enumerate(labels)
        )
        for key, (op, coeffs, rhos, labels) in op_dict.items()
    }


def SWAP_op(idx_0, idx_1, truncated_dims):
    """SWAP between two subsystems"""
    dim_0 = truncated_dims[idx_0]
    dim_1 = truncated_dims[idx_1]
    result = 0.0
    # running below as a product over map because dim_0 and dim_1
    # could themselves be lists of integers (think of V2 operation)
    all_Fock_prods = product(*map(Fock_prods, [dim_0, dim_1]))
    for Fock_prod in all_Fock_prods:
        id_list = [qeye(dim) for dim in truncated_dims]
        (Fock_0,) = construct_basis_states_list(
            [
                Fock_prod[0],
            ],
            dim_0,
        )
        (Fock_1,) = construct_basis_states_list(
            [
                Fock_prod[1],
            ],
            dim_1,
        )
        id_list[idx_0] = Fock_1 * Fock_0.dag()
        id_list[idx_1] = Fock_0 * Fock_1.dag()
        result += tensor(*id_list)
    return result


def Fock_prods(dim: Union[int, list[int]]) -> Union[list, range]:
    """Fock-state specification for all states as specified by dim"""
    if type(dim) == int:
        return range(dim)
    else:
        return list(product(*map(range, dim)))


def prop_or_mesolve_factory(
    H: Qobj, t: float, dt: float, c_ops: Optional[list[Qobj]], state: Optional[Qobj]
) -> Qobj:
    """
    Parameters
    ----------
    H: Qobj
        Hamiltonian
    t: float
        time for which the Hamiltonian acts
    dt: float
        dt for mesolve
    c_ops: list[Qobj]
        collapse operators
    state: Qobj
        state to time evolve. if None, return the propogator
    Returns
    -------
        Qobj of propogator or final state
    """
    if state is None:
        return _propagator(H, t, c_ops=c_ops)
    else:
        tlist = np.linspace(0.0, t, int(t / dt))
        result = mesolve(
            H, state, tlist, c_ops=c_ops, options=Options(store_final_state=True)
        )
        return result.final_state
