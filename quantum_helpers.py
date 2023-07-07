from itertools import product
from typing import List, Callable, Optional, Union

import numpy as np
from qutip import (
    Qobj,
    liouvillian,
    mesolve,
    Options, qeye, tensor,
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
