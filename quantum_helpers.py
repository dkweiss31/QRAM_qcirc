from itertools import product
from typing import List, Callable, Optional

import numpy as np
from qutip import (
    Qobj,
    liouvillian,
    mesolve,
    Options,
)
from utils import get_map


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


def _Fock_prods(dim) -> list:
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
