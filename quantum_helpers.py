from itertools import product
from typing import List, Callable

import numpy as np
from qutip import (
    Qobj,
    liouvillian, mesolve, Options,
)
from utils import get_map


def _propagator(H: Qobj, t: float, c_ops: List[Qobj] = None):
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


def apply_gate_to_states(gate: Callable, states_dict: dict, num_cpus: int = 1):
    """
    apply the specified gate to the collection of states. I'm worried about ordering issues
    which is why I turn them into lists before applying the gate and then turning the collection of states back
    into a dict, but maybe with python 3.7+ I shouldn't worry?!?
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
    labels_list, states_list = [], []
    for (label, state) in states_dict.items():
        labels_list.append(label)
        states_list.append(state)
    target_map = get_map(num_cpus)
    # only want to apply the costly function to unique states. below combine the
    # results as appropriate for the propagated states
    mapped_states = list(target_map(gate, states_list))
    return dict(zip(labels_list, mapped_states))


def _Fock_prods(dim):
    if type(dim) == int:
        return range(dim)
    else:
        return list(product(*map(range, dim)))


def _prop_or_mesolve_factory(H, t, dt, c_ops, state):
    if state is None:
        return _propagator(H, t, c_ops=c_ops)
    else:
        tlist = np.linspace(0.0, t, int(t / dt))
        result = mesolve(
            H, state, tlist, c_ops=c_ops, options=Options(store_final_state=True)
        )
        return result.final_state
