import numpy as np
from qutip import destroy, sigmaz

from utils import id_wrap_ops, project_U


def test_cZZU(self):
    tmon_dim = 2
    cavity_dim = 3
    cav_a_idx = 0
    cav_b_idx = 1
    tmon_idx = 2
    truncated_dims = [cavity_dim, cavity_dim, tmon_dim]
    cavity_fock_trunc = 2
    a = id_wrap_ops(destroy(cavity_dim), cav_a_idx, truncated_dims)
    b = id_wrap_ops(destroy(cavity_dim), cav_b_idx, truncated_dims)
    sz = id_wrap_ops(sigmaz(), tmon_idx, truncated_dims)
    chi = 2.0 * np.pi * 0.002
    Fock_states_spec = [
        (i, j, k)
        for i in range(cavity_fock_trunc)
        for j in range(cavity_fock_trunc)
        for k in range(2)
    ]
    cZZU_projected = project_U(
        self.cZZU(a, b, chi), Fock_states_spec, truncated_dims
    )
    ideal_cZZU = (-1j * (np.pi / 2) * sz * (a.dag() * a + b.dag() * b)).expm()
    ideal_cZZU_projected = project_U(ideal_cZZU, Fock_states_spec, truncated_dims)
    assert ideal_cZZU_projected == cZZU_projected