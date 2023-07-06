import numpy as np
from qutip import destroy, sigmaz, Qobj, sigmax

from simulate_bosonic_ops import SimulateBosonicOperations
from utils import id_wrap_ops, project_U


class TestSimulateBosonicOps:
    @classmethod
    def setup_class(cls):
        cls.sbo = SimulateBosonicOperations(gf_tmon=True, cavity_dim=4, tmon_dim=3)

    def test_SWAP(self):
        idx_0 = 0
        idx_1 = 1
        dims = [2, 2, 2]
        true_SWAP = Qobj([[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]], dims=[dims, dims])
        sbo_SWAP = self.sbo.SWAP_op(idx_0, idx_1, dims)
        assert true_SWAP == sbo_SWAP

    def test_R_tmon(self):
        ideal_R_tmon = (-0.5 * 1j * sigmax()).expm()
        R_tmon = self.sbo.R_tmon(1.0, 1.0, "X")
        Fock_states_spec = [(0, 0, 0), (0, 0, 2)]
        R_tmon_projected = project_U(R_tmon, Fock_states_spec, self.sbo.truncated_dims)
        assert R_tmon_projected == ideal_R_tmon

    def test_cZZU(self):
        cav_a_idx = 0
        cav_b_idx = 1
        tmon_idx = 2
        cavity_fock_trunc = 2
        a = id_wrap_ops(destroy(self.sbo.cavity_dim), cav_a_idx, self.sbo.truncated_dims)
        b = id_wrap_ops(destroy(self.sbo.cavity_dim), cav_b_idx, self.sbo.truncated_dims)
        sz = self.sbo.sz
        chi = 2.0 * np.pi * 0.002
        Fock_states_spec = [
            (i, j, k)
            for i in range(cavity_fock_trunc)
            for j in range(cavity_fock_trunc)
            for k in (0, 2)
        ]
        cZZU_projected = project_U(
            self.sbo.cZZU(a, b, chi), Fock_states_spec, self.sbo.truncated_dims
        )
        ideal_cZZU = (-1j * (np.pi / 2) * sz * (a.dag() * a + b.dag() * b)).expm()
        ideal_cZZU_projected = project_U(ideal_cZZU, Fock_states_spec, self.sbo.truncated_dims)
        assert ideal_cZZU_projected == cZZU_projected
