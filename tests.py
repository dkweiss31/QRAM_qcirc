from functools import partial

import numpy as np
from qutip import (
    destroy,
    Qobj,
    sigmax,
    operator_to_vector,
    vector_to_operator,
    tensor,
    to_super,
)

from QRAM_utils.quantum_helpers import (
    apply_gate_to_states,
    Fock_prods,
    SWAP_op,
    operators_from_states,
    operator_basis_lidar,
)
from QRAM_utils.utils import id_wrap_ops, project_U, construct_basis_states_list
from bosonic.simulate_bosonic_ops import SimulateBosonicOperations, SimulateBosonicOperationsDR

param_dict = {
    "tmon_dim": 3,
    "cavity_dim": 2,
    "control_dt": 4.0,
    "chi": 2.0 * np.pi * 0.002,
    "tmon_d_strength": 2.0 * np.pi * 0.01,
    "eta_gg": 0.9999,
    "eta_ge": 0.01,
    "eta_gf": 0.01**2,
    "Gamma_1_ge": 1.0 / (200 * 10**3),
    "Gamma_1_ef": 2.0 / (200 * 10**3),
    "Gamma_phi_gg": 0.0,
    "Gamma_phi_ee": 1.0 / (400 * 10**3),
    "Gamma_phi_ff": 4.0 / (400 * 10**3),
    "Gamma_1_res": 1.0 / (600 * 10**3),
    "Gamma_phi_res": 1.0 / (5000 * 10**3),
    "nth": 0.01,
    "postselection": True,
    "num_cpus": 8,
    "liouvillian": False,
}

# TODO add an explicit check that U_eJP correctly reproduces the analytical result


class TestSimulateBosonicOps:
    @classmethod
    def setup_class(cls):
        cls.sbo = SimulateBosonicOperations(
            gf_tmon=True, cavity_dim=2, tmon_dim=3, control_dt=4.0
        )
        cls.sbodr = SimulateBosonicOperationsDR(
            gf_tmon=True, cavity_dim=2, tmon_dim=3, control_dt=4.0
        )

    def test_SWAP(self):
        idx_0 = 0
        idx_1 = 1
        dims = [2, 2, 2]
        true_SWAP = Qobj(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dims=[dims, dims],
        )
        sbo_SWAP = SWAP_op(idx_0, idx_1, dims)
        assert true_SWAP == sbo_SWAP

    def test_Fock_prod(self):
        dim_list = [2, 3]
        test_Fock_prods = Fock_prods(dim_list)
        real_Fock_prods = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        assert test_Fock_prods == real_Fock_prods

    def test_R_tmon(self):
        ideal_R_tmon = (-0.5 * 1j * sigmax()).expm()
        R_tmon = self.sbo.R_tmon(1.0, 1.0, "X")
        Fock_states_spec = [(0, 0, 0), (0, 0, 2)]
        R_tmon_projected = project_U(R_tmon, Fock_states_spec, self.sbo.truncated_dims)
        assert R_tmon_projected == ideal_R_tmon

    def test_cZZU(self):
        cav_a_idx = 0
        cav_b_idx = 1
        cavity_fock_trunc = 2
        a = id_wrap_ops(
            destroy(self.sbo.cavity_dim), cav_a_idx, self.sbo.truncated_dims
        )
        b = id_wrap_ops(
            destroy(self.sbo.cavity_dim), cav_b_idx, self.sbo.truncated_dims
        )
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
        ideal_cZZU_projected = project_U(
            ideal_cZZU, Fock_states_spec, self.sbo.truncated_dims
        )
        assert ideal_cZZU_projected == cZZU_projected

    def test_main(self):
        tmon_dim = self.sbo.tmon_dim
        cavity_dim = self.sbo.cavity_dim
        chi = param_dict["chi"]
        tmon_d_strength = param_dict["tmon_d_strength"]
        params = (tmon_d_strength, chi)
        cav_a_idx = 0
        cav_b_idx = 1
        truncated_dims_SR = [cavity_dim, cavity_dim, tmon_dim]
        a = id_wrap_ops(destroy(cavity_dim), cav_a_idx, truncated_dims_SR)
        b = id_wrap_ops(destroy(cavity_dim), cav_b_idx, truncated_dims_SR)
        g_Fock_states_spec_SR = [(i, j, 0) for i in range(2) for j in range(2)]
        labels_SR = ["00", "01", "10", "11"]
        labels_DR = ["1100", "1001", "0110", "0011"]
        g_comp_basis_states_SR = construct_basis_states_list(
            g_Fock_states_spec_SR, truncated_dims_SR
        )
        # unitary evolution
        U_eJP_ideal_SR = self.sbo.U_eJP_func(a, b, params)
        for init_state in g_comp_basis_states_SR:
            final_state = self.sbo.U_eJP_func(a, b, params, state=init_state)
            assert (
                np.max(np.abs(final_state.data - (U_eJP_ideal_SR * init_state).data))
                < 1e-4
            )

        # open system evolution
        c_ops = self.sbo.construct_c_ops(a, b, **param_dict)
        U_eJP_SR = self.sbo.U_eJP_func(a, b, params, c_ops=c_ops)
        for init_state in g_comp_basis_states_SR:
            final_state = self.sbo.U_eJP_func(
                a, b, params, c_ops=c_ops, state=init_state * init_state.dag()
            )
            init_vec = operator_to_vector(init_state * init_state.dag())
            assert (
                np.max(
                    np.abs(
                        final_state.data - vector_to_operator(U_eJP_SR * init_vec).data
                    )
                )
                < 1e-4
            )
        # DR tests. want to compare the method of directly computing the superoperator
        # to that of combining individual SR final states
        g_comp_basis_states_DR = self.sbodr.DR_basis(g_comp_basis_states_SR)
        op_dict_DR, unique_state_dict_DR = operator_basis_lidar(
            g_comp_basis_states_DR, labels_DR
        )
        U_eJP_DR = tensor(U_eJP_SR, U_eJP_SR)
        V2 = self.sbodr.V_2_op()
        DR_dims = to_super(tensor(U_eJP_ideal_SR, U_eJP_ideal_SR)).dims
        final_DR_prop = Qobj(V2.dag().data @ U_eJP_DR.data @ V2.data, dims=DR_dims)
        # construct SR final states and ops
        op_dict_SR, unique_state_dict_SR = operator_basis_lidar(
            g_comp_basis_states_SR, labels_SR
        )
        U_eJP_partial = partial(self.sbo.U_eJP_func, a, b, params, c_ops)
        final_SR = apply_gate_to_states(U_eJP_partial, unique_state_dict_SR, 8)
        final_SR_ops = operators_from_states(op_dict_SR, final_SR)
        final_DR_states = {
            label: self.sbodr.DR_state_from_SR_ops(label, final_SR_ops)
            for label, state in unique_state_dict_DR.items()
        }
        for label, init_state in unique_state_dict_DR.items():
            final_state_prop = vector_to_operator(
                final_DR_prop * operator_to_vector(init_state)
            )
            assert (
                np.max(np.abs(final_state_prop.data - final_DR_states[label].data))
                < 1e-4
            )
