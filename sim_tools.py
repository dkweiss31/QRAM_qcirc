from itertools import product
from typing import List

from qutip import (
    tensor,
    basis,
    Qobj,
    operator_to_vector,
    vector_to_operator,
    sigmaz,
    destroy,
    liouvillian,
    to_super, qeye,
)
import numpy as np
from utils import id_wrap_ops, project_U, construct_basis_states_list


class SimulateBosonicOperations:
    """
    """
    def __init__(self, gf_tmon=True, tmon_dim=3, cavity_dim=3):
        self.gf_tmon = gf_tmon
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim
        self.truncated_dims = [cavity_dim, cavity_dim, tmon_dim]
        # below we define the s_ops for the transmon
        self.sx = None
        self.sy = None
        self.sz = None
        self.s_gg = None
        self.s_ee = None
        self.s_ff = None
        self.sminus_ge = None
        self.sminus_ef = None
        if self.gf_tmon:
            self.s_ops_gi_tmon(2)
        else:
            self.s_ops_gi_tmon(1)

    def s_ops_gi_tmon(self, i):
        """construct gi tmon where i can be e or f"""
        sx_gi = (basis(self.tmon_dim, 0) * basis(self.tmon_dim, i).dag()
                 + basis(self.tmon_dim, i) * basis(self.tmon_dim, 0).dag())
        sy_gi = (-1j * basis(self.tmon_dim, 0) * basis(self.tmon_dim, i).dag()
                 + 1j * basis(self.tmon_dim, i) * basis(self.tmon_dim, 0).dag())
        sz_gi = (basis(self.tmon_dim, 0) * basis(self.tmon_dim, 0).dag()
                 - basis(self.tmon_dim, i) * basis(self.tmon_dim, i).dag())
        # below assumes that tmon is in position 2 (zero indexed) in truncated_dims
        tmon_idx = 2
        self.sx = id_wrap_ops(sx_gi, tmon_idx, self.truncated_dims)
        self.sy = id_wrap_ops(sy_gi, tmon_idx, self.truncated_dims)
        self.sz = id_wrap_ops(sz_gi, tmon_idx, self.truncated_dims)
        # define below ops for collapse operators
        self.sminus_ge = id_wrap_ops(basis(3, 0) * basis(3, 1).dag(), tmon_idx, self.truncated_dims)
        self.s_gg = id_wrap_ops(basis(3, 0) * basis(3, 0).dag(), tmon_idx, self.truncated_dims)
        self.s_ee = id_wrap_ops(basis(3, 1) * basis(3, 1).dag(), tmon_idx, self.truncated_dims)
        if i == 2:
            self.sminus_ef = id_wrap_ops(basis(3, 1) * basis(3, 2).dag(), tmon_idx, self.truncated_dims)
            self.s_ff = id_wrap_ops(basis(3, 2) * basis(3, 2).dag(), tmon_idx, self.truncated_dims)

    def construct_c_ops(self, a: Qobj, b: Qobj, Gamma_1_ge=0.0, Gamma_1_ef=0.0, Gamma_phi_gg=0.0,
                        Gamma_phi_ee=0.0, Gamma_phi_ff=0.0, Gamma_1_res=0.0,
                        Gamma_phi_res=0.0, nth=0.0, **kwargs):
        c_ops = [np.sqrt(Gamma_1_ge) * self.sminus_ge,
                 np.sqrt(Gamma_1_ef) * self.sminus_ef,
                 np.sqrt(nth * Gamma_1_ge) * self.sminus_ge.dag(),
                 np.sqrt(nth * Gamma_1_ef) * self.sminus_ef.dag(),
                 np.sqrt(Gamma_phi_gg) * self.s_gg,
                 np.sqrt(Gamma_phi_ee) * self.s_ee,
                 np.sqrt(Gamma_phi_ff) * self.s_ff,
                 np.sqrt(Gamma_1_res) * a,
                 np.sqrt(Gamma_1_res) * b,
                 np.sqrt(nth * Gamma_1_res) * a.dag(),
                 np.sqrt(nth * Gamma_1_res) * b.dag(),
                 np.sqrt(Gamma_phi_res) * a.dag() * a,
                 np.sqrt(Gamma_phi_res) * b.dag() * b,
                 ]
        return c_ops

    def cZZU(self, a_op: Qobj, b_op: Qobj, chi: float, c_ops: List[Qobj] = None):
        """
        Parameters
        ----------
        a_op: Qobj
            lowering operator for the first bosonic mode, coupled to the transmon
        b_op: Qobj
            lowering operator for the second bosonic mode
        c_ops: List[Qobj]
            collapse operators

        Returns
        -------
        propagator corresponding to the cZZ unitary as described in
        Tsunoda et al. arXiv:2212.11196 (2022) and Teoh et al. arXiv:2212.12077 (2022)
        """
        g = np.sqrt(3) * chi / 2
        H = -0.5 * chi * self.sz * a_op.dag() * a_op + 0.5 * g * (
            a_op.dag() * b_op + a_op * b_op.dag()
        )
        Omega = np.sqrt(g**2 + (chi / 2) ** 2)
        t = 2.0 * np.pi / Omega
        return self._propagator(H, t, c_ops=c_ops)

    @staticmethod
    def R_osc(a_op: Qobj, phi: float, c_ops: List[Qobj] = None):
        """
        this gate is done in software and thus can be done with unit fidelity
        (hence why the collapse operators are not used even in the case when they are passed)
        Parameters
        ----------
        a_op: Qobj
            lowering operator of the mode we want to apply a
            phase correction to
        phi: float
            parameter describing the phase correction
        c_ops: List[Qobj]
            unused collapse operators
        Returns
        -------
        propagator corresponding to a single-cavity rotation
        """
        cav_rotation = (-1j * phi * a_op.dag() * a_op).expm()
        if c_ops is None:
            return cav_rotation
        return to_super(cav_rotation)

    def R_tmon(self, g: float, t: float, direction: str = "X", c_ops: List[Qobj] = None):
        r"""
        Parameters
        ----------
        g: float
            drive strength: the Hamiltonian is :math:`H = 0.5 * g * \sigma_{i}, i\in\{x,y,z\}`
        t: float
            time for which the interaction should act
        direction: str
            either "X", "Y", or "Z" depending on the rotation direction
        c_ops: List[Qobj]
            collapse operators

        Returns
        -------
        propagator corresponding to a transmon rotation
        """
        if direction == "X":
            s_op = self.sx
        elif direction == "Y":
            s_op = self.sy
        elif direction == "Z":
            s_op = self.sz
        else:
            raise RuntimeError("specified direction must be 'X', 'Y', or 'Z'")
        return self._propagator(0.5 * g * s_op, t, c_ops=c_ops)

    @staticmethod
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

    def beamsplitter(self, a_op: Qobj, b_op: Qobj, g: float, t: float, c_ops=None):
        H = 0.5 * g * a_op.dag() * b_op + 0.5 * np.conjugate(g) * b_op.dag() * a_op
        return self._propagator(H, t, c_ops=c_ops)

    def cZU(self, a_op: Qobj, chi: float, c_ops=None):
        H = 0.5 * chi * self.sz * a_op.dag() * a_op
        t = np.pi / chi
        return self._propagator(H, t, c_ops=c_ops)

    def cZZZU(self, a_op: Qobj, b_op: Qobj, c_op: Qobj, chi: float, c_ops=None):
        U1 = self.cZZU(b_op, a_op, chi, c_ops=c_ops)
        U2 = self.cZZU(b_op, c_op, chi, c_ops=c_ops)
        U3 = self.cZU(b_op, chi, c_ops=c_ops)
        return U1 * U2 * U3

    @staticmethod
    def SWAP(a_op: Qobj, b_op: Qobj):
        return ((np.pi / 2) * (a_op.dag() * b_op - a_op * b_op.dag())).expm()

    def cZZ_time(self, chi):
        g = np.sqrt(3) * chi / 2
        Omega = np.sqrt(g**2 + (chi / 2) ** 2)
        return 2.0 * np.pi / Omega

    def U_eJP_func(self, a_op: Qobj, b_op: Qobj, params, c_ops=None):
        (tmon_d_strength, chi) = params
        tmon_d_time = np.pi / (2 * tmon_d_strength)
        JPP = self.cZZU(a_op, b_op, chi, c_ops=c_ops)
        U_tmon_y = self.R_tmon(tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops)
        # don't want to use dagger here as it can be a superoperator (need to understand better)
        U_tmon_y_min = self.R_tmon(
            -tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops
        )
        U_tmon_x = self.R_tmon(tmon_d_strength, tmon_d_time, direction="X", c_ops=c_ops)
        U_a = self.R_osc(a_op, -np.pi / 2, c_ops=c_ops)
        U_b = self.R_osc(b_op, -np.pi / 2, c_ops=c_ops)
        return U_a * U_b * U_tmon_y_min * JPP * U_tmon_x * JPP * U_tmon_y

    def U_erasure_check(self, a_op: Qobj, b_op: Qobj, params, c_ops=None):
        (tmon_d_strength, chi) = params
        tmon_d_time = np.pi / (2 * tmon_d_strength)
        JPP = self.cZZU(a_op, b_op, chi, c_ops=c_ops)
        U_tmon_y = self.R_tmon(tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops)
        U_a = self.R_osc(a_op, np.pi / 2, c_ops=c_ops)
        U_b = self.R_osc(b_op, np.pi / 2, c_ops=c_ops)
        return U_tmon_y * U_a * U_b * JPP * U_tmon_y

    def measurement_op_tmon_projector(self, idx):
        """project onto a specific tmon eigenstate"""
        Fock_states_spec = [(i, j, idx) for i in range(self.cavity_dim)
                            for j in range(self.cavity_dim)]
        Fock_states = construct_basis_states_list(Fock_states_spec, self.truncated_dims)
        return sum([Fock_state * Fock_state.dag() for Fock_state in Fock_states])

    def measurement_op_DR_parity(self):
        measurement_op = 0.0
        for idx in range(self.tmon_dim):
            Fock_states_spec = [(i, j, idx) for i in range(2)
                                for j in range(2)]
            Fock_states = construct_basis_states_list(Fock_states_spec, self.truncated_dims)
            Fock_states_DR = self.DR_basis(Fock_states)
            measurement_op += sum([detected_state * detected_state.dag()
                                   for detected_state in Fock_states_DR])
        return measurement_op

    @staticmethod
    def _Fock_prods(dim):
        if type(dim) == int:
            return range(dim)
        else:
            return list(product(*map(range, dim)))

    def SWAP_op(self, idx_0, idx_1, dims=None):
        """SWAP between two subsystems"""
        if dims is None:
            dims = self.truncated_dims
        dim_0 = dims[idx_0]
        dim_1 = dims[idx_1]
        Fock_prods = product(*map(self._Fock_prods, [dim_0, dim_1]))
        result = 0.0
        for Fock_prod in Fock_prods:
            id_list = [qeye(dim) for dim in dims]
            (Fock_0,) = construct_basis_states_list([Fock_prod[0], ], dim_0)
            (Fock_1,) = construct_basis_states_list([Fock_prod[1], ], dim_1)
            id_list[idx_0] = Fock_1 * Fock_0.dag()
            id_list[idx_1] = Fock_0 * Fock_1.dag()
            result += tensor(*id_list)
        return result

    def V_2_op(self):
        """see https://arxiv.org/abs/1111.6950 , operator that SWAPs the internal indices
        to properly order a tensor product of superoperators"""
        return self.SWAP_op(1, 2, dims=4 * [self.truncated_dims])

    @staticmethod
    def DR_basis(SR_comp_bas_states):
        # express logical DR states in terms of the basis states of the cavities
        # basis states originally in |router, input, tmon=0>
        # ordered as router, input, router, input
        # |00>_{L} = |10>_{r}|10>_{i} --> |1>_{r}|1>_{i}|0>_{r}|0>_{i}
        # |01>_{L} = |10>|01> --> |1>|0>|0>|1>
        # |10>_{L} = |01>|10> --> |0>|1>|1>|0>
        # |11>_{L} = |01>|01> --> |0>|0>|1>|1>
        basis_state_DR = [tensor(SR_comp_bas_states[3], SR_comp_bas_states[0]),
                          tensor(SR_comp_bas_states[2], SR_comp_bas_states[1]),
                          tensor(SR_comp_bas_states[1], SR_comp_bas_states[2]),
                          tensor(SR_comp_bas_states[0], SR_comp_bas_states[3])]
        return basis_state_DR

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
        ideal_cZZU_projected = project_U(
            ideal_cZZU, Fock_states_spec, truncated_dims
        )
        assert ideal_cZZU_projected == cZZU_projected


class FidelityBosonicOperations:
    def __init__(self, comp_basis_states):
        self.comp_basis_states = comp_basis_states

    @staticmethod
    def _operator_basis_lidar(ket_0, ket_1):
        pl_state = (ket_0 + ket_1).unit()
        min_state = (ket_0 + 1j * ket_1).unit()
        return ((1, 1j, -0.5 * (1 + 1j), -0.5 * (1 + 1j)),
                (pl_state * pl_state.dag(), min_state * min_state.dag(),
                ket_0 * ket_0.dag(), ket_1 * ket_1.dag()))

    def operator_basis_lidar(self, basis_states=None):
        if basis_states is None:
            basis_states = self.comp_basis_states
        op_basis, alpha_list, state_list = [], [], []
        for i, ket_0 in enumerate(basis_states):
            for j, ket_1 in enumerate(basis_states):
                if i == j:
                    alpha_list.append((1.0,))
                    op_basis.append(ket_0 * ket_0.dag())
                    state_list.append((ket_0 * ket_0.dag(),))
                else:
                    op_basis.append(ket_0 * ket_1.dag())
                    alpha_coeffs, states = self._operator_basis_lidar(ket_0, ket_1)
                    alpha_list.append(alpha_coeffs)
                    state_list.append(states)
                    assert ket_0 * ket_1.dag() == sum([alpha_coeffs[i] * states[i] for i in range(len(alpha_coeffs))])
        return alpha_list, state_list, op_basis

    @staticmethod
    def measurement_channel(rho, measurement_op):
        if measurement_op.type == "oper":
            new_rho = measurement_op * rho * measurement_op.dag()
            return new_rho, np.trace(new_rho)
        elif measurement_op.type == "super":
            new_rho = measurement_op * rho
            return new_rho, np.trace(vector_to_operator(new_rho))
        else:
            raise RuntimeError(
                'measurement_op should be either of type "oper" or "super"'
            )

    @staticmethod
    def process_fidelity_nielsen(entanglement_fidelity, num_qubits=2):
        dim = num_qubits ** 2
        return (dim * entanglement_fidelity + 1) / (dim + 1)

    def entanglement_fidelity_nielsen(
            self,
            U_real,
            U_ideal,
            basis_states,
            measurement_op=None,
            ptrace_idxs=None,
            num_qubits=2,
    ):
        dim = 2 ** num_qubits
        alpha_list, state_list, op_basis = self.operator_basis_lidar(basis_states=basis_states)
        overall_contr = 0.0
        total_prob = 0.0
        num_states = 0
        for j, op in enumerate(op_basis):
            for k, (coeff, pauli_rho) in enumerate(zip(alpha_list[j], state_list[j])):
                rho = operator_to_vector(pauli_rho)
                propagated_rho = U_real * rho
                if measurement_op is not None:
                    propagated_rho, prob = self.measurement_channel(propagated_rho, measurement_op)
                else:
                    prob = 0.0
                propagated_rho = vector_to_operator(propagated_rho)
                total_prob += prob
                if ptrace_idxs is not None:
                    propagated_rho = propagated_rho.ptrace(ptrace_idxs)
                    op = op.ptrace(ptrace_idxs)
                projected_rho = project_U(
                    propagated_rho, basis_states=basis_states
                )
                projected_op = project_U(op, basis_states=basis_states)
                state_contr = coeff * np.trace(
                    U_ideal * projected_op.dag() * U_ideal.dag() * projected_rho
                )
                overall_contr += state_contr
                num_states += 1
        #       Nielsen formula for an orthogonal basis that obeys tr(U_{j}^dag U_{k}) = delta_{jk}
        #       (as opposed to dim delta_{jk} has one less factor of dim in the denominator
        return overall_contr / dim ** 2, total_prob / num_states
