from typing import List

from qutip import (
    tensor,
    Qobj,
    operator_to_vector,
    vector_to_operator,
    sigmaz,
    destroy,
    liouvillian,
    to_super,
)
import numpy as np
from utils import id_wrap_ops, project_U


class SimulateBosonicOperations:
    """
    """
    def __init__(self, sx, sy, sz, chi, tmon_dim, cavity_dim):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.chi = chi
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim

    def cZZU(self, a_op: Qobj, b_op: Qobj, c_ops: List[Qobj] = None):
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
        g = np.sqrt(3) * self.chi / 2
        H = -0.5 * self.chi * self.sz * a_op.dag() * a_op + 0.5 * g * (
            a_op.dag() * b_op + a_op * b_op.dag()
        )
        Omega = np.sqrt(g**2 + (self.chi / 2) ** 2)
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

    def cZU(self, a_op: Qobj, c_ops=None):
        H = 0.5 * self.chi * self.sz * a_op.dag() * a_op
        t = np.pi / self.chi
        return self._propagator(H, t, c_ops=c_ops)

    def cZZZU(self, a_op: Qobj, b_op: Qobj, c_op: Qobj, c_ops=None):
        U1 = self.cZZU(b_op, a_op, c_ops=c_ops)
        U2 = self.cZZU(b_op, c_op, c_ops=c_ops)
        U3 = self.cZU(b_op, c_ops=c_ops)
        return U1 * U2 * U3

    @staticmethod
    def SWAP(a_op: Qobj, b_op: Qobj):
        return ((np.pi / 2) * (a_op.dag() * b_op - a_op * b_op.dag())).expm()

    def cZZ_time(self):
        g = np.sqrt(3) * self.chi / 2
        Omega = np.sqrt(g**2 + (self.chi / 2) ** 2)
        return 2.0 * np.pi / Omega

    def U_eJP_func(self, a_op: Qobj, b_op: Qobj, params, c_ops=None):
        (tmon_d_strength, _) = params
        tmon_d_time = np.pi / (2 * tmon_d_strength)
        JPP = self.cZZU(a_op, b_op, c_ops=c_ops)
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
        (tmon_d_strength, _) = params
        tmon_d_time = np.pi / (2 * tmon_d_strength)
        JPP = self.cZZU(a_op, b_op, c_ops=c_ops)
        U_tmon_y = self.R_tmon(tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops)
        U_a = self.R_osc(a_op, np.pi / 2, c_ops=c_ops)
        U_b = self.R_osc(b_op, np.pi / 2, c_ops=c_ops)
        return U_tmon_y * U_a * U_b * JPP * U_tmon_y

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
        Fock_states_spec = [
            (i, j, k)
            for i in range(cavity_fock_trunc)
            for j in range(cavity_fock_trunc)
            for k in range(2)
        ]
        cZZU_projected = project_U(
            self.cZZU(a, b), Fock_states_spec, truncated_dims
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
