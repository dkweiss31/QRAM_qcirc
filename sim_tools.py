from qutip import (
    basis,
    tensor,
    Qobj,
    qeye,
    operator_to_vector,
    vector_to_operator,
    sigmax,
    to_kraus,
    sigmay,
    sigmaz,
    destroy,
    liouvillian,
    to_super,
)
import numpy as np
from utils import id_wrap_ops, construct_basis_states_list, project_U


class SimulateBosonicOperations:
    def __init__(self, sx, sy, sz, chi, comp_basis_states, tmon_dim, cavity_dim):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.chi = chi
        self.comp_basis_states = comp_basis_states
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim

    def cZZU(self, a_op: Qobj, b_op: Qobj, c_ops=None):
        g = np.sqrt(3) * self.chi / 2
        H = -0.5 * self.chi * self.sz * a_op.dag() * a_op + 0.5 * g * (
            a_op.dag() * b_op + a_op * b_op.dag()
        )
        Omega = np.sqrt(g**2 + (self.chi / 2) ** 2)
        t = 2.0 * np.pi / Omega
        return self._propagator(H, t, c_ops=c_ops)

    def R_osc(self, a_op: Qobj, phi: float, c_ops=None):
        """this gate is done in software and thus can be done with unit fidelity
        (hence why the collapse operators are not used even in the case when they are passed)"""
        cav_rotation = (-1j * phi * a_op.dag() * a_op).expm()
        if c_ops is None:
            return cav_rotation
        return to_super(cav_rotation)

    def R_tmon(self, g: float, t: float, direction="X", c_ops=None):
        if direction == "X":
            s_op = self.sx
        elif direction == "Y":
            s_op = self.sy
        elif direction == "Z":
            s_op == self.sz
        else:
            raise RuntimeError("specified direction must be 'X', 'Y', or 'Z'")
        return self._propagator(0.5 * g * s_op, t, c_ops=c_ops)

    def _propagator(self, H: Qobj, t: float, c_ops=None):
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

    def SWAP(self, a_op: Qobj, b_op: Qobj):
        return ((np.pi / 2) * (a_op.dag() * b_op - a_op * b_op.dag())).expm()

    def logical_error_prob(
        self, U_diss, initial_Fock_states_spec, bad_Fock_states_spec, truncated_dims
    ):
        """
        Parameters
        ----------
        U_diss: superoperator propagator
        initial_Fock_states_spec:
        bad_Fock_states_spec
        truncated_dims

        Returns
        -------

        """
        initial_basis_list = construct_basis_states_list(
            initial_Fock_states_spec, truncated_dims
        )
        num_initial_states = len(initial_Fock_states_spec)
        bad_basis_list = construct_basis_states_list(
            bad_Fock_states_spec, truncated_dims
        )
        bad_pop_total = 0.0
        for i, initial_state in enumerate(initial_basis_list):
            final_state_vec = U_diss * operator_to_vector(
                initial_state * initial_state.dag()
            )
            final_dm = vector_to_operator(final_state_vec)
            # assume below that most of the population is in the
            # correct final state. This corrects for the sum below
            # which will include the correct final state as a "bad_vec"
            correct_final_state_pop = np.real(np.max(np.diag(final_dm)))
            for j, bad_state in enumerate(bad_basis_list):
                # I think the below is the right way to extract the diagonal element
                # (that is the probability) but need to think more
                bad_pop = bad_state.dag() * final_dm * bad_state
                bad_pop_total += np.real(bad_pop.data.toarray()[0, 0])
            bad_pop_total = bad_pop_total - correct_final_state_pop
        return bad_pop_total / num_initial_states

    def gate_failure_prob(
        self,
        U_diss,
        initial_Fock_states_spec,
        detected_Fock_states_spec,
        truncated_dims,
    ):
        initial_basis_list = construct_basis_states_list(
            initial_Fock_states_spec, truncated_dims
        )
        num_initial_states = len(initial_Fock_states_spec)
        detected_basis_list = construct_basis_states_list(
            detected_Fock_states_spec, truncated_dims
        )
        detected_pop_total = 0.0
        for i, initial_state in enumerate(initial_basis_list):
            final_state_vec = U_diss * operator_to_vector(
                initial_state * initial_state.dag()
            )
            final_dm = vector_to_operator(final_state_vec)
            for j, detected_state in enumerate(detected_basis_list):
                detected_pop = detected_state.dag() * final_dm * detected_state
                #            print(detected_Fock_states_spec[j], detected_pop)
                detected_pop_total += np.real(detected_pop.data.toarray()[0, 0])
        return detected_pop_total / num_initial_states

    def cZZ_time(self, params):
        (_, g) = params
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

    def epsilon_undid(self, U_eJP, U_eJP_ideal, initial_state):
        kraus_list = to_kraus(to_super(U_eJP_ideal.dag()) * U_eJP)
        return (
            1
            - initial_state.dag() * kraus_list[0].dag() * kraus_list[0] * initial_state
        )

    def kraus_list_measure(self, U_diss, measurement_op, initial_rho):
        new_rho_vec = U_diss * operator_to_vector(initial_rho)
        prob = np.trace(vector_to_operator(to_super(measurement_op) * new_rho_vec))
        op_undid_measure = to_super(measurement_op) * U_diss / np.sqrt(prob)
        return to_kraus(op_undid_measure)

    def measurement_channel(self, rho, measurement_op):
        if measurement_op.type == "oper":
            new_rho = measurement_op * rho * measurement_op.dag()
            prob = np.trace(new_rho)
            return new_rho / prob
        elif measurement_op.type == "super":
            new_rho = measurement_op * rho
            prob = np.trace(vector_to_operator(new_rho))
            return new_rho / prob
        else:
            raise RuntimeError(
                'measurement_op should be either of type "oper" or "super"'
            )

    def apply_channel(self, rho, U_real, measurement_op=None):
        assert U_real.type == "super"
        new_rho = U_real * rho
        if measurement_op is None:
            return new_rho
        else:
            return self.measurement_channel(new_rho, measurement_op)

    def fidelity_kraus(self, superop, ideal_op, keep_idxs):
        dim = len(keep_idxs)
        kraus_ops = to_kraus(superop)
        kraus_m_ops = np.array([ideal_op.dag() * kraus_op for kraus_op in kraus_ops])
        return (1.0 / (dim * (dim + 1))) * (
            np.trace(np.sum([kraus_op.dag() * kraus_op for kraus_op in kraus_m_ops]))
            + np.sum([np.abs(np.trace(kraus_op)) ** 2 for kraus_op in kraus_m_ops])
        )

    def failure_rate(self, U_real):
        comp_basis_states, superpos_states = self.state_basis()
        all_init_states = comp_basis_states + superpos_states
        failure_rate = 0.0
        measurement_op = tensor(
            qeye(self.cavity_dim),
            qeye(self.cavity_dim),
            basis(self.tmon_dim, 0) * basis(self.tmon_dim, 0).dag(),
        )
        for idx, state in enumerate(all_init_states):
            rho = operator_to_vector(state * state.dag())
            propagated_rho = vector_to_operator(
                self.apply_channel(rho, U_real, measurement_op=None)
            )
            failure_rate += 1 - np.trace(measurement_op * propagated_rho)
        return failure_rate / len(all_init_states)

    def fidelity_nielsen(
        self,
        U_real,
        U_ideal,
        Fock_states_spec,
        truncated_dims,
        s_ops_cavs,
        measurement_op=None,
        ptrace_idxs=None,
        num_qubits=2,
    ):
        dim = 2**num_qubits
        op_basis, pauli_keys = self.operator_basis(s_ops_cavs)
        st_contr = 0.0
        for j, op in enumerate(op_basis):
            alpha_state_pair = self.decompose_op_into_state_basis(
                op, pauli_keys[j]
            )
            for k, (coeff, pauli_rho) in enumerate(alpha_state_pair):
                rho = operator_to_vector(pauli_rho)
                propagated_rho = vector_to_operator(
                    self.apply_channel(rho, U_real, measurement_op=measurement_op)
                )
                if ptrace_idxs is not None:
                    propagated_rho = propagated_rho.ptrace(ptrace_idxs)
                    op = op.ptrace(ptrace_idxs)
                projected_rho = project_U(
                    propagated_rho, Fock_states_spec, truncated_dims
                )
                projected_op = project_U(op, Fock_states_spec, truncated_dims)
                st_contr += coeff * np.trace(
                    U_ideal * projected_op.dag() * U_ideal.dag() * projected_rho
                )
        return (st_contr + dim**2) / (dim**2 * (dim + 1))

    def state_basis(self):
        superpos_states = []
        for i, comp_bas_state_1 in enumerate(self.comp_basis_states):
            for j, comp_bas_state_2 in enumerate(self.comp_basis_states):
                if j > i:
                    superpos_states.append((comp_bas_state_1 + comp_bas_state_2).unit())
                    superpos_states.append((comp_bas_state_1 - comp_bas_state_2).unit())
                    superpos_states.append(
                        (comp_bas_state_1 + 1j * comp_bas_state_2).unit()
                    )
                    superpos_states.append(
                        (comp_bas_state_1 - 1j * comp_bas_state_2).unit()
                    )
        return self.comp_basis_states, superpos_states

    def decompose_op_into_state_basis(self, op, pauli_key):
        comp_basis_states, superpos_basis_states = self.state_basis()
        if "X" not in pauli_key and "Y" not in pauli_key:
            basis_states = comp_basis_states
        else:
            basis_states = comp_basis_states + superpos_basis_states
        st_rho = [state * state.dag() for state in basis_states]
        alpha_coeffs = np.array([np.trace(rho * op) for rho in st_rho])
        assert sum([alpha_coeffs[i] * rho for i, rho in enumerate(st_rho)]) == op
        alpha_state_pair = [
            (alpha_coeffs[i], state)
            for i, state in enumerate(st_rho)
            if alpha_coeffs[i] != 0
        ]
        return alpha_state_pair

    def operator_basis(self, s_ops_cavs):
        id_cav, sx_cav, sy_cav, sz_cav = s_ops_cavs
        ops = {"I": id_cav, "X": sx_cav, "Y": sy_cav, "Z": sz_cav}
        pauli_ops = []
        pauli_keys = []
        for key_a in ops.keys():
            for key_b in ops.keys():
                # start with the tmon in the ground state
                pauli_ops.append(
                    tensor(
                        ops[key_a],
                        ops[key_b],
                        basis(self.tmon_dim, 0) * basis(self.tmon_dim, 0).dag(),
                    )
                )
                pauli_keys.append(key_a + key_b)
        return pauli_ops, pauli_keys

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
        chi = 2.0 * np.pi * 0.001
        Fock_states_spec = [
            (i, j, k)
            for i in range(cavity_fock_trunc)
            for j in range(cavity_fock_trunc)
            for k in range(2)
        ]
        cZZU_projected = project_U(
            self.cZZU(chi, a, b, sz), Fock_states_spec, truncated_dims
        )
        ideal_cZZU = (-1j * (np.pi / 2) * sz * (a.dag() * a + b.dag() * b)).expm()
        ideal_cZZU_projected = project_U(
            ideal_cZZU, Fock_states_spec, truncated_dims
        )
        assert ideal_cZZU_projected == cZZU_projected
