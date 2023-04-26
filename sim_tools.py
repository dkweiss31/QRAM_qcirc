from qutip import basis, tensor, Qobj, qeye, operator_to_vector, vector_to_operator, sigmax, to_kraus, \
    sigmay, sigmaz, destroy, liouvillian, to_super
import numpy as np
from qutip.qip.operations import hadamard_transform


def cZZU(a_op: Qobj, b_op: Qobj, sz: Qobj, chi, c_ops=None):
    g = np.sqrt(3) * chi / 2
    H = -0.5 * chi * sz * a_op.dag() * a_op + 0.5 * g * (a_op.dag() * b_op + a_op * b_op.dag())
    Omega = np.sqrt(g**2 + (chi/2)**2)
    t = 2.0 * np.pi / Omega
    return _propagator(H, t, c_ops=c_ops)


def R_osc(a_op: Qobj, phi: float, c_ops=None):
    """this gate is done in software and thus can be done with unit fidelity
    (hence why the collapse operators are not used even in the case when they are passed)"""
    cav_rotation = (-1j * phi * a_op.dag() * a_op).expm()
    if c_ops is None:
        return cav_rotation
    return to_super(cav_rotation)


def R_tmon(s_op, g, t, c_ops=None):
    return _propagator(0.5 * g * s_op, t, c_ops=c_ops)


def _propagator(H, t, c_ops=None):
    if c_ops is None:
        return (-1j * H * t).expm()
    return (liouvillian(H, c_ops) * t).expm()


def beamsplitter(a_op, b_op, g, t, c_ops=None):
    H = 0.5 * g * a_op.dag() * b_op + 0.5 * np.conjugate(g) * b_op.dag() * a_op
    return _propagator(H, t, c_ops=c_ops)


def cZU(a_op: Qobj, sz: Qobj, chi, c_ops=None):
    H = 0.5 * chi * sz * a_op.dag() * a_op
    t = np.pi / chi
    return _propagator(H, t, c_ops=c_ops)


def cZZZU(a_op: Qobj, b_op: Qobj, c_op: Qobj, sz: Qobj, chi, c_ops=None):
    U1 = cZZU(b_op, a_op, sz, chi, c_ops)
    U2 = cZZU(b_op, c_op, sz, chi, c_ops)
    U3 = cZU(b_op, sz, chi, c_ops)
    return U1 * U2 * U3


def SWAP(a_op: Qobj, b_op: Qobj):
    return ((np.pi / 2) * (a_op.dag() * b_op - a_op * b_op.dag())).expm()


def id_wrap_ops(op: Qobj, idx: int, truncated_dims: list):
    assert op.dims[0][0] == truncated_dims[idx]
    id_list = [qeye(dim) for dim in truncated_dims]
    id_list[idx] = op
    return tensor(*id_list)


def construct_basis_states_list(Fock_states_spec: list, truncated_dims: list):
    basis_states = []
    for i, state_spec in enumerate(Fock_states_spec):
        basis_list = [basis(truncated_dims[i], state_spec[i])
                      for i in range(len(truncated_dims))]
        basis_states.append(tensor(*basis_list))
    return basis_states


def project_U(U: Qobj, Fock_states_spec: list, truncated_dims: list):
    dim_new_U = len(Fock_states_spec)
    basis_states = np.zeros((dim_new_U, np.prod(truncated_dims)), dtype=complex)
    # converting to numpy array - this is a hack!
    for i, state_spec in enumerate(Fock_states_spec):
        basis_list = [basis(truncated_dims[i], state_spec[i])
                      for i in range(len(truncated_dims))]
        basis_states[i, :] = tensor(*basis_list).data.toarray()[:, 0]
    new_U = np.conjugate(basis_states) @ U.data.toarray() @ basis_states.T
    return Qobj(new_U)


def logical_error_prob(U_diss, initial_Fock_states_spec, bad_Fock_states_spec, truncated_dims):
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
    initial_basis_list = construct_basis_states_list(initial_Fock_states_spec, truncated_dims)
    num_initial_states = len(initial_Fock_states_spec)
    bad_basis_list = construct_basis_states_list(bad_Fock_states_spec, truncated_dims)
    bad_pop_total = 0.0
    for i, initial_state in enumerate(initial_basis_list):
        final_state_vec = U_diss * operator_to_vector(initial_state * initial_state.dag())
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
    return bad_pop_total/num_initial_states


def gate_failure_prob(U_diss, initial_Fock_states_spec, detected_Fock_states_spec, truncated_dims):
    initial_basis_list = construct_basis_states_list(initial_Fock_states_spec, truncated_dims)
    num_initial_states = len(initial_Fock_states_spec)
    detected_basis_list = construct_basis_states_list(detected_Fock_states_spec, truncated_dims)
    detected_pop_total = 0.0
    for i, initial_state in enumerate(initial_basis_list):
        final_state_vec = U_diss * operator_to_vector(initial_state * initial_state.dag())
        final_dm = vector_to_operator(final_state_vec)
        for j, detected_state in enumerate(detected_basis_list):
            detected_pop = detected_state.dag() * final_dm * detected_state
#            print(detected_Fock_states_spec[j], detected_pop)
            detected_pop_total += np.real(detected_pop.data.toarray()[0, 0])
    return detected_pop_total/num_initial_states


def truncate_superoperator(superop, keep_idxs):
    """
    Parameters
    ----------
    superop
        superoperator to truncate. We consider the situation where certain
        states are relevant for predicting time evolution, however the gate under consideration
        does not care about the time evolution of those states themselves (with population
        e.g. beginning in that state.
    keep_idxs
        indices of the states to keep
    Returns
    -------
        truncated superoperator

    """
    keep_dim = len(keep_idxs)
    total_dim = superop
    trunc_dim = total_dim - keep_dim
    truncated_mat = np.zeros((keep_dim ** 2, keep_dim ** 2), dtype=complex)
    total_dim = keep_dim + trunc_dim
    locs = [total_dim * keep_idx_i + keep_idx_j
            for keep_idx_i in keep_idxs
            for keep_idx_j in keep_idxs]
    for i, loc_i in enumerate(locs):
        for j, loc_j in enumerate(locs):
            truncated_mat[i, j] = superop.data.toarray()[loc_i, loc_j]
    return Qobj(truncated_mat, type='super', dims=[[[keep_dim], [keep_dim]],
                                                   [[keep_dim], [keep_dim]]])

def my_to_chi(q_oper):
    """
    q_oper
        superoperator to transform into chi matrix
    """
    pauli_ops_oneq = [qeye(2) / 2, sigmax() / 2,
                      sigmay() / 2, sigmaz() / 2]
    pauli_ops = [Qobj(tensor(pauli_op1, pauli_op2), dims=[[4], [4]]) for pauli_op1 in pauli_ops_oneq
                 for pauli_op2 in pauli_ops_oneq]
    kraus_ops = to_kraus(q_oper)
    e_ij_coeffs = np.array([[np.trace(kraus_op.dag() * pauli_op)
                             for pauli_op in pauli_ops]
                            for kraus_op in kraus_ops]
                          )
    return np.conjugate(e_ij_coeffs).T @ e_ij_coeffs

def calc_fidel_chi(chi_real, chi_ideal):
    return (4 * np.trace(chi_real @ chi_ideal) + np.trace(chi_real))/5

# def unconditional_SWAP(a_op, b_op, sz, chi, c_ops=None):
#     g = 1j * chi / 2
#     H = 0.5 * chi * sz * a_op.dag() * a_op + g * a_op.dag() * b_op - g.conj() * a_op * b_op.dag()
#     t = 2.0 * np.pi / Omega
#     if c_ops is None:
#         return (-1j * H * t).expm()
#     else:
#         return (liouvillian_ref(H, c_ops) * t).expm()


def test_cZZU():
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
    Fock_states_spec = [(i, j, k)
                         for i in range(cavity_fock_trunc)
                              for j in range(cavity_fock_trunc)
                              for k in range(2)]
    cZZU_projected = project_U(cZZU(chi, a, b, sz), Fock_states_spec, truncated_dims)
    ideal_cZZU = (-1j * (np.pi / 2) * sz * (a.dag() * a + b.dag() * b)).expm()
    ideal_cZZU_projected = project_U(ideal_cZZU, Fock_states_spec, truncated_dims)
    assert ideal_cZZU_projected == cZZU_projected