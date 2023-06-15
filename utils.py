from qutip import tensor, Qobj, basis, qeye, to_kraus, sigmax, sigmay, sigmaz
import numpy as np

def id_wrap_ops(op: Qobj, idx: int, truncated_dims: list):
    assert op.dims[0][0] == truncated_dims[idx]
    id_list = [qeye(dim) for dim in truncated_dims]
    id_list[idx] = op
    return tensor(*id_list)

def construct_basis_states_list(Fock_states_spec: list, truncated_dims: list):
    basis_states = []
    for state_spec in Fock_states_spec:
        basis_list = [
            basis(truncated_dims[i], state_spec[i])
            for i in range(len(truncated_dims))
        ]
        basis_states.append(tensor(*basis_list))
    return basis_states

def project_U(U: Qobj, Fock_states_spec: list, truncated_dims: list):
    dim_new_U = len(Fock_states_spec)
    basis_states = np.zeros((dim_new_U, np.prod(truncated_dims)), dtype=complex)
    # converting to numpy array - this is a hack!
    for i, state_spec in enumerate(Fock_states_spec):
        basis_list = [
            basis(truncated_dims[j], state_spec[j])
            for j in range(len(truncated_dims))
        ]
        basis_states[i, :] = tensor(*basis_list).data.toarray()[:, 0]
    new_U = np.conjugate(basis_states) @ U.data.toarray() @ basis_states.T
    return Qobj(new_U)

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
    truncated_mat = np.zeros((keep_dim**2, keep_dim**2), dtype=complex)
    total_dim = keep_dim + trunc_dim
    locs = [
        total_dim * keep_idx_i + keep_idx_j
        for keep_idx_i in keep_idxs
        for keep_idx_j in keep_idxs
    ]
    for i, loc_i in enumerate(locs):
        for j, loc_j in enumerate(locs):
            truncated_mat[i, j] = superop.data.toarray()[loc_i, loc_j]
    return Qobj(
        truncated_mat,
        type="super",
        dims=[[[keep_dim], [keep_dim]], [[keep_dim], [keep_dim]]],
    )

def my_to_chi(q_oper):
    """
    q_oper
        superoperator to transform into chi matrix
    """
    pauli_ops_oneq = [qeye(2) / 2, sigmax() / 2, sigmay() / 2, sigmaz() / 2]
    pauli_ops = [
        Qobj(tensor(pauli_op1, pauli_op2), dims=[[4], [4]])
        for pauli_op1 in pauli_ops_oneq
        for pauli_op2 in pauli_ops_oneq
    ]
    kraus_ops = to_kraus(q_oper)
    e_ij_coeffs = np.array(
        [
            [np.trace(kraus_op.dag() * pauli_op) for pauli_op in pauli_ops]
            for kraus_op in kraus_ops
        ]
    )
    return np.conjugate(e_ij_coeffs).T @ e_ij_coeffs

def calc_fidel_chi(chi_real, chi_ideal):
    return (4 * np.trace(chi_real @ chi_ideal) + np.trace(chi_real)) / 5