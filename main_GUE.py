import numpy as np
from qutip import tensor, Qobj

from simulate_GUE import SimulateGUETwoWay, SimulateGUEOneWay
from utils import construct_basis_states_list
from qutip.fileio import qsave


def main_one_way(filepath, param_dict):
    gamma_b_avg = param_dict["gamma_b_avg"]
    gamma_c_avg = param_dict["gamma_c_avg"]
    gamma_b_dev = param_dict["gamma_b_dev"]
    gamma_c_dev = param_dict["gamma_c_dev"]
    scale_b = param_dict["scale_b"]
    scale_c = param_dict["scale_c"]
    t_half = param_dict["t_half"]
    B = param_dict["B"]
    c = param_dict["c"]
    num_cpus = param_dict["num_cpus"]
    cavity_dim = 2
    guefidelity = SimulateGUEOneWay(
        gamma_b_avg=gamma_b_avg,
        gamma_b_dev=gamma_b_dev,
        gamma_c_avg=gamma_c_avg,
        gamma_c_dev=gamma_c_dev,
        cavity_dim=cavity_dim,
        additional_label=False,  # make more general plz
    )
    guefidelity_label = SimulateGUEOneWay(
        gamma_b_avg=gamma_b_avg,
        gamma_b_dev=gamma_b_dev,
        gamma_c_avg=gamma_c_avg,
        gamma_c_dev=gamma_c_dev,
        cavity_dim=cavity_dim,
        additional_label=True,  # make more general plz
    )
    Fock_states_spec = [(0, 0, 0, 0, 0, 0, 0, 0),
                        (1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0),
                        (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0),
                        (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 1)]
    (state_0000,
     state_1000,
     state_0100,
     state_0010,
     state_0001,
     r_state_1000,
     r_state_0100,
     r_state_0010,
     r_state_0001) = construct_basis_states_list(Fock_states_spec, guefidelity.truncated_dims)

    psi_init = (state_1000 + 1j * state_0100).unit()
    psi_fin = (state_0010 + 1j * state_0001).unit()
    rho_init = psi_init * psi_init.dag()
    rho_fin = psi_fin * psi_fin.dag()

    c_ops_label = guefidelity_label.construct_c_ops(
        Gamma_1_cav=0.0, Gamma_1_transfer_nr=0.0, nth=0.0
    )
    c_ops = guefidelity.construct_c_ops(
        Gamma_1_cav=0.0, Gamma_1_transfer_nr=0.0, nth=0.0
    )

    args = {
        "c": c,
        "B": B,
        "t_half": t_half,
        "gamma_b_avg": gamma_b_avg,
        "scale_b": scale_b,
        "scale_c": scale_c,
    }
    result = guefidelity.run_state_transfer(args, c_ops=c_ops, init_state=rho_init)
    initial_basis_states = [state_0000, state_0000, psi_init, psi_init]
    ideal_final_basis_states = [state_0000, state_0000, psi_fin, psi_fin]
    # want to simulate going to the left and to the right. These are independent
    # so we want to save resources by not simulating the whole thing. However we need to
    # signify that the initial states going in different directions are orthogonal. So we add
    # a fictitious "spin" label that ensures their orthogonality
    additional_labels = [0, 1, 0, 1]
    overall_st_t_fidel = guefidelity_label.state_transfer_fidelity(initial_basis_states, ideal_final_basis_states,
                                                                additional_labels, args, c_ops_label, num_cpus=num_cpus)
    zero_vec_DR = tensor(state_0000, state_0000)
    fidel = (psi_fin.dag() * result.final_state * psi_fin).data.toarray()[0, 0]
    zero_pop_SR = (state_0000.dag() * result.final_state * state_0000).data.toarray()[0, 0]
    final_state_DR = tensor(result.final_state, result.final_state)
    norm = np.trace(final_state_DR)
    zero_pop_DR = (zero_vec_DR.dag() * final_state_DR * zero_vec_DR).data.toarray()[0, 0]
    measurement_op = guefidelity.measurement_op_DR()
    final_state_measured_DR = measurement_op * final_state_DR * measurement_op.dag()
    psi_fin_DR = tensor(psi_fin, psi_fin)
    prob = np.trace(final_state_measured_DR)
    fidel_DR = (psi_fin_DR.dag() * final_state_measured_DR * psi_fin_DR).data.toarray()[0, 0]
    print(f"fidelity of state transfer is {fidel}")
    print(
        f"unnormalized fidelity, probability, and normalized fidelity of DR state transfer are "
        f"{fidel_DR, prob, fidel_DR/prob}"
    )
    qsave(result, filepath)


def main_two_way(filepath, param_dict):
    gamma_a_avg = param_dict["gamma_a_avg"]
    gamma_b_avg = param_dict["gamma_b_avg"]
    gamma_c_avg = param_dict["gamma_c_avg"]
    gamma_a_dev = param_dict["gamma_a_dev"]
    gamma_b_dev = param_dict["gamma_b_dev"]
    gamma_c_dev = param_dict["gamma_c_dev"]
    scale_a = param_dict["scale_a"]
    scale_b = param_dict["scale_b"]
    scale_c = param_dict["scale_c"]
    t_half = param_dict["t_half"]
    B = param_dict["B"]
    c = param_dict["c"]
    cavity_dim = 2
    guefidelitytwoway = SimulateGUETwoWay(
        gamma_a_avg=gamma_a_avg,
        gamma_a_dev=gamma_a_dev,
        gamma_b_avg=gamma_b_avg,
        gamma_b_dev=gamma_b_dev,
        gamma_c_avg=gamma_c_avg,
        gamma_c_dev=gamma_c_dev,
        cavity_dim=cavity_dim,
    )
    zero_vec = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert len(zero_vec) == 12
    Fock_states_spec = 12 * [zero_vec]
    for idx in range(12):
        Fock_vec = Fock_states_spec[idx]
        Fock_vec[idx] = 1
        Fock_states_spec[idx] = tuple(Fock_vec)
    (
        state_a1,
        state_a2,
        state_b1,
        state_b2,
        state_c1,
        state_c2,
        state_a1_r,
        state_a2_r,
        state_b1_r,
        state_b2_r,
        state_c1_r,
        state_c2_r,
    ) = construct_basis_states_list(Fock_states_spec, guefidelitytwoway.truncated_dims)

    psi_init = (
        (state_b1 - 1j * state_b2).unit() + (state_b1 + 1j * state_b2).unit()
    ).unit()
    psi_fin = (
        (state_a1 - 1j * state_a2).unit() + (state_c1 + 1j * state_c2).unit()
    ).unit()
    rho_init = psi_init * psi_init.dag()
    rho_fin = psi_fin * psi_fin.dag()

    c_ops = guefidelitytwoway.construct_c_ops(
        Gamma_1_cav=0.0, Gamma_1_transfer_nr=0.0, nth=0.0
    )
    e_ops = [rho_init, rho_fin]

    args = {
        "c": c,
        "B": B,
        "t_half": t_half,
        "gamma_b_avg": gamma_b_avg,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "scale_c": scale_c,
    }
    result = guefidelitytwoway.run_state_transfer(
        rho_init, args, c_ops=c_ops, e_ops=e_ops
    )
    fidel = (psi_fin.dag() * result.final_state * psi_fin).data.toarray()[0, 0]
    print(f"fidelity of state transfer is {fidel}")
    qsave(result, filepath)
