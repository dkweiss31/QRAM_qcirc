from functools import partial

import numpy as np
from qutip import tensor, basis

from quantum_helpers import operator_basis_lidar, apply_gate_to_states, operators_from_states
from simulate_GUE import SimulateGUETwoWay, SimulateGUEOneWay, SimulateGUEOneWayDR
from utils import construct_basis_states_list, get_map
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
    guefidelity_label_DR = SimulateGUEOneWayDR(
        gamma_b_avg=gamma_b_avg,
        gamma_b_dev=gamma_b_dev,
        gamma_c_avg=gamma_c_avg,
        gamma_c_dev=gamma_c_dev,
        cavity_dim=cavity_dim,
        additional_label=True,
    )

    Fock_states_spec = [
        (0, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ]
    (
        state_0000,
        state_1000,
        state_0100,
        state_0010,
        state_0001,
        r_state_1000,
        r_state_0100,
        r_state_0010,
        r_state_0001,
    ) = construct_basis_states_list(Fock_states_spec, guefidelity.truncated_dims)

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

    def append_label(states, labels):
        return list(
            [
                tensor(state, basis(2, label))
                for state, label in zip(states, labels)
            ]
        )
    additional_labels = [0, 0, 1]  # vacuum then right and left
    initial_basis_states = append_label([state_0000, psi_init, psi_init], additional_labels)
    ideal_final_basis_states = append_label([state_0000, psi_fin, psi_fin], additional_labels)
    # want to simulate going to the left and to the right. These are independent
    # so we want to save resources by not simulating the whole thing. However we need to
    # signify that the initial states going in different directions are orthogonal. So we add
    # a fictitious "spin" label that ensures their orthogonality
    label_list_SR = ["0R", "1R", "1L"]
    label_list_DR = ["1R0R", "1L0R", "0R1R", "0R1L"]
    op_dict_SR, initial_cardinal_states = operator_basis_lidar(initial_basis_states, label_list=label_list_SR)
    _, ideal_final_cardinal_states = operator_basis_lidar(ideal_final_basis_states, label_list=label_list_SR)
    num_states = len(initial_cardinal_states)
    # first calc state transfer fidel in the SR case
    partial_state_tran = partial(guefidelity_label.run_state_transfer, args, c_ops_label, None)
    final_SR = apply_gate_to_states(partial_state_tran, initial_cardinal_states, num_cpus)
    fidel_SR = 0.0
    for (real_final_result, ideal_final_state) in zip(final_SR.values(), ideal_final_cardinal_states.values()):
        fidel_SR += np.trace(real_final_result.final_state * ideal_final_state)
    fidel_SR = fidel_SR / num_states
    print(f"fidelity of SR state transfer is {fidel_SR}")
    # now for DR
    initial_basis_states_DR = guefidelity_label_DR.DR_basis(initial_basis_states)
    ideal_final_basis_states_DR = guefidelity_label_DR.DR_basis(ideal_final_basis_states)
    op_dict_DR, unique_state_dict_DR = operator_basis_lidar(initial_basis_states_DR, label_list=label_list_DR)
    _, ideal_final_cardinal_states_DR = operator_basis_lidar(ideal_final_basis_states_DR, label_list=label_list_DR)
    final_SR_ops = operators_from_states(op_dict_SR, final_SR)
    final_DR_states = {
        label: guefidelity_label_DR.DR_state_from_SR_ops(label, final_SR_ops)
        for label, state in unique_state_dict_DR.items()
    }
    fidel_DR = 0.0
    total_prob = 0.0
    for (real_final_result, ideal_final_state) in zip(final_DR_states.values(),
                                                      ideal_final_cardinal_states_DR.values()):
        meas_op_DR = guefidelity_label_DR.measurement_op_DR(guefidelity.c1_idx, guefidelity.c2_idx)
        meas_final_state = meas_op_DR * real_final_result.final_state * meas_op_DR.dag()
        prob = np.trace(meas_final_state)
        total_prob += prob
        fidel_DR += np.trace(meas_final_state * ideal_final_state)
    fidel_DR = fidel_DR / num_states
    total_prob = total_prob / num_states
    print(f"fidelity of DR state transfer is {fidel_DR} with prob {total_prob}")


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
