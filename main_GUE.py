from functools import partial

from qutip import tensor, basis

from quantum_helpers import (
    operator_basis_lidar,
    apply_gate_to_states,
    operators_from_states,
)
from simulate_GUE import SimulateGUEOneWay, SimulateGUEOneWayDR
from utils import construct_basis_states_list


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
            [tensor(state, basis(2, label)) for state, label in zip(states, labels)]
        )

    additional_labels = [0, 0, 1]  # vacuum then right and left
    initial_basis_states = append_label(
        [state_0000, psi_init, psi_init], additional_labels
    )
    ideal_final_basis_states = append_label(
        [state_0000, psi_fin, psi_fin], additional_labels
    )
    # want to simulate going to the left and to the right. These are independent
    # so we want to save resources by not simulating the whole thing. However we need to
    # signify that the initial states going in different directions are orthogonal. So we add
    # a fictitious "spin" label that ensures their orthogonality
    label_list_SR = ["0R", "1R", "1L"]
    label_list_DR = ["1R0R", "1L0R", "0R1R", "0R1L"]
    op_dict_SR, initial_cardinal_states = operator_basis_lidar(
        initial_basis_states, label_list=label_list_SR
    )
    _, ideal_final_cardinal_states = operator_basis_lidar(
        ideal_final_basis_states, label_list=label_list_SR
    )
    num_states = len(initial_cardinal_states)
    # first calc state transfer fidel in the SR case
    partial_state_tran = partial(
        guefidelity_label.run_state_transfer, args, c_ops_label, None
    )
    final_SR_states = apply_gate_to_states(
        partial_state_tran, initial_cardinal_states, num_cpus
    )
    fidel_SR, _ = guefidelity_label.state_transfer_fidelity(final_SR_states, ideal_final_cardinal_states)
    print(f"fidelity of SR state transfer is {fidel_SR}")
    # now for DR
    initial_basis_states_DR = guefidelity_label_DR.DR_basis(initial_basis_states)
    ideal_final_basis_states_DR = guefidelity_label_DR.DR_basis(
        ideal_final_basis_states
    )
    op_dict_DR, unique_state_dict_DR = operator_basis_lidar(
        initial_basis_states_DR, label_list=label_list_DR
    )
    _, ideal_final_cardinal_states_DR = operator_basis_lidar(
        ideal_final_basis_states_DR, label_list=label_list_DR
    )
    final_SR_ops = operators_from_states(op_dict_SR, final_SR_states)
    final_DR_states = {
        label: guefidelity_label_DR.DR_state_from_SR_ops(label, final_SR_ops)
        for label, state in unique_state_dict_DR.items()
    }
    meas_op_DR = guefidelity_label_DR.measurement_op_DR(guefidelity_label_DR.c1_idx, guefidelity_label_DR.c2_idx)
    fidel_DR, prob_DR = guefidelity_label_DR.state_transfer_fidelity(final_DR_states, ideal_final_cardinal_states_DR,
                                                                     measurement_op=meas_op_DR)
    print(f"fidelity of DR state transfer is {fidel_DR} with prob {prob_DR}")
