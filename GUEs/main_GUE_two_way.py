from qutip import tensor

from QRAM_utils.utils import write_to_h5


def main_GUE_two_way(filepath, guefidelity, guefidelity_DR, param_dict):
    red_zero_state = guefidelity.reduced_zero_state()
    red_right_state = guefidelity.reduced_rightward_state()
    red_left_state = guefidelity.reduced_leftward_state()
    state_000000 = guefidelity.vacuum_state()
    state_001000 = guefidelity.b1.dag() * state_000000
    state_000100 = guefidelity.b2.dag() * state_000000
    psi_init_R = (state_001000 + 1j * state_000100).unit()
    psi_init_L = (state_001000 - 1j * state_000100).unit()
    label_list_SR = ["0", "R", "L"]
    label_list_DR = ["R0", "L0", "0R", "0L"]
    keep_idxs = [guefidelity.a1_idx, guefidelity.a2_idx, guefidelity.c1_idx, guefidelity.c2_idx]
    initial_basis_states = [state_000000, psi_init_R, psi_init_L]
    ideal_final_basis_states = [red_zero_state, red_right_state, red_left_state]
    ideal_final_basis_states_DR = [
        tensor(red_right_state, red_zero_state),
        tensor(red_left_state, red_zero_state),
        tensor(red_zero_state, red_right_state),
        tensor(red_zero_state, red_left_state),
    ]
    fidel_SR, final_SR_states = guefidelity.overall_state_transfer_fidelity(
        initial_basis_states, label_list_SR, ideal_final_basis_states, keep_idxs
    )
    print(f"fidelity of two-way SR state transfer is {fidel_SR}")
    final_DR_states, ideal_final_cardinal_states_DR = guefidelity_DR.DR_final_states(
        initial_basis_states,
        label_list_SR,
        ideal_final_basis_states_DR,
        label_list_DR,
        final_SR_states,
    )
    meas_op_DR = sum([state * state.dag() for state in ideal_final_basis_states_DR])
    fidel_DR, prob_DR = guefidelity_DR.state_transfer_fidelity(
        final_DR_states, ideal_final_cardinal_states_DR, measurement_op=meas_op_DR
    )
    print(f"fidelity of two-way DR state transfer is {fidel_DR} with prob {prob_DR}")
    data_dict = {"fidel_SR": fidel_SR, "fidel_DR": fidel_DR, "prob_DR": prob_DR}
    write_to_h5(filepath, data_dict, param_dict)
