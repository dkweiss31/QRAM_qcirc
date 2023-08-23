from qutip import tensor

from GUEs.simulate_GUE import (
    SimulateGUE,
    SimulateGUEDR,
)
from QRAM_utils.utils import write_to_h5


def main_GUE(filepath: str, guefidelity: SimulateGUE,
             guefidelity_DR: SimulateGUEDR, param_dict: dict):
    red_right_state = guefidelity.reduced_rightward_state()
    red_zero_state = guefidelity.reduced_zero_state()
    state_0000 = guefidelity.vacuum_state()
    state_1000 = guefidelity.b1.dag() * state_0000
    state_0100 = guefidelity.b2.dag() * state_0000
    psi_init = (state_1000 + 1j * state_0100).unit()
    label_list_SR = ["0", "1"]
    label_list_DR = ["10", "01"]
    keep_idxs = [guefidelity.c1_idx, guefidelity.c2_idx]
    initial_basis_states = [state_0000, psi_init]
    ideal_final_basis_states = [red_zero_state, red_right_state]
    ideal_final_basis_states_DR = [
        tensor(red_right_state, red_zero_state),
        tensor(red_zero_state, red_right_state),
    ]
    fidel_SR, final_SR_states = guefidelity.overall_state_transfer_fidelity(
        initial_basis_states, label_list_SR, ideal_final_basis_states, keep_idxs
    )
    print(f"fidelity of SR state transfer is {fidel_SR}")
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
    print(f"fidelity of DR state transfer is {fidel_DR} with prob {prob_DR}")
    data_dict = {"fidel_SR": fidel_SR, "fidel_DR": fidel_DR, "prob_DR": prob_DR}
    write_to_h5(filepath, data_dict, param_dict)
