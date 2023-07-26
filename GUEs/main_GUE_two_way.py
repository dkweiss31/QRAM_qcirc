import numpy as np
from qutip import tensor, Qobj

from GUEs.simulate_GUE import SimulateGUETwoWay, SimulateGUETwoWayDR
from utils.hashing import Hashing
from utils.utils import write_to_h5


def main_GUE_two_way(filepath, param_dict):
    guefidelity = SimulateGUETwoWay(**param_dict)
    guefidelity_DR = SimulateGUETwoWayDR(**param_dict)
    hashing_hilbert_dim = guefidelity.hilbert_dim()
    vac = np.zeros(hashing_hilbert_dim)
    vac[0] = 1.0
    state_000000 = Qobj(vac)
    state_001000 = guefidelity.b1.dag() * state_000000
    state_000100 = guefidelity.b2.dag() * state_000000
    psi_init_R = (state_001000 + 1j * state_000100).unit()
    psi_init_L = (state_001000 - 1j * state_000100).unit()
    label_list_SR = ["0", "R", "L"]
    label_list_DR = ["R0", "L0", "0R", "0L"]
    keep_idxs = [guefidelity.a1_idx, guefidelity.a2_idx, guefidelity.c1_idx, guefidelity.c2_idx]
    initial_basis_states = [state_000000, psi_init_R, psi_init_L]
    red_hash = Hashing(number_degrees_freedom=4, num_exc=param_dict["num_exc"])
    red_hilbert_dim = red_hash.hilbert_dim()
    vac = np.zeros(red_hilbert_dim)
    vac[0] = 1.0
    zero_state_GUE = Qobj(vac)
    (red_a1, red_a2, red_c1, red_c2) = (red_hash.a_operator(idx) for idx in range(4))
    zero_right = ((red_c1.dag() + 1j * red_c2.dag()) * zero_state_GUE).unit()
    left_zero = ((red_a1.dag() - 1j * red_a2.dag()) * zero_state_GUE).unit()
    ideal_final_basis_states = [zero_state_GUE, zero_right, left_zero]
    ideal_final_basis_states_DR = [
        tensor(zero_right, zero_state_GUE),
        tensor(left_zero, zero_state_GUE),
        tensor(zero_state_GUE, zero_right),
        tensor(zero_state_GUE, left_zero),
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
