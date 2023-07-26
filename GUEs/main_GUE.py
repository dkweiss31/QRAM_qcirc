import numpy as np
from qutip import tensor, basis, Qobj

from GUEs.simulate_GUE import (
    SimulateGUE,
    SimulateGUEDR,
    SimulateGUEHashing,
    SimulateGUEHashingDR,
)
from utils.hashing import Hashing
from utils.utils import construct_basis_states_list, write_to_h5


def main_GUE(filepath, param_dict):
    cavity_dim = param_dict["cavity_dim"]
    hashing = param_dict.pop("hashing")
    if hashing:
        guefidelity = SimulateGUEHashing(**param_dict)
        guefidelity_DR = SimulateGUEHashingDR(**param_dict)
        hashing_hilbert_dim = guefidelity.hilbert_dim()
        vac = np.zeros(hashing_hilbert_dim)
        vac[0] = 1.0
        state_0000 = Qobj(vac)
        state_1000 = guefidelity.b1.dag() * state_0000
        state_0100 = guefidelity.b2.dag() * state_0000
        psi_init = (state_1000 + 1j * state_0100).unit()
        new_hash = Hashing(number_degrees_freedom=2, num_exc=param_dict["num_exc"])
        red_hilbert_dim = new_hash.hilbert_dim()
        vac = np.zeros(red_hilbert_dim)
        vac[0] = 1.0
        zero_state = Qobj(vac)
        red_c1 = new_hash.a_operator(0)
        red_c2 = new_hash.a_operator(1)
        right_state = ((red_c1.dag() + 1j * red_c2.dag()) * zero_state).unit()
    else:
        guefidelity = SimulateGUE(**param_dict)
        guefidelity_DR = SimulateGUEDR(**param_dict)
        Fock_states_spec = [
            (0, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0),
        ]
        (
            state_0000,
            state_1000,
            state_0100,
            state_0010,
            state_0001,
        ) = construct_basis_states_list(Fock_states_spec, guefidelity.truncated_dims)
        psi_init = (state_1000 + 1j * state_0100).unit()
        zero_state = tensor(*[basis(dim, 0) for dim in [cavity_dim, cavity_dim]])
        c1_idx = guefidelity_DR.c1_idx
        c2_idx = guefidelity_DR.c2_idx
        right_state = guefidelity.rightward_state(c1_idx, c2_idx)
    label_list_SR = ["0", "1"]
    label_list_DR = ["10", "01"]
    keep_idxs = [guefidelity.c1_idx, guefidelity.c2_idx]
    initial_basis_states = [state_0000, psi_init]
    ideal_final_basis_states = [zero_state, right_state]
    ideal_final_basis_states_DR = [
        tensor(right_state, zero_state),
        tensor(zero_state, right_state),
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
