import numpy as np
from qutip import tensor, basis, Qobj, qsave

from GUEs.simulate_GUE import (
    SimulateGUE,
    SimulateGUEDR,
    SimulateGUEHashing,
    SimulateGUEHashingDR,
)
from QRAM_utils.hashing import Hashing
from QRAM_utils.utils import write_to_h5


def main_GUE(filepath, param_dict):
    hashing = param_dict.pop("hashing")
    plot = param_dict.pop("plot")
    if hashing:
        guefidelity = SimulateGUEHashing(**param_dict)
        guefidelity_DR = SimulateGUEHashingDR(**param_dict)
        new_hash = Hashing(number_degrees_freedom=2, num_exc=param_dict["num_exc"])
        red_hilbert_dim = new_hash.hilbert_dim()
        vac = np.zeros(red_hilbert_dim)
        vac[0] = 1.0
        red_zero_state = Qobj(vac)
        red_c1 = new_hash.a_operator(0)
        red_c2 = new_hash.a_operator(1)
        red_right_state = ((red_c1.dag() + 1j * red_c2.dag()) * red_zero_state).unit()
    else:
        cavity_dim = param_dict["cavity_dim"]
        guefidelity = SimulateGUE(**param_dict)
        guefidelity_DR = SimulateGUEDR(**param_dict)
        red_zero_state = tensor(*[basis(dim, 0) for dim in [cavity_dim, cavity_dim]])
        c1_idx = guefidelity_DR.c1_idx
        c2_idx = guefidelity_DR.c2_idx
        red_right_state = guefidelity.rightward_state(c1_idx, c2_idx)
    state_0000 = guefidelity.vacuum_state()
    state_1000 = guefidelity.b1.dag() * state_0000
    state_0100 = guefidelity.b2.dag() * state_0000
    state_0010 = guefidelity.c1.dag() * state_0000
    state_0001 = guefidelity.c2.dag() * state_0000
    r_state_1000 = guefidelity.b1_r.dag() * state_0000
    r_state_0100 = guefidelity.b2_r.dag() * state_0000
    r_state_0010 = guefidelity.c1_r.dag() * state_0000
    r_state_0001 = guefidelity.c2_r.dag() * state_0000
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
    if plot:
        psi_b_r = (r_state_1000 + 1j * r_state_0100).unit()
        psi_c_r = (r_state_0010 + 1j * r_state_0001).unit()
        psi_fin = (state_0010 + 1j * state_0001).unit()
        e_ops = [
            psi_init * psi_init.dag(),
            psi_fin * psi_fin.dag(),
            psi_b_r * psi_b_r.dag(),
            psi_c_r * psi_c_r.dag()
        ]
        result = guefidelity.run_state_transfer(psi_init * psi_init.dag(), e_ops=e_ops, final_state_only=False)
        qsave(result, filepath)
    else:
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
