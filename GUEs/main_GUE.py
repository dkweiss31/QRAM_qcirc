from qutip import tensor, basis

from GUEs.simulate_GUE import SimulateGUE, SimulateGUEDR
from utils.quantum_helpers import (
    operator_basis_lidar,
    apply_gate_to_states,
    operators_from_states,
)
from utils.utils import construct_basis_states_list, write_to_h5


def main_GUE(filepath, param_dict):
    cavity_dim = param_dict["cavity_dim"]
    num_cpus = param_dict["num_cpus"]
    additional_label = param_dict["additional_label"]
    guefidelity = SimulateGUE(**param_dict)
    guefidelity_DR = SimulateGUEDR(**param_dict)
    c1_idx = guefidelity_DR.c1_idx
    c2_idx = guefidelity_DR.c2_idx

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
    psi_fin = (state_0010 + 1j * state_0001).unit()

    # trace out initial GUEs as well as transfer resonators
    def trace_out_dict(state_dict, keep_idxs):
        return {
            label: final_state.ptrace(keep_idxs)
            for label, final_state in state_dict.items()
        }

    if additional_label:

        def append_label(states, labels):
            assert states[0].isket
            return list(
                [tensor(state, basis(2, label)) for state, label in zip(states, labels)]
            )

        # want to simulate going to the left and to the right. These are independent
        # so we want to save resources by not simulating the whole thing. However we need to
        # signify that the initial states going in different directions are orthogonal. So we add
        # a fictitious "spin" label that ensures their orthogonality
        additional_labels = [0, 0, 1]  # vacuum then right and left
        label_list_SR = ["0R", "1R", "1L"]
        label_list_DR = ["1R0R", "1L0R", "0R1R", "0R1L"]
        # Keep final GUE and additional label
        keep_idxs = [guefidelity.c1_idx, guefidelity.c2_idx, 8]
        initial_basis_states = append_label(
            [state_0000, psi_init, psi_init], additional_labels
        )
        # TODO need to do trace out thing for this approach also
        ideal_final_basis_states = append_label(
            [state_0000, psi_fin, psi_fin], additional_labels
        )
        # now for DR. want to construct the states that we get after tracing out irrelevant
        # degrees of freedom
        zero_state = tensor(*[basis(dim, 0) for dim in [cavity_dim, cavity_dim, 2]])
        right_state = guefidelity_DR.rightward_state(c1_idx, c2_idx)
        one_R = tensor(right_state, basis(2, 0))
        one_L = tensor(right_state, basis(2, 1))
        ideal_final_basis_states_DR = [
            tensor(one_R, zero_state),
            tensor(one_L, zero_state),
            tensor(zero_state, one_R),
            tensor(zero_state, one_L),
        ]
    else:
        label_list_SR = ["0", "1"]
        label_list_DR = ["10", "01"]
        keep_idxs = [guefidelity.c1_idx, guefidelity.c2_idx]
        initial_basis_states = [state_0000, psi_init]
        # need to do the below because we trace out the intermediate states when
        # calculating the fidelity
        zero_state = tensor(*[basis(dim, 0) for dim in [cavity_dim, cavity_dim]])
        right_state = guefidelity.rightward_state(c1_idx, c2_idx)
        ideal_final_basis_states = [zero_state, right_state]
        # DR
        ideal_final_basis_states_DR = [
            tensor(right_state, zero_state),
            tensor(zero_state, right_state),
        ]
    op_dict_SR, initial_cardinal_states = operator_basis_lidar(
        initial_basis_states, label_list=label_list_SR
    )
    _, ideal_final_cardinal_states = operator_basis_lidar(
        ideal_final_basis_states, label_list=label_list_SR
    )
    # first calc state transfer fidel in the SR case
    final_SR_states = apply_gate_to_states(
        guefidelity.run_state_transfer, initial_cardinal_states, num_cpus
    )
    final_SR_states = trace_out_dict(final_SR_states, keep_idxs)
#    ideal_final_cardinal_states = trace_out_dict(ideal_final_cardinal_states, keep_idxs)
    fidel_SR, _ = guefidelity.state_transfer_fidelity(
        final_SR_states, ideal_final_cardinal_states
    )
    print(f"fidelity of SR state transfer is {fidel_SR}")
    _, ideal_final_cardinal_states_DR = operator_basis_lidar(
        ideal_final_basis_states_DR, label_list=label_list_DR
    )
    # no need to trace out op_dict_SR since its only used for labels and coefficients decomposition
    final_SR_ops = operators_from_states(op_dict_SR, final_SR_states)
    final_DR_states = {
        label: guefidelity_DR.DR_state_from_SR_ops(label, final_SR_ops)
        for label, state in ideal_final_cardinal_states_DR.items()
    }
    meas_op_DR = guefidelity_DR.measurement_op_DR(
        guefidelity_DR.c1_idx, guefidelity_DR.c2_idx
    )
    fidel_DR, prob_DR = guefidelity_DR.state_transfer_fidelity(
        final_DR_states, ideal_final_cardinal_states_DR, measurement_op=meas_op_DR
    )
    print(f"fidelity of DR state transfer is {fidel_DR} with prob {prob_DR}")
    data_dict = {"fidel_SR": fidel_SR, "fidel_DR": fidel_DR, "prob_DR": prob_DR}
    write_to_h5(filepath, data_dict, param_dict)
