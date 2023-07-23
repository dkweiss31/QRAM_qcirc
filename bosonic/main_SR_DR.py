from functools import partial

import numpy as np
from qutip import tensor, destroy, Qobj, to_super

from simulate_bosonic_ops import SimulateBosonicOperations, SimulateBosonicOperationsDR
from utils.fidelity import Fidelity
from utils.quantum_helpers import (
    apply_gate_to_states,
    operator_basis_lidar,
    operators_from_states,
)
from utils.utils import (
    id_wrap_ops,
    construct_basis_states_list,
    project_U,
    write_to_h5,
)


def main_SR_DR(filepath, param_dict):
    tmon_dim = param_dict["tmon_dim"]
    cavity_dim = param_dict["cavity_dim"]
    chi = param_dict["chi"]
    tmon_d_strength = param_dict["tmon_d_strength"]
    params = (tmon_d_strength, chi)
    eta_gg = param_dict["eta_gg"]
    eta_ge = param_dict["eta_ge"]
    eta_gf = param_dict["eta_gf"]
    liouv = param_dict["liouvillian"]
    num_cpus = param_dict["num_cpus"]
    atol = param_dict["atol"]
    rtol = param_dict["rtol"]
    nsteps = param_dict["nsteps"]
    cav_a_idx = 0
    cav_b_idx = 1
    tmon_idx = 2
    truncated_dims_SR = [cavity_dim, cavity_dim, tmon_dim]
    a = id_wrap_ops(destroy(cavity_dim), cav_a_idx, truncated_dims_SR)
    b = id_wrap_ops(destroy(cavity_dim), cav_b_idx, truncated_dims_SR)
    bosonic_sim_SR = SimulateBosonicOperations(
        gf_tmon=True,
        tmon_dim=tmon_dim,
        cavity_dim=cavity_dim,
        atol=atol,
        rtol=rtol,
        nsteps=nsteps,
    )
    bosonic_sim_DR = SimulateBosonicOperationsDR(
        gf_tmon=True,
        tmon_dim=tmon_dim,
        cavity_dim=cavity_dim,
        atol=atol,
        rtol=rtol,
        nsteps=nsteps,
    )
    # computational basis states include only Fock 0, 1 and transmon 0
    g_Fock_states_spec_SR = [(i, j, 0) for i in range(2) for j in range(2)]
    labels_SR = ["00", "01", "10", "11"]
    labels_DR = ["1100", "1001", "0110", "0011"]
    g_comp_basis_states_SR = construct_basis_states_list(
        g_Fock_states_spec_SR, truncated_dims_SR
    )
    g_comp_basis_states_DR = bosonic_sim_DR.DR_basis(g_comp_basis_states_SR)
    # measurement operator allowing for nonideal measurement
    measurement_op_SR = (
        np.sqrt(eta_gg) * bosonic_sim_SR.measurement_op_tmon_projector(0)
        + np.sqrt(eta_ge) * bosonic_sim_SR.measurement_op_tmon_projector(1)
        + np.sqrt(eta_gf) * bosonic_sim_SR.measurement_op_tmon_projector(2)
    )
    # ideal case, no dissipation
    U_eJP_ideal_SR = bosonic_sim_SR.U_eJP_func(a, b, params)
    U_eJP_ideal_DR = tensor(U_eJP_ideal_SR, U_eJP_ideal_SR)
    projected_ideal_SR = project_U(U_eJP_ideal_SR, basis_states=g_comp_basis_states_SR)
    projected_ideal_DR = project_U(U_eJP_ideal_DR, basis_states=g_comp_basis_states_DR)

    fidelity_SR = Fidelity(g_comp_basis_states_SR, labels_SR)
    fidelity_DR = Fidelity(g_comp_basis_states_DR, labels_DR)

    c_ops = bosonic_sim_SR.construct_c_ops(a, b, **param_dict)
    # explicitly construct the propogator superoperator by exponentiating the Liouvillian
    if liouv:
        # construct the single-rail superoperator
        final_SR = bosonic_sim_SR.U_eJP_func(a, b, params, c_ops)
        # for dual rail tensor together the two superoperators then reorder appropriately
        U_eJP_DR = tensor(final_SR, final_SR)
        DR_dims = to_super(U_eJP_ideal_DR).dims
        V2 = bosonic_sim_DR.V_2_op()
        final_DR = Qobj(V2.dag().data @ U_eJP_DR.data @ V2.data, dims=DR_dims)
    # otherwise track only basis states of interest
    else:
        # find the states to sum over for fidelity purposes
        op_dict_SR, unique_state_dict_SR = operator_basis_lidar(
            g_comp_basis_states_SR, labels_SR
        )
        op_dict_DR, unique_state_dict_DR = operator_basis_lidar(
            g_comp_basis_states_DR, labels_DR
        )
        # apply the gate to each state, in parallel if desired
        U_eJP_partial = partial(bosonic_sim_SR.U_eJP_func, a, b, params, c_ops)
        final_SR = apply_gate_to_states(U_eJP_partial, unique_state_dict_SR, num_cpus)
        # construct the final dual-rail states from the computed final single rail states. to
        # do this we first reconstruct the final SR ops
        final_SR_ops = operators_from_states(op_dict_SR, final_SR)
        final_DR = {
            label: bosonic_sim_DR.DR_state_from_SR_ops(label, final_SR_ops)
            for label, state in unique_state_dict_DR.items()
        }
    measurement_op_parity = bosonic_sim_DR.measurement_op_DR_parity()
    measurement_op_tmon_DR = tensor(measurement_op_SR, measurement_op_SR)
    measurement_op_DR = measurement_op_parity * measurement_op_tmon_DR
    e_fidel_SR, prob_SR, e_fidel_SR_nops = fidelity_SR.entanglement_fidelity_nielsen(
        final_SR,
        projected_ideal_SR,
        measurement_op=measurement_op_SR,
    )
    e_fidel_DR, prob_DR, e_fidel_DR_nops = fidelity_DR.entanglement_fidelity_nielsen(
        final_DR,
        projected_ideal_DR,
        measurement_op=measurement_op_DR,
    )
    process_fidel_SR = fidelity_SR.process_fidelity_nielsen(e_fidel_SR / prob_SR)
    process_fidel_SR_nops = fidelity_SR.process_fidelity_nielsen(e_fidel_SR_nops)
    process_fidel_DR = fidelity_DR.process_fidelity_nielsen(e_fidel_DR / prob_DR)
    process_fidel_DR_nops = fidelity_DR.process_fidelity_nielsen(e_fidel_DR_nops)
    print(f"entanglement fidelity, success prob single rail: {e_fidel_SR, prob_SR}")
    print(f"postselected process fidelity single rail:  {process_fidel_SR}")
    print(f"unpostselected process fidelity single rail:  {process_fidel_SR_nops}")
    print(f"entanglement fidelity, success prob dual rail: {e_fidel_DR, prob_DR}")
    print(f"postselected process fidelity dual rail:  {process_fidel_DR}")
    print(f"unpostselected process fidelity dual rail:  {process_fidel_DR_nops}")
    data_dict = {
        "e_fidel_SR": e_fidel_SR,
        "e_fidel_SR_nops": e_fidel_SR_nops,
        "e_fidel_DR": e_fidel_DR,
        "e_fidel_DR_nops": e_fidel_DR_nops,
        "prob_SR": prob_SR,
        "prob_DR": prob_DR,
        "process_fidel_SR": process_fidel_SR,
        "process_fidel_SR_nops": process_fidel_SR_nops,
        "process_fidel_DR": process_fidel_DR,
        "process_fidel_DR_nops": process_fidel_DR_nops,
    }
    write_to_h5(filepath, data_dict, param_dict)
