from qutip import tensor, destroy, Qobj, to_super, qeye
from sim_tools import SimulateBosonicOperations, FidelityBosonicOperations
from utils import (
    id_wrap_ops,
    construct_basis_states_list,
    project_U, generate_file_path,
)
import numpy as np
import h5py


def main(filepath, param_dict):
    tmon_dim = param_dict["tmon_dim"]
    cavity_dim = param_dict["cavity_dim"]
    chi = param_dict["chi"]
    tmon_d_strength = param_dict["tmon_d_strength"]
    params = (tmon_d_strength, chi)
    eta_gg = param_dict["eta_gg"]
    eta_ge = param_dict["eta_ge"]
    eta_gf = param_dict["eta_gf"]
    postselection = param_dict["postselection"]
    cav_a_idx = 0
    cav_b_idx = 1
    tmon_idx = 2
    truncated_dims = [cavity_dim, cavity_dim, tmon_dim]
    a = id_wrap_ops(destroy(cavity_dim), cav_a_idx, truncated_dims)
    b = id_wrap_ops(destroy(cavity_dim), cav_b_idx, truncated_dims)
    bosonic_sim = SimulateBosonicOperations(
        gf_tmon=True, tmon_dim=tmon_dim, cavity_dim=cavity_dim
    )
    # computational basis states include only Fock 0, 1 and transmon 0
    g_Fock_states_spec = [(i, j, 0) for i in range(2) for j in range(2)]
    g_comp_basis_states = construct_basis_states_list(
        g_Fock_states_spec, truncated_dims
    )
    g_comp_basis_states_DR = bosonic_sim.DR_basis(g_comp_basis_states)
    # measurement operator allowing for nonideal measurement
    measurement_op_SR = (
        np.sqrt(eta_gg) * bosonic_sim.measurement_op_tmon_projector(0)
        + np.sqrt(eta_ge) * bosonic_sim.measurement_op_tmon_projector(1)
        + np.sqrt(eta_gf) * bosonic_sim.measurement_op_tmon_projector(2)
    )

    # # ideal case, no dissipation
    U_eJP_ideal_SR = bosonic_sim.U_eJP_func(a, b, params)
    U_eJP_ideal_DR = tensor(U_eJP_ideal_SR, U_eJP_ideal_SR)
    projected_ideal_SR = project_U(U_eJP_ideal_SR, basis_states=g_comp_basis_states)
    projected_ideal_DR = project_U(U_eJP_ideal_DR, basis_states=g_comp_basis_states_DR)

    # # construct the single-rail superoperator
    c_ops = bosonic_sim.construct_c_ops(a, b, **param_dict)
    # maybe here want to actually apply the gate to each state individually. That way easier to tensor together
    # for dual-rail purposes. Doing it at the level of the fidelity function doesn't seem right
    # U_eJP = bosonic_sim.U_eJP_func(a, b, params, c_ops)
    #
    # # for dual rail tensor together the two superoperators then reorder appropriately
    # U_eJP_DR = tensor(U_eJP, U_eJP)
    # DR_dims = to_super(U_eJP_ideal_DR).dims
    # V2 = bosonic_sim.V_2_op()
    # U_eJP_DR_V2 = Qobj(V2.dag().data @ U_eJP_DR.data @ V2.data, dims=DR_dims)

    if postselection:
        measurement_op_super_SR = measurement_op_SR
        measurement_op_super_parity = bosonic_sim.measurement_op_DR_parity()
        measurement_op_super_tmon_DR = tensor(measurement_op_SR, measurement_op_SR)
        measurement_op_DR = measurement_op_super_parity * measurement_op_super_tmon_DR
    else:
        measurement_op_super_SR = qeye(truncated_dims)
        measurement_op_DR = tensor(qeye(truncated_dims), qeye(truncated_dims))
    bosonic_fidel = FidelityBosonicOperations(gf_tmon=True, tmon_dim=tmon_dim, cavity_dim=cavity_dim)

    e_fidel_SR, prob_SR = bosonic_fidel.entanglement_fidelity_nielsen_states(
        "U_eJP_func",
        (a, b, params, c_ops),
        projected_ideal_SR,
        g_comp_basis_states,
        measurement_op=measurement_op_super_SR,
    )
    # e_fidel_DR, prob_DR = bosonic_fidel.entanglement_fidelity_nielsen(
    #     U_eJP_DR_V2,
    #     projected_ideal_DR,
    #     g_comp_basis_states_DR,
    #     measurement_op=measurement_op_DR,
    # )
    print(f"saving run to {filepath}")
    print(f"entanglement fidelity single rail: {e_fidel_SR}")
    print(f"success probability single rail: {prob_SR}")
    # print(f"entanglement fidelity dual rail: {e_fidel_DR}")
    # print(f"success probability dual rail: {prob_DR}")
    with h5py.File(filepath, "w") as f:
        e_fidel_SR = f.create_dataset("e_fidel_SR", data=e_fidel_SR)
        e_fidel_DR = f.create_dataset("e_fidel_DR", data=e_fidel_DR)
        prob_SR = f.create_dataset("prob_SR", data=prob_SR)
        prob_DR = f.create_dataset("prob_DR", data=prob_DR)
        for kwarg in param_dict.keys():
            f.attrs[kwarg] = param_dict[kwarg]


# example call
if __name__ == "__main__":
    directory = "out"
    filepath = generate_file_path("h5py", "entangle_fidel_SR_DR", directory)
    param_dict = {
        "tmon_dim": 3,
        "cavity_dim": 2,
        "chi": 2.0 * np.pi * 0.002,
        "tmon_d_strength": 2.0 * np.pi * 0.01,
        "eta_gg": 0.9999,
        "eta_ge": 0.01,
        "eta_gf": 0.01**2,
        "Gamma_1_ge": 1.0 / (200 * 10**3),
        "Gamma_1_ef": 2.0 / (200 * 10**3),
        "Gamma_phi_gg": 0.0,
        "Gamma_phi_ee": 1.0 / (400 * 10**3),
        "Gamma_phi_ff": 4.0 / (400 * 10**3),
        "Gamma_1_res": 1.0 / (600 * 10**3),
        "Gamma_phi_res": 1.0 / (5000 * 10**3),
        "nth": 0.01,
        "postselection": True,
    }
    main(filepath, param_dict)
