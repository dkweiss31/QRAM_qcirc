import numpy as np

from simulate_GUE import SimulateGUETwoWay
from utils import construct_basis_states_list
from qutip.fileio import qsave


def main(filepath, param_dict):
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
        gamma_a_avg=gamma_a_avg, gamma_a_dev=gamma_a_dev,
        gamma_b_avg=gamma_b_avg, gamma_b_dev=gamma_b_dev,
        gamma_c_avg=gamma_c_avg, gamma_c_dev=gamma_c_dev,
        cavity_dim=cavity_dim)
    zero_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
    assert len(zero_vec) == 12
    Fock_states_spec = 12 * [zero_vec]
    for idx in range(12):
        Fock_vec = Fock_states_spec[idx]
        Fock_vec[idx] = 1
        Fock_states_spec[idx] = tuple(Fock_vec)
    (state_a1,
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
     state_c2_r) = construct_basis_states_list(Fock_states_spec, guefidelitytwoway.truncated_dims)

    psi_init = ((state_b1 - 1j * state_b2).unit() + (state_b1 + 1j * state_b2).unit()).unit()
    psi_fin = ((state_a1 - 1j * state_a2).unit() + (state_c1 + 1j * state_c2).unit()).unit()
    rho_init = psi_init * psi_init.dag()
    rho_fin = psi_fin * psi_fin.dag()

    c_ops = guefidelitytwoway.construct_c_ops(Gamma_1_cav=0.0, Gamma_1_transfer_nr=0.0, nth=0.0)
    e_ops = [rho_init, rho_fin]

    args = {"c": c, "B": B, "t_half": t_half, "gamma_b_avg": gamma_b_avg,
            "scale_a": scale_a, "scale_b": scale_b, "scale_c": scale_c}
    result = guefidelitytwoway.run_state_transfer(rho_init, args, c_ops=c_ops, e_ops=e_ops)
    fidel = (psi_fin.dag() * result.final_state * psi_fin).data.toarray()[0, 0]
    print(f"fidelity of state transfer is {fidel}")
    qsave(result, filepath)
