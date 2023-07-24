import numpy as np

cav_idx_dict = {"b1_idx": 0, "b2_idx": 1, "c1_idx": 2, "c2_idx": 3}
tran_res_idx_dict = {"b1_r_idx": 4, "b2_r_idx": 5, "c1_r_idx": 6, "c2_r_idx": 7}
ideal_param_dict = {
    "cavity_dim": 2,
    "gamma_b_avg": 2.0 * np.pi * 0.02,
    "gamma_c_avg": 2.0 * np.pi * 0.02,
    "gamma_b_dev": 2.0 * np.pi * 0.0,
    "gamma_c_dev": 2.0 * np.pi * 0.0,
    "cav_idx_dict": cav_idx_dict,
    "tran_res_idx_dict": tran_res_idx_dict,
    "scale_b": 1.018,
    "scale_c": 1.017,
    "t_half": 600,
    "B": 0.006,
    "c": 2.8284e-5,
    "num_cpus": 8,
    "Gamma_1_cav": 0.0,
    "Gamma_1_transfer_nr": 0.0,
    "Gamma_phi_cav": 0.0,
    "Gamma_phi_transfer": 0.0,
    "nth": 0.0,
    "additional_label": True,
    "nsteps": 3000,
    "atol": 1e-12,
    "rtol": 1e-12,
}

param_dict_1 = {
    "cavity_dim": 2,
    "gamma_b_avg": 2.0 * np.pi * 0.02,
    "gamma_c_avg": 2.0 * np.pi * 0.02,
    "gamma_b_dev": 2.0 * np.pi * 0.0,
    "gamma_c_dev": 2.0 * np.pi * 0.0,
    "cav_idx_dict": cav_idx_dict,
    "tran_res_idx_dict": tran_res_idx_dict,
    "scale_b": 1.018,
    "scale_c": 1.017,
    "t_half": 600,
    "B": 0.006,
    "c": 2.8284e-5,
    "num_cpus": 8,
    "Gamma_1_cav": 1.0 / (600 * 10 ** 3),
    "Gamma_1_transfer_nr": 1.0 / (100 * 10**3),
    "Gamma_phi_cav": 1.0 / (5000 * 10 ** 3),  #
    "Gamma_phi_transfer": 1.0 / (100 * 10**3),
    "nth": 0.01,
    "additional_label": True,
    "nsteps": 3000,
    "atol": 1e-12,
    "rtol": 1e-12,
}

T1_res = 25000 * 10**3
T2_res = 34000 * 10**3
Tphi_res = ((1 / T2_res) - (1 / (2 * T1_res))) ** (-1)
param_dict_2 = {
    "cavity_dim": 2,
    "gamma_b_avg": 2.0 * np.pi * 0.02,
    "gamma_c_avg": 2.0 * np.pi * 0.02,
    "gamma_b_dev": 2.0 * np.pi * 0.0,
    "gamma_c_dev": 2.0 * np.pi * 0.0,
    "cav_idx_dict": cav_idx_dict,
    "tran_res_idx_dict": tran_res_idx_dict,
    "scale_b": 1.018,
    "scale_c": 1.017,
    "t_half": 600,
    "B": 0.006,
    "c": 2.8284e-5,
    "num_cpus": 8,
    "Gamma_1_cav": 1.0 / T1_res,
    "Gamma_1_transfer_nr": 1.0 / (200 * 10**3),
    "Gamma_phi_cav": 1.0 / Tphi_res,
    "Gamma_phi_transfer": 1.0 / (200 * 10**3),
    "nth": 0.01,
    "additional_label": True,
    "nsteps": 3000,
    "atol": 1e-12,
    "rtol": 1e-12,
}