import numpy as np

ideal_param_dict = {
    "tmon_dim": 3,
    "cavity_dim": 4,
    "chi": 2.0 * np.pi * 0.002,
    "tmon_d_strength": 2.0 * np.pi * 0.01,
    "eta_gg": 1.0,
    "eta_ge": 0.0,
    "eta_gf": 0.0,
    "Gamma_1_ge": 0.0,
    "Gamma_1_ef": 0.0,
    "Gamma_phi_gg": 0.0,
    "Gamma_phi_ee": 0.0,
    "Gamma_phi_ff": 0.0,
    "Gamma_1_res": 0.0,
    "Gamma_phi_res": 0.0,
    "nth": 0.00,
    "num_cpus": 8,
    "liouvillian": False,
    "atol": 1e-10,
    "rtol": 1e-10,
    "nsteps": 2000,
}

param_dict_1 = {
    "tmon_dim": 3,
    "cavity_dim": 4,
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
    "num_cpus": 8,
    "liouvillian": False,
    "atol": 1e-10,
    "rtol": 1e-10,
    "nsteps": 2000,
}

T1_res = 25000 * 10**3
T2_res = 34000 * 10**3
Tphi_res = ((1 / T2_res) - (1 / (2 * T1_res))) ** (-1)
param_dict_2 = {
    "tmon_dim": 3,
    "cavity_dim": 4,
    "chi": 2.0 * np.pi * 0.002,
    "tmon_d_strength": 2.0 * np.pi * 0.01,
    "eta_gg": 0.9999,
    "eta_ge": 0.01,
    "eta_gf": 0.01**2,
    "Gamma_1_ge": 1.0 / (400 * 10**3),
    "Gamma_1_ef": 2.0 / (400 * 10**3),
    "Gamma_phi_gg": 0.0,
    "Gamma_phi_ee": 1.0 / (900 * 10**3),
    "Gamma_phi_ff": 4.0 / (900 * 10**3),
    "Gamma_1_res": 1.0 / T1_res,
    "Gamma_phi_res": 1.0 / Tphi_res,
    "nth": 0.01,
    "num_cpus": 8,
    "liouvillian": False,
    "atol": 1e-10,
    "rtol": 1e-10,
    "nsteps": 2000,
}

param_dict_3 = {
    "tmon_dim": 3,
    "cavity_dim": 4,
    "chi": 2.0 * np.pi * 0.002,
    "tmon_d_strength": 2.0 * np.pi * 0.01,
    "eta_gg": 0.9999,
    "eta_ge": 0.01,
    "eta_gf": 0.01**2,
    "Gamma_1_ge": 1.0 / (2000 * 10**3),
    "Gamma_1_ef": 2.0 / (2000 * 10**3),
    "Gamma_phi_gg": 0.0,
    "Gamma_phi_ee": 1.0 / (4000 * 10**3),
    "Gamma_phi_ff": 4.0 / (4000 * 10**3),
    "Gamma_1_res": 1.0 / T1_res,
    "Gamma_phi_res": 1.0 / Tphi_res,
    "nth": 0.01,
    "num_cpus": 8,
    "liouvillian": False,
    "atol": 1e-10,
    "rtol": 1e-10,
    "nsteps": 2000,
}