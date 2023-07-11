import copy

import numpy as np

from main_SR_DR import main
from utils.utils import generate_file_path

directory = "out"
filepath = generate_file_path("h5py", "entangle_fidel_SR_DR", directory)
param_dict_1 = {
    "tmon_dim": 3,
    "cavity_dim": 5,
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
    "num_cpus": 8,
    "liouvillian": False,
    "atol": 1e-10,
    "rtol": 1e-10,
    "nsteps": 2000,
}
main(filepath, param_dict_1)

# run for better params
filepath = generate_file_path("h5py", "entangle_fidel_SR_DR", directory)
T1_res = 25000 * 10**3
T2_res = 34000 * 10**3
Tphi_res = ((1 / T2_res) - (1 / (2 * T1_res))) ** (-1)
param_dict_2 = copy.deepcopy(param_dict_1)
param_dict_2["Gamma_1_ge"] = 1.0 / (500 * 10**3)
param_dict_2["Gamma_1_ef"] = 2.0 / (500 * 10 ** 3)
param_dict_2["Gamma_phi_ee"] = 1.0 / (900 * 10**3)
param_dict_2["Gamma_phi_ff"] = 4.0 / (900 * 10**3)
param_dict_2["Gamma_1_res"] = 1.0 / T1_res
param_dict_2["Gamma_phi_res"] = 1.0 / Tphi_res

main(filepath, param_dict_2)
