import numpy as np
from fidelity_SR_DR import main
from utils import generate_file_path

directory = "out"
filepath = generate_file_path("h5py", "entangle_fidel_SR_DR", directory)
param_dict = {
    "tmon_dim": 3,
    "cavity_dim": 2,
    "control_dt": 4.0,
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
    "liouvillian": False
}
main(filepath, param_dict)
