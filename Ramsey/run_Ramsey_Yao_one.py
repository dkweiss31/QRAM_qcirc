import numpy as np

from QRAM_utils.utils import generate_file_path
from Ramsey.ramsey_Yao import RamseyExperiment, CoherentDephasing
from param_dicts import param_dict_1

run_coherent = True
run_thermal = True
directory = "Ramsey/out"
param_dict_1["cavity_dim"] = 10
param_dict_1["nsteps"] = 200000
if run_thermal:
    filepath = generate_file_path(
        "hdf5", f"Ramsey_onecav_dim_{param_dict_1['cavity_dim']}", directory
    )
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_experiment_two = RamseyExperiment(**param_dict_1)
    ramsey_experiment_two.main_ramsey(filepath, p0)

if run_coherent:
    omega_d_cav = 2.0 * np.pi * 3.35
    epsilon_array = 2.0 * np.pi * np.array([0.005, ])
    param_dict_1["omega_d_cav"] = omega_d_cav
    param_dict_1["epsilon_array"] = epsilon_array
    ramsey_experiment_two_cohere = CoherentDephasing(**param_dict_1)
    filepath = generate_file_path(
        "hdf5", f"coherent_onecav_dim_{param_dict_1['cavity_dim']}", directory
    )
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_experiment_two_cohere.main_ramsey(filepath, p0)
