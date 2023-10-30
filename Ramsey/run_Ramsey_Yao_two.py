import numpy as np

from QRAM_utils.utils import generate_file_path
from Ramsey.ramsey_Yao import RamseyExperiment, CoherentDephasing
from param_dicts import param_dict_2

run_coherent = True
run_thermal = False
directory = "Ramsey/out"
param_dict_2["cavity_dim"] = 4
if run_thermal:
    filepath = generate_file_path(
        "hdf5", f"Ramsey_cav_{param_dict_2['cavity_dim']}_interfer_{param_dict_2['interference']}", directory
    )
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_experiment_two = RamseyExperiment(**param_dict_2)
    ramsey_experiment_two.main_ramsey(filepath, p0)

if run_coherent:
    omega_d_cav = 2.0 * np.pi * 3.38
    epsilon_array = 2.0 * np.pi * np.array([0.01, 0.01])
    param_dict_2["omega_d_cav"] = omega_d_cav
    param_dict_2["epsilon_array"] = epsilon_array
    param_dict_2["temp"] = 1e-8
    ramsey_experiment_two_cohere = CoherentDephasing(**param_dict_2)
    filepath = generate_file_path(
        "hdf5", f"cohere_Ramsey_cav_{param_dict_2['cavity_dim']}_interfer_{param_dict_2['interference']}", directory
    )
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_experiment_two_cohere.main_ramsey(filepath, p0)

