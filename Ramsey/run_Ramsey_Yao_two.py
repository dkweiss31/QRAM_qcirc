import numpy as np

from Ramsey.ramsey_Yao import RamseyExperiment, CoherentDephasing
from utils.utils import generate_file_path

run_coherent = False
run_thermal = True
directory = "out"
omega_a = 2.0 * np.pi * 3.3261
omega_b = 2.0 * np.pi * 3.4712
omega_cavs = np.array([omega_a, omega_b])
kappa_a = 2.0 * np.pi * 0.04268
kappa_b = 2.0 * np.pi * 0.04784
kappa_cavs = np.array([kappa_a, kappa_b])
chiaq = -2.0 * np.pi * 0.000322
chibq = -2.0 * np.pi * 0.000571
chi_cavstmon = np.array([chiaq, chibq])
param_dict = {
    "cavity_dim": 4,
    "num_cavs": 2,
    "nsteps": 100000,
    "atol": 1e-10,
    "rtol": 1e-10,
    "interference": True,
    "tmon_dim": 2,
    "delay_times": np.linspace(0.0, 2000.0, 301),
    "omega_cavs": omega_cavs,
    "kappa_cavs": kappa_cavs,
    "omega_tmon": 2.0 * np.pi * 5.7423,
    "omega_d_tmon": 2.0 * np.pi * (5.7423 - 0.0071),
    "chi_cavstmon": chi_cavstmon,
    "temp": 0.1,
}
if run_thermal:
    filepath = generate_file_path(
        "hdf5", f"Ramsey_cav_{param_dict['cavity_dim']}_interfer_{param_dict['interference']}", directory
    )
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_experiment_two = RamseyExperiment(**param_dict)
    ramsey_experiment_two.main_ramsey(filepath, p0)

if run_coherent:
    omega_d_cav = 2.0 * np.pi * 3.1
    epsilon_array = 2.0 * np.pi * np.array([0.001, 0.001])
    param_dict["omega_d_cav"] = omega_d_cav
    param_dict["epsilon_array"] = epsilon_array
    ramsey_experiment_two_cohere = CoherentDephasing(**param_dict)
    filepath = generate_file_path(
        "hdf5", f"cohere_Ramsey_cav_{param_dict['cavity_dim']}_interfer_{param_dict['interference']}", directory
    )
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_experiment_two_cohere.main_ramsey(filepath, p0)

