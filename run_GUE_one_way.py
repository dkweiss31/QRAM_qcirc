import numpy as np

from main_GUE import main_one_way
from utils import generate_file_path

directory = "out"
filepath = generate_file_path("hdf5", "GUE_state_transfer_one_way", directory)
param_dict = {
    "gamma_b_avg": 2.0 * np.pi * 0.02,
    "gamma_c_avg": 2.0 * np.pi * 0.02,
    "gamma_b_dev": 2.0 * np.pi * 0.0,
    "gamma_c_dev": 2.0 * np.pi * 0.0,
    "scale_b": 1.018,
    "scale_c": 1.017,
    "t_half": 600,
    "B": 0.006,
    "c": 2.8284e-5,
    "num_cpus": 8,
    "Gamma_1_cav": 1.0 / (25 * 10**6),
    "Gamma_1_transfer_nr": 1.0 / (100 * 10**3),
    "nth": 0.01,
}
main_one_way(filepath, param_dict)
