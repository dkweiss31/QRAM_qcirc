import numpy as np
from main_GUE import main
from utils import generate_file_path

directory = "out"
filepath = generate_file_path("", "GUE_state_transfer", directory)
param_dict = {
    "gamma_a_avg": 2.0 * np.pi * 0.02,
    "gamma_b_avg": 2.0 * np.pi * 0.02,
    "gamma_c_avg": 2.0 * np.pi * 0.02,
    "gamma_a_dev": 2.0 * np.pi * 0.0,
    "gamma_b_dev": 2.0 * np.pi * 0.0,
    "gamma_c_dev": 2.0 * np.pi * 0.0,
    "scale_a": 1.017,
    "scale_b": 1.018,
    "scale_c": 1.017,
    "t_half": 600,
    "B": 0.006,
    "c": 2.8284e-5
}
main(filepath, param_dict)
