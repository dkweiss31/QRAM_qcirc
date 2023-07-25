import numpy as np

from GUEs.main_GUE import main_GUE
from GUEs.param_dicts import ideal_param_dict
from utils.utils import generate_file_path

T1_list = np.array([100 * 2.15 ** i for i in range(0, 10)])
ideal_param_dict["nth"] = 0.01


def run_GUE_transfer_T1():
    for T1 in T1_list:
        filepath = generate_file_path("h5py", "fidel_T1_transfer", "out")
        ideal_param_dict["Gamma_1_transfer_nr"] = 1.0 / T1
        ideal_param_dict["num_cpus"] = 8
        main_GUE(filepath, ideal_param_dict)


if __name__ == "__main__":
    run_GUE_transfer_T1()
