import numpy as np

from GUEs.main_GUE import main_GUE
from GUEs.param_dicts import ideal_param_dict
from utils.utils import generate_file_path

T1_list = np.array([2.6 ** i for i in range(0, 10)]) * 2 * 10**4
ideal_param_dict["nth"] = 0.01
ideal_param_dict["additional_label"] = False


def run_GUE_cav_T1():
    for T1 in T1_list:
        filepath = generate_file_path("h5py", "fidel_T1_cav", "coherence_sweeps/out")
        ideal_param_dict["Gamma_1_cav"] = 1.0 / T1
        main_GUE(filepath, ideal_param_dict)


if __name__ == "__main__":
    run_GUE_cav_T1()
