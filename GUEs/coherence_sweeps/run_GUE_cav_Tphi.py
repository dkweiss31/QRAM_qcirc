import numpy as np

from GUEs.main_GUE import main_GUE
from GUEs.param_dicts import ideal_param_dict
from utils.utils import generate_file_path

Tphi_list = np.array([2.6 ** i for i in range(0, 10)]) * 2 * 10**4


def run_GUE_cav_Tphi():
    for Tphi in Tphi_list:
        filepath = generate_file_path("h5py", "fidel_Tphi_cav", "out")
        ideal_param_dict["Gamma_phi_cav"] = 1.0 / Tphi
        ideal_param_dict["num_cpus"] = 8
        main_GUE(filepath, ideal_param_dict)
