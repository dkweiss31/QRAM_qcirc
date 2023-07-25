import numpy as np

from GUEs.main_GUE import main_GUE
from GUEs.param_dicts import ideal_param_dict
from utils.utils import generate_file_path

Tphi_list = np.array([100 * 2.15 ** i for i in range(0, 10)])


def run_GUE_transfer_Tphi():
    for Tphi in Tphi_list:
        filepath = generate_file_path("h5py", "fidel_Tphi_transfer", "out")
        ideal_param_dict["Gamma_phi_transfer"] = 1.0 / Tphi
        ideal_param_dict["num_cpus"] = 8
        main_GUE(filepath, ideal_param_dict)


if __name__ == "__main__":
    run_GUE_transfer_Tphi()
