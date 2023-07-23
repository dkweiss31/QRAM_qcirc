import numpy as np

from main_SR_DR import main_SR_DR
from param_dicts import ideal_param_dict
from utils.utils import generate_file_path

Tphi_list = np.array([2.6 ** i for i in range(0, 10)]) * 2 * 10**4

for Tphi in Tphi_list:
    filepath = generate_file_path("h5py", "fidel_Tphi_cav", "out")
    ideal_param_dict["Gamma_phi_res"] = 1.0 / Tphi
    main_SR_DR(filepath, ideal_param_dict)
