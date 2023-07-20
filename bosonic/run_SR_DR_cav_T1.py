import numpy as np

from main_SR_DR import main
from param_dicts import ideal_param_dict
from utils.utils import generate_file_path

T1_list = np.array([2.6 ** i for i in range(0, 10)]) * 2 * 10**4

ideal_param_dict["nth"] = 0.01
for T1 in T1_list:
    filepath = generate_file_path("h5py", "fidel_T1_cav", "out")
    ideal_param_dict["Gamma_1_res"] = 1.0 / T1
    main(filepath, ideal_param_dict)
