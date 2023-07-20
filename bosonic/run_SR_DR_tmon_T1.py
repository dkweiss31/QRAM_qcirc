import numpy as np

from main_SR_DR import main
from param_dicts import ideal_param_dict
from utils.utils import generate_file_path

T1_list = np.array([2.33 ** i for i in range(0, 10)]) * 10**3

# sweep tmon T1
for T1 in T1_list:
    filepath = generate_file_path("h5py", "fidel_T1_tmon", "out")
    ideal_param_dict["Gamma_1_ge"] = 1.0 / T1
    ideal_param_dict["Gamma_1_ef"] = 2.0 / T1
    main(filepath, ideal_param_dict)
