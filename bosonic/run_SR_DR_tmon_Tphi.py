import numpy as np

from main_SR_DR import main
from param_dicts import ideal_param_dict
from utils.utils import generate_file_path

Tphi_list = np.array([2.33 ** i for i in range(0, 10)]) * 10**3

# sweep tmon Tphi
for Tphi in Tphi_list:
    filepath = generate_file_path("h5py", "fidel_Tphi_tmon", "out")
    ideal_param_dict["Gamma_phi_ee"] = 1.0 / Tphi
    ideal_param_dict["Gamma_phi_ff"] = 4.0 / Tphi
    main(filepath, ideal_param_dict)
