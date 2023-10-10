import numpy as np

from GUEs.main_GUE import main_GUE
from GUEs.simulate_GUE import SimulateGUE, SimulateGUEHashing
from GUEs.simulate_GUE import SimulateGUEDR, SimulateGUEHashingDR
from GUEs.simulate_GUE import SimulateGUEHashingOptControl
from QRAM_utils.utils import generate_file_path
from param_dicts import param_dict_2

param_dict = param_dict_2
hashing = True
opt_control = False
control_file_location = "/Users/danielweiss/PycharmProjects/QRAM_qcirc/GUEs/notebooks/qram_v10.h5"
directory = "out"
filepath = generate_file_path("h5py", "GUE_opt_tst", directory)
param_dict["cavity_dim"] = 2
if opt_control:
    param_dict.pop("cavity_dim")
    param_dict["num_exc"] = 3
    param_dict["control_file_location"] = control_file_location
    param_dict["phi"] = np.pi / 2
    gue = SimulateGUEHashingOptControl(**param_dict)
    param_dict.pop("control_file_location")
    gue_DR = SimulateGUEHashingDR(**param_dict)
elif hashing:
    param_dict.pop("cavity_dim")
    param_dict["num_exc"] = 3
    param_dict["phi"] = -np.pi / 2
    gue = SimulateGUEHashing(**param_dict)
    gue_DR = SimulateGUEHashingDR(**param_dict)
else:
    param_dict["phi"] = -np.pi / 2
    gue = SimulateGUE(**param_dict)
    gue_DR = SimulateGUEDR(**param_dict)
main_GUE(filepath, gue, gue_DR, param_dict)
