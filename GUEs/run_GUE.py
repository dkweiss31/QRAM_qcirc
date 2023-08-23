from GUEs.main_GUE import main_GUE
from GUEs.simulate_GUE import SimulateGUE, SimulateGUEHashing
from GUEs.simulate_GUE import SimulateGUEDR, SimulateGUEHashingDR
from QRAM_utils.utils import generate_file_path
from param_dicts import param_dict_2

param_dict = param_dict_2
hashing = True
directory = "out"
filepath = generate_file_path("h5py", "GUE_tst", directory)
param_dict["cavity_dim"] = 2
if hashing:
    param_dict.pop("cavity_dim")
    param_dict["num_exc"] = 3
    gue = SimulateGUEHashing(**param_dict)
    gue_DR = SimulateGUEHashingDR(**param_dict)
else:
    gue = SimulateGUE(**param_dict)
    gue_DR = SimulateGUEDR(**param_dict)
main_GUE(filepath, gue, gue_DR, param_dict)
