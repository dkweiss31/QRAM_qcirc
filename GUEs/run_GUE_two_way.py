from GUEs.main_GUE_two_way import main_GUE_two_way
from GUEs.simulate_GUE import SimulateGUETwoWay, SimulateGUETwoWayDR
from QRAM_utils.utils import generate_file_path
from param_dicts import param_dict_2_two_way

param_dict = param_dict_2_two_way  # ideal_param_dict_two_way
param_dict["num_exc"] = 2
directory = "out"
filepath = generate_file_path("h5py", "GUE_two_way_ps1", directory)
gue = SimulateGUETwoWay(**param_dict)
gue_DR = SimulateGUETwoWayDR(**param_dict)
main_GUE_two_way(filepath, gue, gue_DR, param_dict)
