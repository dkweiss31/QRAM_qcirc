from GUEs.main_GUE import main_GUE
from QRAM_utils.utils import generate_file_path
from param_dicts import ideal_param_dict

directory = "out"
# filepath = generate_file_path("hdf5", "GUE_plotting", directory)
# filepath = generate_file_path("qu", "GUE_plotting", directory)
filepath = generate_file_path("h5py", "GUE_dark_violation", directory)
ideal_param_dict["plot"] = True
ideal_param_dict.pop("cavity_dim")
ideal_param_dict["num_exc"] = 3
main_GUE(filepath, ideal_param_dict)
