from GUEs.main_GUE import main_GUE
from param_dicts import param_dict_2
from utils.utils import generate_file_path

directory = "GUEs/out"
filepath = generate_file_path("hdf5", "fidel_param_2", directory)
param_dict_2["additional_label"] = False

main_GUE(filepath, param_dict_2)
