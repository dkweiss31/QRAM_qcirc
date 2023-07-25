from GUEs.main_GUE_two_way import main_GUE_two_way
from param_dicts import param_dict_2_two_way
from utils.utils import generate_file_path

directory = "GUEs/out"
filepath = generate_file_path("hdf5", "fidel_two_way_param_2", directory)

main_GUE_two_way(filepath, param_dict_2_two_way)
