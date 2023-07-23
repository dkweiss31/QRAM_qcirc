from GUEs.main_GUE import main_GUE
from utils.utils import generate_file_path

directory = "GUEs/out"
filepath = generate_file_path("hdf5", "fidel_param_1", directory)

main_GUE(filepath, param_dict)