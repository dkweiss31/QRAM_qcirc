from QRAM_utils.utils import generate_file_path
from main_SR_DR import main_SR_DR
from param_dicts import param_dict_1

filepath = generate_file_path("h5py", "fidel_SR_DR_param_dict_1", "out")
main_SR_DR(filepath, param_dict_1)
