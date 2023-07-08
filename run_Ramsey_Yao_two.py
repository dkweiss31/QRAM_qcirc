from main_Ramsey_Yao_two import main_ramsey_two
from utils import generate_file_path

directory = "out"
cavity_dim = 4
control_dt = 0.1
interference = True
filepath = generate_file_path("hdf5", f"Ramsey_cav_{cavity_dim}_controldt_{control_dt}_interfer_{interference}",
                              directory)
param_dict = {
    "cavity_dim": cavity_dim,
    "control_dt": control_dt,
    "interference": interference,
}
main_ramsey_two(filepath, param_dict)
