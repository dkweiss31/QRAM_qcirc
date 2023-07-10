from main_Ramsey_Yao_two import main_ramsey_two
from utils.utils import generate_file_path

directory = "out"
cavity_dim = 6
nsteps = 2500
interference = True
filepath = generate_file_path(
    "hdf5", f"Ramsey_cav_{cavity_dim}_interfer_{interference}", directory
)
param_dict = {
    "cavity_dim": cavity_dim,
    "nsteps": nsteps,
    "interference": interference,
}
main_ramsey_two(filepath, param_dict)
