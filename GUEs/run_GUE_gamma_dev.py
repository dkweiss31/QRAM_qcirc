import numpy as np

from GUEs.main_GUE import main_GUE
from GUEs.param_dicts import param_dict_2
from utils.utils import generate_file_path


def run_gamma_dev(dev_amount, idx, num_pts):
    gamma_dev_list = 2.0 * np.pi * np.linspace(-dev_amount, dev_amount, num_pts)
    row_idx, col_idx = np.unravel_index(idx, (num_pts, num_pts))
    filepath = generate_file_path("h5py", f"GUE_gamma_dev_{dev_amount}_pts_{num_pts}_row_{row_idx}_col_{col_idx}", "out")
    param_dict_2["gamma_b_dev"] = gamma_dev_list[row_idx]
    param_dict_2["gamma_c_dev"] = gamma_dev_list[col_idx]
    param_dict_2["num_cpus"] = 8
    main_GUE(filepath, param_dict_2)


if __name__ == "__main__":
    run_gamma_dev(0.03, 100, 21)
