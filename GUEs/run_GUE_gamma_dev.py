import argparse
import sys

import numpy as np

sys.path.append("/gpfs/gibbs/project/puri/dkw34/qram_fidelity/QRAM_qcirc/")
from GUEs.main_GUE_two_way import main_GUE_two_way
from GUEs.param_dicts import param_dict_2
from utils.utils import generate_file_path


def run_gamma_dev(dev_amount, idx, num_pts, num_cpus=8):
    gamma_dev_list = 2.0 * np.pi * np.linspace(-dev_amount, dev_amount, num_pts)
    row_idx, col_idx = np.unravel_index(idx, (num_pts, num_pts))
    filepath = generate_file_path(
        "h5py",
        f"GUE_two_way_gamma_idx_{str(idx).zfill(5)}_dev_{dev_amount}_pts_{num_pts}",
        "out",
    )
    param_dict_2["gamma_b_dev"] = gamma_dev_list[row_idx]
    param_dict_2["gamma_c_dev"] = gamma_dev_list[col_idx]
    param_dict_2["num_cpus"] = num_cpus
    main_GUE_two_way(filepath, param_dict_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUE gamma_dev")
    parser.add_argument(
        "--dev_amount", default="0.0", type=str, help="gamma deviation in GHz"
    )
    parser.add_argument("--idx", default=0, type=int, help="index that is unraveled")
    parser.add_argument(
        "--num_pts", default=5, type=int, help="number of points in gamma_dev_list"
    )
    parser.add_argument("--num_cpus", default=8, type=int, help="num cpus")
    args = parser.parse_args()
    run_gamma_dev(
        float(args.dev_amount), args.idx, args.num_pts, num_cpus=args.num_cpus
    )
