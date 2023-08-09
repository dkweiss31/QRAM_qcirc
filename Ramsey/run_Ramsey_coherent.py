import argparse
import sys

sys.path.append("/gpfs/gibbs/project/puri/dkw34/qram_fidelity/QRAM_qcirc/")
sys.path.append("/Users/danielweiss/PycharmProjects/QRAM_qcirc")
import numpy as np
from Ramsey.ramsey_Yao import CoherentDephasing
from param_dicts import param_dict_1


def run_coherent(idx, num_pts, eps):
    omega_d_list = np.linspace(2.0, 4.5, num_pts)
    omega_d_cav = omega_d_list[idx]
    epsilon_array = 2.0 * np.pi * np.array([eps, ])
    param_dict_1["omega_d_cav"] = omega_d_cav
    param_dict_1["epsilon_array"] = epsilon_array
    ramsey_experiment_one_cohere = CoherentDephasing(**param_dict_1)
    filepath = (f"out/{str(idx).zfill(5)}_numcav_{param_dict_1['num_cavs']}"
                + f"_cavdim_{param_dict_1['cavity_dim']}_eps_{eps}_cohere_Ramsey.h5py")
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_experiment_one_cohere.main_ramsey(filepath, p0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ramsey Coherent num cavs 1")
    parser.add_argument("--idx", default=0, type=int, help="index that is unraveled")
    parser.add_argument(
        "--num_pts", default=5, type=int, help="number of points in param list"
    )
    parser.add_argument("--eps", default=0.0, type=float, help="drive strength in GHz")
    args = parser.parse_args()
    run_coherent(args.idx, args.num_pts, args.eps)
