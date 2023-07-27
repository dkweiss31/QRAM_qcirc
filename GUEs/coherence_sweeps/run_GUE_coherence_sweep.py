import argparse
import sys

import numpy as np

sys.path.append("/gpfs/gibbs/project/puri/dkw34/qram_fidelity/QRAM_qcirc/")
from GUEs.main_GUE_two_way import main_GUE_two_way
from GUEs.param_dicts import ideal_param_dict_two_way


def run_GUE_coherence(param_key, idx, num_pts, base, prefactor, nth, num_cpus=8):
    T_list = np.array([base ** i for i in range(0, num_pts+1)]) * prefactor
    ideal_param_dict_two_way["nth"] = nth
    ideal_param_dict_two_way["num_cpus"] = num_cpus
    filepath = f"out/fidel_GUE_coherence_{param_key}_{str(idx).zfill(5)}.h5py"
    ideal_param_dict_two_way[param_key] = 1.0 / T_list[idx]
    main_GUE_two_way(filepath, ideal_param_dict_two_way)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUE coherence sweep")
    parser.add_argument(
        "--param_key", default="None", type=str, help="parameter to sweep over"
    )
    parser.add_argument("--idx", default=0, type=int, help="index of the sweep")
    parser.add_argument("--num_pts", default=10, type=int, help="number of points in the sweep")
    parser.add_argument("--base", default=2.0, type=float, help="base of the sweep")
    parser.add_argument(
        "--prefactor", default=2.0 * 10**4, type=float, help="prefactor of the sweep"
    )
    parser.add_argument("--nth", default=0.0, type=float, help="thermal population")
    parser.add_argument("--num_cpus", default=8, type=int, help="num cpus")
    args = parser.parse_args()
    run_GUE_coherence(
        args.param_key, args.idx, args.num_pts, args.base, args.prefactor, args.nth, num_cpus=args.num_cpus
    )
