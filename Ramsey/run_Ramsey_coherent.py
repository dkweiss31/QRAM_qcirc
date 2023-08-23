import argparse
import sys

sys.path.append("/gpfs/gibbs/project/puri/dkw34/qram_fidelity/QRAM_qcirc/")
sys.path.append("/Users/danielweiss/PycharmProjects/QRAM_qcirc")
import numpy as np
from Ramsey.ramsey_Yao import CoherentDephasing
from Ramsey.param_dicts import param_dict_1, param_dict_2


def run_coherent(idx, num_pts, eps, cav_dim, num_cavs=1, nsteps=200000, thermal_time=200.0):
    omega_d_list = 2.0 * np.pi * np.linspace(3.0, 3.7, num_pts)
    omega_d_cav = omega_d_list[idx]
    if num_cavs == 1:
        param_dict = param_dict_1
        epsilon_array = 2.0 * np.pi * np.array([eps, ])
    elif num_cavs == 2:
        param_dict = param_dict_2
        epsilon_array = 2.0 * np.pi * np.array([eps, eps])
    else:
        raise RuntimeError("only one or two cavities supported")
    param_dict["omega_d_cav"] = omega_d_cav
    param_dict["epsilon_array"] = epsilon_array
    param_dict["cavity_dim"] = cav_dim
    param_dict["nsteps"] = nsteps
    param_dict["thermal_time"] = thermal_time
    ramsey_coherent = CoherentDephasing(**param_dict)
    filepath = (f"out/{str(idx).zfill(5)}_numcav_{param_dict['num_cavs']}"
                + f"_cavdim_{param_dict['cavity_dim']}_eps_{eps}_cohere_Ramsey.h5py")
    p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
    ramsey_coherent.main_ramsey(filepath, p0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ramsey Coherent num cavs 1")
    parser.add_argument("--idx", default=0, type=int, help="index that is unraveled")
    parser.add_argument(
        "--num_pts", default=5, type=int, help="number of points in param list"
    )
    parser.add_argument("--eps", default=0.0, type=float, help="drive strength in GHz")
    parser.add_argument("--cav_dim", default=6, type=int, help="cavity dimension")
    parser.add_argument("--num_cavs", default=1, type=int, help="number of cavities")
    parser.add_argument("--nsteps", default=200000, type=int, help="nsteps for mesolve")
    parser.add_argument("--thermal_time", default=200.0, type=float, help="time spent thermalizing")
    args = parser.parse_args()
    run_coherent(
        args.idx,
        args.num_pts,
        args.eps,
        args.cav_dim,
        num_cavs=args.num_cavs,
        nsteps=args.nsteps,
        thermal_time=args.thermal_time,
    )
