import argparse
import sys

from quantum_utils import generate_file_path

sys.path.append("/gpfs/gibbs/project/puri/dkw34/qram_fidelity/QRAM_qcirc/")
sys.path.append("/Users/danielweiss/PycharmProjects/QRAM_qcirc")
import numpy as np
from Ramsey.ramsey_Yao import CoherentDephasing
from Ramsey.param_dicts import param_dict_1, param_dict_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ramsey Coherent num cavs 1")
    parser.add_argument("--idx", default=-1, type=int, help="index that is unraveled")
    parser.add_argument(
        "--num_pts", default=1, type=int, help="number of points in omega_d list"
    )
    parser.add_argument("--exp_type", default="ramsey", type=str, help="experiment type")
    parser.add_argument("--interference", default=1, type=int, help="interference between the two modes")
    parser.add_argument("--omega_d_vals", default="(3.3261,3.3261)", type=str, help="omega_d endpoint")
    parser.add_argument("--eps", default=0.01, type=float, help="drive strength in GHz")
    parser.add_argument("--cav_dim", default=7, type=int, help="cavity dimension")
    parser.add_argument("--num_cavs", default=1, type=int, help="number of cavities")
    # parser.add_argument("--delay_times", default="(0,32000,4801)", type=str, help="delay times to scan over")
    parser.add_argument("--delay_times", default="(0,2000,301)", type=str, help="delay times to scan over")
    parser.add_argument("--nsteps", default=200000, type=int, help="nsteps for mesolve")
    parser.add_argument("--temp", default=1e-6, type=float, help="temperature")
    parser.add_argument("--destructive_interference", default=-1, type=complex, help="destructive interference")
    parser.add_argument("--interference_scale", default=1, type=int, help="scale of the chi_ab term")
    parser.add_argument("--include_stark_shifts", default=0, type=int, help="include stark shifts")
    parser.add_argument("--thermal_time", default=1000.0, type=float, help="time spent thermalizing")
    parser.add_argument("--full_cosine", default=False, type=bool, help="full cosine or not")
    args = parser.parse_args()
    if args.idx != -1:
        filepath = f"out/{str(args.idx).zfill(5)}_cohere_Ramsey.h5py"
    else:
        filepath = generate_file_path("h5py", f"cohere_Ramsey", "out")
    if args.num_cavs == 1:
        param_dict = param_dict_1
        epsilon_array = 2.0 * np.pi * np.array([args.eps, ])
    elif args.num_cavs == 2:
        param_dict = param_dict_2
        # kappas = param_dict["kappa_cavs"]
        # kappa_a, kappa_b = kappas[0], kappas[1]
        # u = np.sqrt(kappa_a / (kappa_a + kappa_b))
        # v = np.sqrt(1 - u ** 2)
        # epsilon_array = 2.0 * np.pi * np.array([u * args.eps, v * args.eps])
        P = (2.0 * np.pi * args.eps / 2) ** 2 * param_dict_1["omega_cavs"][0] / param_dict_1["kappa_cavs"][0]
        kappas = param_dict["kappa_cavs"]
        omegas = param_dict["omega_cavs"]
        kappa_a, kappa_b = kappas[0], kappas[1]
        omega_a, omega_b = omegas[0], omegas[1]
        def g_func(omega, kappa):
            return 2.0 * np.sqrt(kappa * P / omega)
        epsilon_array = np.array([g_func(omega_a, kappa_a), g_func(omega_b, kappa_b)])
    else:
        raise RuntimeError("only one or two cavities supported")
    omega_d_vals = eval(args.omega_d_vals)
    delay_time_vals = eval(args.delay_times)
    delay_times = np.linspace(delay_time_vals[0], delay_time_vals[1], delay_time_vals[2])
    omega_d_list = 2.0 * np.pi * np.linspace(omega_d_vals[0], omega_d_vals[1], args.num_pts)
    omega_d_cav = omega_d_list[args.idx]
    param_dict["omega_d_cav"] = omega_d_cav
    param_dict["epsilon_array"] = epsilon_array
    param_dict["cavity_dim"] = args.cav_dim
    param_dict["delay_times"] = delay_times
    param_dict["nsteps"] = args.nsteps
    param_dict["thermal_time"] = args.thermal_time
    param_dict["temp"] = args.temp
    param_dict["destructive_interference"] = args.destructive_interference
    param_dict["include_stark_shifts"] = args.include_stark_shifts
    param_dict["interference_scale"] = args.interference_scale
    param_dict["interference"] = args.interference
    if args.exp_type == "ramsey":
        # p0 = (6 * 10 ** 4, 0.045, -1.7)
        p0 = (1 * 10 ** 4, 0.0433, 0.5, 0.5, -1.8)
    elif args.exp_type == "T1":
        p0 = (4 * 10 ** 4, 1.0, 0.0,)
    else:
        raise RuntimeError("unsupported experiment type")
    ramsey_coherent = CoherentDephasing(**param_dict)
    ramsey_coherent.main_ramsey(filepath, p0, full_cosine=args.full_cosine)
