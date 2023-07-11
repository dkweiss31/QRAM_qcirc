import h5py
import numpy as np

from ramsey_Yao import RamseyExperiment


def main_ramsey_two(filepath, param_dict):
    interference = param_dict["interference"]
    cavity_dim = param_dict["cavity_dim"]
    nsteps = param_dict["nsteps"]
    atol = param_dict["atol"]
    rtol = param_dict["rtol"]
    tmon_dim = 2
    delay_times = np.linspace(0.0, 2000.0, 301)
    window = (0, 300)
    omega_a = 2.0 * np.pi * 3.3261  # 3.40 #3.326
    omega_b = 2.0 * np.pi * 3.4712
    omega_array = np.array([omega_a, omega_b])
    omega_tmon = 2.0 * np.pi * 5.7423  # 5.867
    omega_d = omega_tmon - 2.0 * np.pi * 0.0071
    chiaq = -2.0 * np.pi * 0.000322
    chibq = -2.0 * np.pi * 0.000571
    chi_array = np.array([chiaq, chibq])
    kappa_a = 2.0 * np.pi * 0.04268
    kappa_b = 2.0 * np.pi * 0.04784
    kappa_array = np.array([kappa_a, kappa_b])
    temp = 0.1
    ramsey_experiment_two = RamseyExperiment(
        omega_tmon,
        omega_array,
        chi_array,
        kappa_array,
        temp,
        tmon_dim,
        cavity_dim,
        2,
        nsteps=nsteps,
        atol=atol,
        rtol=rtol,
    )
    thermal_state = ramsey_experiment_two.obtain_thermal_state()
    if interference:
        c_ops = ramsey_experiment_two.construct_c_ops_interference()
    else:
        c_ops = ramsey_experiment_two.construct_c_ops_no_interference()
    # ramsey_result = ramsey_experiment_two.ramsey_liouv(thermal_state, delay_times, omega_d, c_ops)
    # gamma_phi_liouv, popt, pcov = ramsey_experiment_two.extract_gammaphi(ramsey_result, delay_times, window=window)
    naive_gamma_phi = sum(
        ramsey_experiment_two.gamma_phi_full_func(
            chi_array, ramsey_experiment_two.nths(), kappa_array
        )
    )
    ramsey_result_indep = ramsey_experiment_two.ramsey_indep(
        thermal_state, delay_times, omega_d, c_ops
    )
    print(f"writing to {filepath}")
    with h5py.File(filepath, "w") as f:
        result_written = f.create_dataset("ramsey_result", data=ramsey_result_indep)
        for kwarg in param_dict.keys():
            f.attrs[kwarg] = param_dict[kwarg]
    p0 = (6 * 10**4, 0.045, 0.5, 0.2, -1)
    gamma_phi_indep, popt, pcov = ramsey_experiment_two.extract_gammaphi(
        ramsey_result_indep, delay_times, p0=p0
    )
    print(
        f"Ramsey experiment with interference = {interference}, cav_dim = {cavity_dim}, 2 cavities"
    )
    print(f"naive gamma_phi = {naive_gamma_phi}")
    print(f"indep gamma_phi = {gamma_phi_indep}")
