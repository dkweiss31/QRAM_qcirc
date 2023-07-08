import h5py
import numpy as np

from ramsey_Yao import RamseyExperiment


def main_ramsey_two(filepath, param_dict):
    interference = param_dict["interference"]
    cavity_dim = param_dict["cavity_dim"]
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
    control_dt = param_dict["control_dt"]
    ramsey_experiment_two = RamseyExperiment(omega_tmon, omega_array, chi_array,
                                             kappa_array, temp,
                                             tmon_dim, cavity_dim, 2, control_dt=control_dt)
    thermal_state = ramsey_experiment_two.obtain_thermal_state()
    if interference:
        c_ops = ramsey_experiment_two.construct_c_ops_interference()
    else:
        c_ops = ramsey_experiment_two.construct_c_ops_no_interference()
    ramsey_result = ramsey_experiment_two.ramsey_liouv(thermal_state, delay_times, omega_d, c_ops)
    gamma_phi, popt, pcov = ramsey_experiment_two.extract_gammaphi(ramsey_result, delay_times, window=window)
    naive_gamma_phi = sum(ramsey_experiment_two.gamma_phi_full_func(chi_array, ramsey_experiment_two.nths(), kappa_array))
    print(f"Ramsey experiment with interference = {interference}, 2 cavities")
    print(f"naive gamma_phi = {naive_gamma_phi}")
    print(f"real gamma_phi = {gamma_phi}")
    with h5py.File(filepath, "w") as f:
        gamma_phi = f.create_dataset("gamma_phi", data=gamma_phi)
        for kwarg in param_dict.keys():
            f.attrs[kwarg] = param_dict[kwarg]

# start_time = time.time()
# ramsey_result_indep = ramsey_experiment_two.ramsey_indep(thermal_state, delay_times, omega_d, c_ops)
# end_time = time.time()
# print(f"indep took {end_time - start_time} seconds")
# gamma_phi, popt, pcov = ramsey_experiment_two.extract_gammaphi(ramsey_result_indep, delay_times)
# naive_gamma_phi = sum(ramsey_experiment_two.gamma_phi_full_func(chi_array, ramsey_experiment_two.nths(), kappa_array))
# print(f"Ramsey experiment with interference = {interference}, 2 cavities")
# print(f"naive gamma_phi = {naive_gamma_phi}")
# print(f"real gamma_phi = {gamma_phi}")