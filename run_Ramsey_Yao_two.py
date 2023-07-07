import time

import numpy as np

from ramsey_Yao import RamseyExperiment

interference = True
tmon_dim = 2
cavity_dim = 4
delay_times = np.linspace(0.0, 2000.0, 301)
thermal_time = 100
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
ramsey_experiment_two = RamseyExperiment(omega_tmon, omega_array, chi_array,
                                         kappa_array, temp,
                                         tmon_dim, cavity_dim, 2)
thermal_state = ramsey_experiment_two.obtain_thermal_state()
if interference:
    c_ops = ramsey_experiment_two.construct_c_ops_interference()
else:
    c_ops = ramsey_experiment_two.construct_c_ops_no_interference()
start_time = time.time()
ramsey_result = ramsey_experiment_two.ramsey_liouv(thermal_state, delay_times, omega_d, c_ops)
end_time = time.time()
print(f"liouv took {end_time - start_time} seconds")
start_time = time.time()
ramsey_result_indep = ramsey_experiment_two.ramsey_indep(thermal_state, delay_times, omega_d, c_ops, num_cpus=1)
end_time = time.time()
print(f"indep took {end_time - start_time} seconds")
start_time = time.time()
ramsey_result_indep_par = ramsey_experiment_two.ramsey_indep(thermal_state, delay_times, omega_d, c_ops, num_cpus=8)
end_time = time.time()
print(f"indep par took {end_time - start_time} seconds")
gamma_phi, popt, pcov = ramsey_experiment_two.extract_gammaphi(ramsey_result, delay_times)
naive_gamma_phi = sum(ramsey_experiment_two.gamma_phi_full_func(chi_array, ramsey_experiment_two.nths(), kappa_array))
print(f"Ramsey experiment with interference = {interference}, 2 cavities")
print(f"naive gamma_phi = {naive_gamma_phi}")
print(f"real gamma_phi = {gamma_phi}")