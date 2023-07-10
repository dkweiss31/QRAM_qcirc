import numpy as np

from ramsey_Yao import RamseyExperiment

tmon_dim = 2
cavity_dim = 7
delay_times = np.linspace(0.0, 2000.0, 301)
thermal_time = 100
window = (0, 300)
omega_a = 2.0 * np.pi * 3.3261  # 3.40 #3.326
omega_tmon = 2.0 * np.pi * 5.7423  # 5.867
omega_d = omega_tmon - 2.0 * np.pi * 0.0071
chiaq = -2.0 * np.pi * 0.000322
kappa_a = 2.0 * np.pi * 0.04268
temp = 0.1
ramsey_experiment_one = RamseyExperiment(
    omega_tmon,
    np.array(
        [
            omega_a,
        ]
    ),
    np.array(
        [
            chiaq,
        ]
    ),
    np.array(
        [
            kappa_a,
        ]
    ),
    temp,
    tmon_dim,
    cavity_dim,
    1,
)
nth = ramsey_experiment_one.nths()
thermal_state = ramsey_experiment_one.obtain_thermal_state()
c_ops = ramsey_experiment_one.construct_c_ops_no_interference()
ramsey_result = ramsey_experiment_one.ramsey_liouv(
    thermal_state, delay_times, omega_d, c_ops
)
gamma_phi, popt, pcov = ramsey_experiment_one.extract_gammaphi(
    ramsey_result, delay_times
)
naive_gamma_phi = ramsey_experiment_one.gamma_phi_full_func(chiaq, nth, kappa_a)
print(f"naive gamma_phi = {naive_gamma_phi}")
print(f"real gamma_phi = {gamma_phi}")
