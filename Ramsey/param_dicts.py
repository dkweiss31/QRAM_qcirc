import numpy as np

omega_a = 2.0 * np.pi * 3.3261
omega_b = 2.0 * np.pi * 3.4712
omega_cavs = np.array([omega_a, omega_b])
kappa_a = 2.0 * np.pi * 0.04268
kappa_b = 2.0 * np.pi * 0.04784
kappa_cavs = np.array([kappa_a, kappa_b])
chiaq = -2.0 * np.pi * 0.000322
chibq = -2.0 * np.pi * 0.000571
chi_cavstmon = np.array([chiaq, chibq])
param_dict_2 = {
    "cavity_dim": 5,
    "num_cavs": 2,
    "nsteps": 100000,
    "atol": 1e-12,
    "rtol": 1e-12,
    "interference": True,
    "tmon_dim": 2,
    "thermal_time": 200.0,
    "delay_times": np.linspace(0.0, 2000.0, 301),
    "omega_cavs": omega_cavs,
    "kappa_cavs": kappa_cavs,
    "omega_tmon": 2.0 * np.pi * 5.7423,
    "omega_d_tmon": 2.0 * np.pi * (5.7423 - 0.0071),
    "chi_cavstmon": chi_cavstmon,
    "temp": 0.1,
    "loss_ratio": 0.0,
}

param_dict_1 = {
    "cavity_dim": 5,
    "num_cavs": 1,
    "nsteps": 100000,
    "atol": 1e-12,
    "rtol": 1e-12,
    "interference": True,
    "tmon_dim": 2,
    "thermal_time": 200.0,
    "delay_times": np.linspace(0.0, 2000.0, 301),
    "omega_cavs": np.array([omega_a, ]),
    "kappa_cavs": np.array([kappa_a, ]),
    "omega_tmon": 2.0 * np.pi * 5.7423,
    "omega_d_tmon": 2.0 * np.pi * (5.7423 - 0.0071),
    "chi_cavstmon": np.array([chiaq, ]),
    "temp": 0.1,
}
