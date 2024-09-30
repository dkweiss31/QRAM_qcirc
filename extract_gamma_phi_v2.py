import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from QRAM_utils.utils import extract_info_from_h5

filepath = "out/00011_cohere_Ramsey_twocav.h5py"
data_dict, param_dict = extract_info_from_h5(filepath)
ramsey_result_indep = data_dict["ramsey_result"]
delay_times = param_dict["delay_times"]

def plot_ramsey(ramsey_result, popt_T2, filename=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(delay_times, ramsey_result, "o")
    plot_times = np.linspace(0.0, delay_times[-1], 2000)
    ax.plot(plot_times, T2_func(plot_times, *popt_T2), linestyle="-")
    ax.plot(plot_times, T2_func(plot_times, 1e6, 0.00014, 0.5, 0.5, 0.0))
    ax.set_ylabel(r"$P(|+\rangle)$", fontsize=12)
    ax.set_xlabel("time [ns]", fontsize=12)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def extract_gammaphi(
    ramsey_result,
    window=None,
    p0=(6 * 10**4, 0.045, 0.5, 0.2, 0),
    plot=True,
):
    if window is None:
        window = (0, len(delay_times))
    popt_T2, pcov_T2 = curve_fit(
        T2_func,
        delay_times[window[0]: window[1]],
        ramsey_result[window[0]: window[1]],
        p0=p0,
        maxfev=6000,
        bounds=((100, -2, -2, -2, -np.pi), (10**15, 2, 2, 2, np.pi)),
    )
    if plot:
        plot_ramsey(ramsey_result, popt_T2)
    print("popt: ", popt_T2)
    print("pcov: ", pcov_T2)
    print("condition_num: ", np.linalg.cond(pcov_T2))
    gamma_stdev = np.sqrt(pcov_T2[0, 0]) / popt_T2[0]**2
    print("stdev of gamma_2 = ", gamma_stdev * 10**6 / (2 * np.pi), "kHz")
    return (1 / popt_T2[0]) * 10**6 / (2 * np.pi), popt_T2, pcov_T2


def T2_func(t, t2, omega, a, b, phi):
    return a * np.exp(-t / t2) * np.cos(omega * t + phi) + b

# p0 = (1 * 10**6, 0.0004, 0.5, 0.5, 0.0) # for 05
p0 = (1 * 10**6, 0.00014, 0.5, 0.5, 0.0)
gamma_phi_indep, popt, pcov = extract_gammaphi(
    ramsey_result_indep, p0=p0, plot=True
)
print("gamma_phi = ", gamma_phi_indep, " kHz")
