import dynamiqs as dq
import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dynamiqs.solver import Tsit5
from quantum_utils import write_to_h5, generate_file_path
from scipy.optimize import curve_fit


exp_type = "ramsey"  # can be "ramsey" or "T1"

omega_a = 2.0 * np.pi * 3.3263
omega_b = 2.0 * np.pi * 3.4715
omega_cavs = np.array([omega_a, omega_b])
kappa_a = 2.0 * np.pi * 0.04268
kappa_b = 2.0 * np.pi * 0.04785
kappa_cavs = np.array([kappa_a, kappa_b])
chiaq = -2.0 * np.pi * 0.000322
chibq = -2.0 * np.pi * 0.000571
chi_cavstmon = np.array([chiaq, chibq])
chi_crosscav = np.array([-2.0 * np.pi * 0.741e-6, ])
EJ = 2.0 * np.pi * 32.7
alpha = -2.0 * np.pi * 0.124
epsilon = 2.0 * jnp.pi * 0.01
delay_times = np.linspace(0.0, 10000, 301)
thermal_time = 200.0
DESTRUCTIVE_INTERFERENCE = -1

P = (epsilon / 2) ** 2 * omega_a / kappa_a


def g_func(omega, kappa):
    return 2.0 * np.sqrt(kappa * P / omega)


epsilon_array = np.array([g_func(omega_a, kappa_a), g_func(omega_b, kappa_b)])
print("epsilon = ", epsilon_array/2/np.pi)


cav_dim = 10
tmon_dim = 10
cav_truncated_dim = 6
tmon_truncated_dim = 2
full_dims = [cav_dim, cav_dim, tmon_dim]
full_dim = cav_dim**2 * tmon_dim
truncated_dims = [cav_truncated_dim, cav_truncated_dim, tmon_truncated_dim]
dim_after_diag = cav_truncated_dim**2 * tmon_truncated_dim
omega_tmon = 2.0 * np.pi * 5.837
##### omega_d_cav = 2.0 * np.pi * 3.3261


def phi_cav(_alpha, _EJ, _chi_cavstmon, idx=0):
    return (_chi_cavstmon[idx] ** 2 / (2 * np.abs(_alpha) * _EJ)) ** (1 / 4)


def phi_q_func(_alpha, _EJ):
    return (2 * np.abs(_alpha) / _EJ) ** (1 / 4)


a, b, q = dq.destroy(cav_dim, cav_dim, tmon_dim)

phi_a = phi_cav(alpha, EJ, chi_cavstmon, idx=0)
phi_b = phi_cav(alpha, EJ, chi_cavstmon, idx=1)
phi_q = phi_q_func(alpha, EJ)
print("phi_a = ", phi_a)
print("phi_b = ", phi_b)
print("phi_q = ", phi_q)

bare_labels = np.array(list(np.ndindex(*full_dims)))
_H0 = omega_a * dq.dag(a) @ a + omega_b * dq.dag(b) @ b + omega_tmon * dq.dag(q) @ q
_H0 += -EJ * dq.cosm(phi_q * (q + dq.dag(q)) + phi_a * (a + dq.dag(a)) + phi_b * (b + dq.dag(b)))
_H0 += -0.5 * EJ * phi_q ** 2 * dq.powm(q + dq.dag(q), 2)
_H0 += - 0.5 * EJ * phi_a ** 2 * dq.powm(a + dq.dag(a), 2)
_H0 += - 0.5 * EJ * phi_b ** 2 * dq.powm(b + dq.dag(b), 2)
_H0 += -EJ * phi_q * phi_a * (q + dq.dag(q)) @ (a + dq.dag(a))
_H0 += -EJ * phi_q * phi_b * (q + dq.dag(q)) @ (b + dq.dag(b))
_H0 += -EJ * phi_a * phi_b * (a + dq.dag(a)) @ (b + dq.dag(b))
evals, evecs = jnp.linalg.eigh(_H0)

a_tilde = dq.dag(evecs) @ a @ evecs
b_tilde = dq.dag(evecs) @ b @ evecs
q_tilde = dq.dag(evecs) @ q @ evecs
_H0_tilde = dq.dag(evecs) @ _H0 @ evecs
H0_tilde = jnp.diag(evals - evals[0])
# can verify that H0_tilde \approx _H0_tilde

# find the mapping from bare to dressed indices. This will allow
# us to eventually truncate away states we are uninterested in
bare_to_dressed_labels = {}
for bare_label in bare_labels:
    init_ket = dq.tensor(
        dq.basis(cav_dim, bare_label[0]),
        dq.basis(cav_dim, bare_label[1]),
        dq.basis(tmon_dim, bare_label[2])
    )
    ovlps = jnp.abs(dq.dag(evecs) @ init_ket)
    max_idx = np.argmax(ovlps, axis=0)
    bare_to_dressed_labels[tuple(bare_label)] = max_idx

# only keep a subset of states. This will order them to have
# the 'natural' tensor product ordering
truncated_bare_labels = np.array(list(np.ndindex(*truncated_dims)))
dressed_indices_to_keep = []
for bare_label in truncated_bare_labels:
    dressed_indices_to_keep.append(bare_to_dressed_labels[tuple(bare_label)])
proj = dq.dag(jnp.squeeze(jnp.asarray([dq.basis(full_dim, dressed_idx)
                                       for dressed_idx in dressed_indices_to_keep])))
# should only do this operation on tilde operators (i.e. not a and q) because
# the ordering of the tilde operators is the dressed ordering (according to energy)
a_tilde_trunc = dq.dag(proj) @ a_tilde @ proj
b_tilde_trunc = dq.dag(proj) @ b_tilde @ proj
q_tilde_trunc = dq.dag(proj) @ q_tilde @ proj
H0_tilde_trunc = dq.dag(proj) @ H0_tilde @ proj

chi_a = (evals[bare_to_dressed_labels[(1, 0, 1)]]
       - evals[bare_to_dressed_labels[(1, 0, 0)]]
       - evals[bare_to_dressed_labels[(0, 0, 1)]]
       + evals[bare_to_dressed_labels[(0, 0, 0)]]
       )
chi_b = (evals[bare_to_dressed_labels[(0, 1, 1)]]
       - evals[bare_to_dressed_labels[(0, 1, 0)]]
       - evals[bare_to_dressed_labels[(0, 0, 1)]]
       + evals[bare_to_dressed_labels[(0, 0, 0)]]
       )
print(chi_a/2/np.pi, chi_b/2/np.pi)
omega_q_tilde = evals[bare_to_dressed_labels[(0, 0, 1)]] - evals[bare_to_dressed_labels[(0, 0, 0)]]
omega_a_tilde = evals[bare_to_dressed_labels[(1, 0, 0)]] - evals[bare_to_dressed_labels[(0, 0, 0)]]
omega_b_tilde = evals[bare_to_dressed_labels[(0, 1, 0)]] - evals[bare_to_dressed_labels[(0, 0, 0)]]
omega_d_cav = 2.0 * jnp.pi * 3.3263 #  3.38 # omega_a_tilde

# define rotating frame to be drive of cavity and frequency of qubit
rot_frame_diag = jnp.squeeze(jnp.asarray(
    [omega_d_cav * bare_1 + omega_d_cav * bare_2 + omega_q_tilde * bare_3
     for (bare_1, bare_2, bare_3) in truncated_bare_labels]))
n_op_a = dq.tensor(dq.number(cav_truncated_dim), dq.eye(cav_truncated_dim), dq.eye(tmon_truncated_dim))
n_op_b = dq.tensor(dq.eye(cav_truncated_dim), dq.number(cav_truncated_dim), dq.eye(tmon_truncated_dim))
rot_frame_mat = jnp.diag(rot_frame_diag)
rot_frame_drive = jnp.reshape(rot_frame_diag, (-1, 1)) - rot_frame_diag
H0_rot_tilde_trunc = H0_tilde_trunc - rot_frame_mat
a_trunc_bare = dq.tensor(dq.destroy(cav_truncated_dim), dq.eye(cav_truncated_dim), dq.eye(tmon_truncated_dim))
b_trunc_bare = dq.tensor(dq.eye(cav_truncated_dim), dq.destroy(cav_truncated_dim), dq.eye(tmon_truncated_dim))
# a_trunc_bare = a_tilde_trunc

H0_tc = dq.timecallable(
    lambda t: jnp.exp(1j * rot_frame_drive * t) * H0_rot_tilde_trunc
)
H_drive = dq.timecallable(
    lambda t: 2.0 * epsilon # 2.0 so that if you make RWA has amplitude espilon
              * jnp.exp(1j * rot_frame_drive * t) # rotating frame
              * jnp.cos(omega_d_cav * t)  # actual drive frequency
              * ((a_trunc_bare + dq.dag(a_trunc_bare))
                 - (b_trunc_bare + dq.dag(b_trunc_bare)))
)

H = H0_tc + H_drive

e_ops = [n_op_a, n_op_b]
# TODO nonzero temp
c_ops = [dq.timecallable(lambda t: (np.sqrt(kappa_a) * a_trunc_bare
                                    + DESTRUCTIVE_INTERFERENCE * np.sqrt(kappa_b) * b_trunc_bare)
                                   * jnp.exp(1j * rot_frame_drive * t)), ]
options = dq.Options(
    save_states=True, progress_meter=None,
)

if exp_type == "ramsey":
    # this makes use of the fact that the first dressed state is (0, 1)
    psi0 = dq.unit(dq.basis(dim_after_diag, 1) + dq.basis(dim_after_diag, 0))
    _psi0 = dq.unit(dq.basis(tmon_truncated_dim, 1) + dq.basis(tmon_truncated_dim, 0))
    m_op = dq.tensor(
        dq.eye(cav_truncated_dim),
        dq.eye(cav_truncated_dim),
        _psi0 @ dq.dag(_psi0)
    )
    p0 = (1 * 10 ** 5, 5.69661179e-04, 0.5, 0.5, 0.1)
elif exp_type == "T1":
    psi0 = dq.basis(dim_after_diag, 1)
    m_op = dq.tensor(
        dq.eye(cav_truncated_dim),
        dq.basis(tmon_truncated_dim, 1) @ dq.dag(dq.basis(tmon_truncated_dim, 1))
    )
    p0 = (4 * 10 ** 3, 1.0, 0.0,)
else:
    raise ValueError("Exp type needs to be T1 or ramsey")


def mesolve_for_final_state(H, init_dm, t):
    result = dq.mesolve(
        H,
        c_ops,
        init_dm,
        (0, t),
        exp_ops=e_ops,
        solver=Tsit5(),
        options=options,
    )
    # take occupation of first cavity as representative
    nbar_a = np.mean(result.expects[0])
    nbar_b = np.mean(result.expects[1])
    print("expectation value of n_1: ", np.max(result.expects[0]))
    gamma_phi_a = gamma_coherent(nbar_a, kappa_a, chi_a, omega_d_cav - omega_a_tilde)/(2.0*np.pi) * 10**6
    gamma_phi_b = gamma_coherent(nbar_b, kappa_b, chi_b, omega_d_cav - omega_b_tilde) / (2.0 * np.pi) * 10 ** 6
    print("analytical coherent gamma_phi = ", gamma_phi_a + gamma_phi_b)
    return result.final_state


def decoherence_experiment():
    final_prob = np.zeros_like(delay_times)
    thermal_state = mesolve_for_final_state(
        H, psi0, thermal_time
    )
    state_after_prev_delay = thermal_state
    final_prob[0] = np.real(np.trace(state_after_prev_delay @ m_op))
    delay_dif = delay_times[1] - delay_times[0]
    for idx in range(1, len(delay_times)):
        # run the delay
        state_after_delay = mesolve_for_final_state(
            H, state_after_prev_delay, delay_dif
        )
        state_after_prev_delay = state_after_delay
        final_prob[idx] = np.real(np.trace(state_after_delay @ m_op))
    return final_prob


def main_ramsey(
    filepath,
    p0=(6 * 10 ** 4, 0.045, 0.5, 0.5, -1.7),
):
    print(f"Saving to filepath {filepath}")
    ramsey_result = decoherence_experiment()
    write_to_h5(
        filepath,
        {"ramsey_result": ramsey_result},
        {"delay_times": delay_times,
         "destructive_interference": DESTRUCTIVE_INTERFERENCE,
         "cav_truncated_dim": cav_truncated_dim,
         "tmon_truncated_dim": tmon_truncated_dim,
         "thermal_time": thermal_time,
         "omega_d_cav": omega_d_cav,
         }
    )
    if exp_type == "ramsey":
        gamma_phi, popt, pcov = extract_gammaphi(
            ramsey_result, p0=p0
        )
        # write this separately in case the fit fails
        with h5py.File(filepath, "a") as f:
            written_data = f.create_dataset("gamma_phi", data=gamma_phi)
        print(f"extracted gamma_phi = {gamma_phi}")
        print(f"optimized parameters {popt}")
    elif exp_type == "T1":
        gamma_1, popt, pcov = extract_gamma1(
            ramsey_result, p0=p0
        )
        # write this separately in case the fit fails
        with h5py.File(filepath, "a") as f:
            written_data = f.create_dataset("gamma_1", data=gamma_1)
        print(f"extracted gamma_1 = {gamma_1}")
        print(f"optimized parameters {popt}")

def plot_ramsey(ramsey_result, popt_T2, filename=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(delay_times, ramsey_result, "o")
    plot_times = np.linspace(0.0, delay_times[-1], 2000)
    ax.plot(plot_times, T2_func(plot_times, *popt_T2), linestyle="-")
    ax.set_ylabel(r"$P(|+\rangle)$", fontsize=12)
    ax.set_xlabel("time [ns]", fontsize=12)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def plot_T1(ramsey_result, popt_T1, filename=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(delay_times, ramsey_result, "o")
    plot_times = np.linspace(0.0, delay_times[-1], 2000)
    ax.plot(plot_times, T1_func(plot_times, *popt_T1), linestyle="-")
    ax.set_ylabel(r"$P(|1\rangle)$", fontsize=12)
    ax.set_xlabel("time [ns]", fontsize=12)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def extract_gamma1(
    ramsey_result,
    window=None,
    p0=(6 * 10 ** 4, 1.0, 0.0),
    plot=True,
):
    if window is None:
        window = (0, len(delay_times))
    popt_T1, pcov_T1 = curve_fit(
        T1_func,
        delay_times[window[0]: window[1]],
        ramsey_result[window[0]: window[1]],
        p0=p0,
        maxfev=6000,
        bounds=((100, -2, -2), (10**15, 2, 2)),
    )
    if plot:
        plot_T1(ramsey_result, popt_T1)
    print("popt: ", popt_T1)
    print("pcov: ", pcov_T1)
    return (1 / popt_T1[0]) * 10**6 / (2 * np.pi), popt_T1, pcov_T1


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

def T1_func(t, t1, a, b):
    return a * np.exp(-t / t1) + b

def T1_gauss_func(t, t1, a, b):
    return a * np.exp(-0.5 * (t / t1)**2) + b

def gamma_coherent(nbar, kappa, chi, delta):
    return 0.5 * chi**2 * nbar * kappa / (delta**2 + (0.5 * kappa)**2)

filepath = generate_file_path("h5py", f"cohere_Ramsey_twocav", "out")
main_ramsey(filepath=filepath, p0=p0)
