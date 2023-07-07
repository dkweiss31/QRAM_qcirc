import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, k
from qutip import (
    destroy,
    sigmax,
    sigmay,
    sigmaz,
    mesolve,
    Options,
    operator_to_vector,
    vector_to_operator,
    liouvillian,
)
from scipy.optimize import curve_fit

from utils import id_wrap_ops, construct_basis_states_list, get_map


class RamseyExperiment:
    def __init__(
        self,
        omega_tmon,
        omega_cavs,
        chi_cavstmon,
        kappa_cavs,
        temp,
        tmon_dim,
        cavity_dim,
        num_cavs,
        control_dt=1.0,
    ):
        assert len(omega_cavs) == len(chi_cavstmon) == len(kappa_cavs) == num_cavs
        self.omega_tmon = omega_tmon
        self.omega_cavs = omega_cavs
        self.chi_cavstmon = chi_cavstmon
        self.kappa_cavs = kappa_cavs
        self.temp = temp
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim
        self.num_cavs = num_cavs
        self.control_dt = control_dt
        self.truncated_dims = num_cavs * [cavity_dim] + [tmon_dim]

    def tmon_Pauli_ops(self):
        tmon_idx = self.num_cavs
        sx = id_wrap_ops(sigmax(), tmon_idx, self.truncated_dims)
        sy = id_wrap_ops(sigmay(), tmon_idx, self.truncated_dims)
        sz = id_wrap_ops(sigmaz(), tmon_idx, self.truncated_dims)
        return sx, sy, sz

    def annihilation_ops(self):
        annihilation_ops_list = []
        for idx in range(self.num_cavs):
            annihilation_ops_list.append(
                id_wrap_ops(destroy(self.cavity_dim), idx, self.truncated_dims)
            )
        return annihilation_ops_list

    def nths(self):
        return 1.0 / (np.exp(hbar * self.omega_cavs * 10**9 / (k * self.temp)) - 1)

    def hamiltonian(self):
        annihilation_ops_list = self.annihilation_ops()
        (sx, sy, sz) = self.tmon_Pauli_ops()
        H0 = sum(omega * a_op.dag() * a_op for (omega, a_op) in zip(self.omega_cavs, self.annihilation_ops()))
        for idx, a_op in enumerate(annihilation_ops_list):
            H0 += 0.5 * self.chi_cavstmon[idx] * a_op.dag() * a_op * sz
        return H0

    def construct_c_ops_interference(self):
        collective_lowering = sum(
            np.sqrt(kappa * (1 + nth)) * a_op
            for (kappa, nth, a_op) in zip(
                self.kappa_cavs, self.nths(), self.annihilation_ops()
            )
        )
        collective_raising = sum(
            np.sqrt(kappa * nth) * a_op.dag()
            for (kappa, nth, a_op) in zip(
                self.kappa_cavs, self.nths(), self.annihilation_ops()
            )
        )
        return [collective_lowering, collective_raising]

    def construct_c_ops_no_interference(self):
        individual_lowering = [
            np.sqrt(kappa * (1 + nth)) * a_op
            for (kappa, nth, a_op) in zip(
                self.kappa_cavs, self.nths(), self.annihilation_ops()
            )
        ]
        individual_raising = [
            np.sqrt(kappa * nth) * a_op.dag()
            for (kappa, nth, a_op) in zip(
                self.kappa_cavs, self.nths(), self.annihilation_ops()
            )
        ]
        return individual_lowering + individual_raising

    def obtain_thermal_state(
        self, total_time: float = 100, initial_state=None, interference=True
    ):
        if total_time < 2.0 / np.min(self.kappa_cavs):
            print("running for too short of a time to get a thermal state")
        if initial_state is None:
            fock_spec = tuple(len(self.truncated_dims) * [0])
            (initial_state,) = construct_basis_states_list(
                [
                    fock_spec,
                ],
                self.truncated_dims,
            )
        tlist = np.linspace(0.0, total_time, int(total_time / self.control_dt))
        H0 = self.hamiltonian()
        if interference:
            c_ops = self.construct_c_ops_interference()
        else:
            c_ops = self.construct_c_ops_no_interference()
        result_thermal = mesolve(
            H0,
            initial_state * initial_state.dag(),
            tlist,
            c_ops,
            options=Options(store_final_state=True),
        )
        return result_thermal.final_state

    @staticmethod
    def T2_func(t, t2, omega, a, b, phi):
        return a * np.exp(-t / t2) * np.cos(omega * t + phi) + b

    @staticmethod
    def T1_func(t, t1, a, b):
        return a * np.exp(-t / t1) + b

    @staticmethod
    def gamma_phi_func(chi, nth, kappa):
        return (chi**2 * nth * (1 + nth) / kappa) * 10**6 / (2 * np.pi)

    @staticmethod
    def gamma_phi_full_func(chi, nth, kappa):
        return (chi**2 * nth * (1 + nth) * kappa / (kappa**2 + chi**2)) * 10**6 / (2 * np.pi)

    def pi_2_pulse(
        self, H, init_dm, c_ops, tmon_drive_amp: float = 2.0 * np.pi * 0.02
    ):
        t_pi2 = np.pi / (2 * tmon_drive_amp)
        tlist = np.linspace(0.0, t_pi2, int(t_pi2 / self.control_dt))
        (sx, sy, sz) = self.tmon_Pauli_ops()
        return mesolve(
            H + 0.5 * tmon_drive_amp * sx, init_dm, tlist, c_ops=c_ops, options=Options(store_final_state=True)
        )

    def ramsey_one_shot(self, H, thermal_state, liouv_delay, c_ops):
        first_pi_2 = self.pi_2_pulse(H, thermal_state, c_ops)
        state_after_delay = liouv_delay * operator_to_vector(first_pi_2.final_state)
        last_pi_2 = self.pi_2_pulse(
            H, vector_to_operator(state_after_delay), c_ops
        )
        result = np.trace(last_pi_2.final_state * thermal_state)
        return result

    def ramsey_indep(self, thermal_state, delay_times, omega_d, c_ops, num_cpus=1):
        (sx, sy, sz) = self.tmon_Pauli_ops()
        H0_q = -0.5 * (self.omega_tmon - omega_d) * sz
        H = self.hamiltonian() + H0_q
        first_pi_2 = self.pi_2_pulse(H, thermal_state, c_ops)
        if num_cpus == 1:
            final_prob = np.zeros_like(delay_times, dtype=complex)
            delay_dif = delay_times[1] - delay_times[0]
            for idx, delay_time in enumerate(delay_times):
                if idx == 0:
                    state_after_prev_delay = first_pi_2.final_state
                    final_state = self.pi_2_pulse(H, state_after_prev_delay, c_ops).final_state
                else:
                    tlist = np.linspace(0.0, delay_dif, int(delay_dif / self.control_dt))
                    state_after_delay = mesolve(H, state_after_prev_delay, tlist, c_ops=c_ops,
                                                options=Options(store_final_state=True)).final_state
                    final_state = self.pi_2_pulse(H, state_after_delay, c_ops).final_state
                    state_after_prev_delay = state_after_delay
                final_prob[idx] = np.trace(final_state * thermal_state)
        else:
            target_map = get_map(num_cpus)

            def _delay_mesolve(delay_time):
                if delay_time == 0.0:
                    return first_pi_2
                tlist = np.linspace(0.0, delay_time, int(delay_time / self.control_dt))
                return mesolve(H, first_pi_2.final_state, tlist, c_ops=c_ops, options=Options(store_final_state=True))
            delay_results = list(target_map(_delay_mesolve, delay_times))
            states_after_delay = [delay_result.final_state for delay_result in delay_results]
            final_states = list(target_map(lambda state: self.pi_2_pulse(H, state, c_ops), states_after_delay))
            return np.array([np.trace(state * thermal_state) for state in final_states])

    def ramsey_liouv(self, thermal_state, delay_times, omega_d, c_ops):
        ramsey_results = np.zeros_like(delay_times)
        delay_dif = delay_times[1] - delay_times[0]  # assume equally spaced array
        (sx, sy, sz) = self.tmon_Pauli_ops()
        H0_q = -0.5 * (self.omega_tmon - omega_d) * sz
        H = self.hamiltonian() + H0_q
        liouv_delay = (liouvillian(H, c_ops) * delay_dif).expm()
        init_liouv_delay = (liouvillian(H, c_ops) * delay_times[0]).expm()
        for i, delay_time in enumerate(delay_times):
            if i == 0:
                next_liouv_delay = init_liouv_delay
            else:
                next_liouv_delay = liouv_delay * prev_liouv_delay
            ramsey_results[i] = np.real(
                self.ramsey_one_shot(H, thermal_state, next_liouv_delay, c_ops)
            )
            prev_liouv_delay = next_liouv_delay
        return ramsey_results

    def plot_ramsey(self, ramsey_result, delay_times, popt_T2):
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.plot(delay_times, ramsey_result, "o")
        plot_times = np.linspace(0.0, delay_times[-1], 2000)
        plt.plot(plot_times, self.T2_func(plot_times, *popt_T2), linestyle="-")
        plt.show()

    def extract_gammaphi(
        self,
        ramsey_result,
        delay_times,
        window=None,
        p0=(6 * 10**4, 0.045, 0.2, 0.2, -2),
        plot=True,
    ):
        if window is None:
            window = (0, len(delay_times))
        popt_T2, pcov_T2 = curve_fit(
            self.T2_func,
            delay_times[window[0] : window[1]],
            ramsey_result[window[0] : window[1]],
            p0=p0,
            maxfev=6000,
            bounds=((100, -2, -2, -2, -np.pi), (10**8, 2, 2, 2, np.pi)),
        )
        if plot:
            self.plot_ramsey(ramsey_result, delay_times, popt_T2)
        return (1 / popt_T2[0]) * 10**6 / (2 * np.pi), popt_T2, pcov_T2
