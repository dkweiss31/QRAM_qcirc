import copy

import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    destroy,
    sigmax,
    sigmay,
    sigmaz,
    mesolve,
    Options,
    qeye,
    tensor,
    basis,
)
from scipy.constants import hbar, k
from scipy.optimize import curve_fit

from utils.utils import id_wrap_ops, construct_basis_states_list


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
        nsteps=1000,
        atol=1e-8,
        rtol=1e-6,
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
        self.nsteps = nsteps
        self.atol = atol
        self.rtol = rtol
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
        H0 = sum(
            omega * a_op.dag() * a_op
            for (omega, a_op) in zip(self.omega_cavs, self.annihilation_ops())
        )
        for idx, a_op in enumerate(annihilation_ops_list):
            H0 += 0.5 * self.chi_cavstmon[idx] * a_op.dag() * a_op * sz
        return [H0, ]

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
        self, total_time: float = 200, initial_state=None, interference=True
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
        H0 = self.hamiltonian()
        if interference:
            c_ops = self.construct_c_ops_interference()
        else:
            c_ops = self.construct_c_ops_no_interference()
        options = Options(
            store_final_state=True, nsteps=self.nsteps, atol=self.atol, rtol=self.rtol
        )
        result_thermal = mesolve(
            H0,
            initial_state * initial_state.dag(),
            (0, total_time),
            c_ops,
            options=options,
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
        return (
            (chi**2 * nth * (1 + nth) * kappa / (kappa**2 + chi**2))
            * 10**6
            / (2 * np.pi)
        )

    def pi_2_pulse(self, H, init_dm, t, c_ops):
        options = Options(
            store_final_state=True, nsteps=self.nsteps, atol=self.atol, rtol=self.rtol
        )
        return mesolve(
            H,
            init_dm,
            (0, t),
            c_ops=c_ops,
            options=options,
        )

    def readout_proj(self):
        op_list = self.num_cavs * [qeye(self.cavity_dim)] + [
            basis(self.tmon_dim, 0) * basis(self.tmon_dim, 0).dag()
        ]
        return tensor(*op_list)

    def ramsey_indep(self, thermal_state, delay_times, omega_d, c_ops, tmon_drive_amp: float = 2.0 * np.pi * 0.01):
        (sx, sy, sz) = self.tmon_Pauli_ops()
        H0_q = -0.5 * (self.omega_tmon - omega_d) * sz
        H = self.hamiltonian()
        H[0] += H0_q
        H_with_drive = copy.deepcopy(H)
        H_with_drive[0] += 0.5 * tmon_drive_amp * sx
        t_pi2 = np.pi / (2 * tmon_drive_amp)
        readout_proj = self.readout_proj()
        final_prob = np.zeros_like(delay_times)
        state_after_prev_delay = self.pi_2_pulse(H_with_drive, thermal_state, t_pi2, c_ops).final_state
        final_state = self.pi_2_pulse(H_with_drive, state_after_prev_delay, t_pi2, c_ops).final_state
        final_prob[0] = np.real(np.trace(final_state * readout_proj))
        delay_dif = delay_times[1] - delay_times[0]
        for idx in range(1, len(delay_times)):
            options = Options(
                store_final_state=True,
                nsteps=self.nsteps,
                atol=self.atol,
                rtol=self.rtol,
            )
            state_after_delay = mesolve(
                H,
                state_after_prev_delay,
                (0, delay_dif),
                c_ops=c_ops,
                options=options,
            ).final_state
            final_state = self.pi_2_pulse(H_with_drive, state_after_delay, t_pi2, c_ops).final_state
            state_after_prev_delay = state_after_delay
            final_prob[idx] = np.real(np.trace(final_state * self.readout_proj()))
        return final_prob

    def plot_ramsey(self, ramsey_result, delay_times, popt_T2, filename=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(delay_times, ramsey_result, "o")
        plot_times = np.linspace(0.0, delay_times[-1], 2000)
        ax.plot(plot_times, self.T2_func(plot_times, *popt_T2), linestyle="-")
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def extract_gammaphi(
        self,
        ramsey_result,
        delay_times,
        window=None,
        p0=(6 * 10**4, 0.045, 0.5, 0.2, 0),
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
            bounds=((100, -2, -2, -2, -np.pi), (10**15, 2, 2, 2, np.pi)),
        )
        if plot:
            self.plot_ramsey(ramsey_result, delay_times, popt_T2)
        return (1 / popt_T2[0]) * 10**6 / (2 * np.pi), popt_T2, pcov_T2


# class
