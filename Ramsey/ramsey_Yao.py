import copy
import warnings

import h5py
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

from QRAM_utils.utils import id_wrap_ops, construct_basis_states_list, write_to_h5


class RamseyExperiment:
    def __init__(
        self,
        interference=True,
        omega_tmon=2.0 * np.pi * 5.0,
        omega_cavs=(2.0 * np.pi * 4.0, ),
        chi_cavstmon=(2.0 * np.pi * 0.001, ),
        kappa_cavs=(2.0 * np.pi * 0.04, ),
        temp=0.1,
        tmon_dim=2,
        cavity_dim=4,
        num_cavs=1,
        thermal_time=200.0,
        delay_times=np.linspace(0.0, 2000, 301),
        omega_d_tmon=2.0 * np.pi * (5.7423 - 0.0071),
        tmon_drive_amp: float = 2.0 * np.pi * 0.01,
        nsteps=1000,
        atol=1e-8,
        rtol=1e-6,
    ):
        assert len(omega_cavs) == len(chi_cavstmon) == len(kappa_cavs) == num_cavs
        self.interference = interference
        self.omega_tmon = omega_tmon
        self.omega_cavs = omega_cavs
        self.chi_cavstmon = chi_cavstmon
        self.kappa_cavs = kappa_cavs
        self.temp = temp
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim
        self.num_cavs = num_cavs
        self.thermal_time = thermal_time
        self.delay_times = delay_times
        self.omega_d_tmon = omega_d_tmon
        self.tmon_drive_amp = tmon_drive_amp
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
        return [
            H0,
        ]

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

    def obtain_thermal_state(self, initial_state=None):
        if self.thermal_time < 2.0 / np.min(self.kappa_cavs):
            print("running for too short of a time to get a thermal state")
        if initial_state is None:
            fock_spec = tuple(len(self.truncated_dims) * [0])
            (initial_state,) = construct_basis_states_list(
                [
                    fock_spec,
                ],
                self.truncated_dims,
            )
        H = self.hamiltonian()
        return self.mesolve_for_final_state(
            H, initial_state * initial_state.dag(), self.thermal_time
        )

    @staticmethod
    def T2_func(t, t2, omega, a, b, phi):
        return a * np.exp(-t / t2) * np.cos(omega * t + phi) + b

    @staticmethod
    def T1_func(t, t1, a, b):
        return a * np.exp(-t / t1) + b

    def gamma_phi_func(self):
        return (
            (
                self.chi_cavstmon ** 2
                * self.nths()
                * (1 + self.nths())
                / self.kappa_cavs
            )
            * 10**6
            / (2 * np.pi)
        )

    def gamma_phi_full_func(self):
        return (
            (
                self.chi_cavstmon ** 2
                * self.nths()
                * (1 + self.nths())
                * self.kappa_cavs
                / (self.kappa_cavs ** 2 + self.chi_cavstmon ** 2)
            )
            * 10**6
            / (2 * np.pi)
        )

    def mesolve_for_final_state(self, H, init_dm, t):
        if self.interference:
            c_ops = self.construct_c_ops_interference()
        else:
            c_ops = self.construct_c_ops_no_interference()
        e_ops = [a.dag() * a for a in self.annihilation_ops()]
        options = Options(
            store_final_state=True, nsteps=self.nsteps, atol=self.atol, rtol=self.rtol
        )
        result = mesolve(
            H,
            init_dm,
            (0, t),
            c_ops=c_ops,
            e_ops=e_ops,
            options=options,
        )
        print(np.max(result.expect))
        if np.max(result.expect) >= self.cavity_dim - 1:
            warnings.WarningMessage("likely need to increase cavity_dim")
        return result.final_state

    def readout_proj(self):
        op_list = self.num_cavs * [qeye(self.cavity_dim)] + [
            basis(self.tmon_dim, 0) * basis(self.tmon_dim, 0).dag()
        ]
        return tensor(*op_list)

    def ramsey_experiment(self):
        final_prob = np.zeros_like(self.delay_times)
        (sx, sy, sz) = self.tmon_Pauli_ops()
        thermal_state = self.obtain_thermal_state()
        H0_q = -0.5 * (self.omega_tmon - self.omega_d_tmon) * sz
        H = self.hamiltonian()
        H[0] += H0_q
        # Hamiltonian for pi/2 pulses
        H_with_drive = copy.deepcopy(H)
        H_with_drive[0] += 0.5 * self.tmon_drive_amp * sx
        t_pi2 = np.pi / (2 * self.tmon_drive_amp)
        # pi/2 pulses
        state_after_prev_delay = self.mesolve_for_final_state(
            H_with_drive, thermal_state, t_pi2
        )
        final_state = self.mesolve_for_final_state(
            H_with_drive, state_after_prev_delay, t_pi2
        )
        final_prob[0] = np.real(np.trace(final_state * self.readout_proj()))
        delay_dif = self.delay_times[1] - self.delay_times[0]
        for idx in range(1, len(self.delay_times)):
            # run the delay
            state_after_delay = self.mesolve_for_final_state(
                H, state_after_prev_delay, delay_dif
            )
            # final pi/2 pulse
            final_state = self.mesolve_for_final_state(
                H_with_drive, state_after_delay, t_pi2
            )
            state_after_prev_delay = state_after_delay
            final_prob[idx] = np.real(np.trace(final_state * self.readout_proj()))
        return final_prob

    def plot_ramsey(self, ramsey_result, popt_T2, filename=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.delay_times, ramsey_result, "o")
        plot_times = np.linspace(0.0, self.delay_times[-1], 2000)
        ax.plot(plot_times, self.T2_func(plot_times, *popt_T2), linestyle="-")
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def main_ramsey(self, filepath, p0=(6 * 10 ** 4, 0.045, 0.5, 0.2, -1)):
        naive_gamma_phi = sum(
            self.gamma_phi_full_func()
        )
        ramsey_result = self.ramsey_experiment()
        write_to_h5(filepath, {"ramsey_result": ramsey_result}, self.__dict__)
        gamma_phi_indep, popt, pcov = self.extract_gammaphi(
            ramsey_result, p0=p0
        )
        # write this separately in case the fit fails
        with h5py.File(filepath, "a") as f:
            written_data = f.create_dataset("gamma_phi", data=gamma_phi_indep)
        print(f"naive gamma_phi = {naive_gamma_phi}")
        print(f"indep gamma_phi = {gamma_phi_indep}")

    def extract_gammaphi(
        self,
        ramsey_result,
        window=None,
        p0=(6 * 10**4, 0.045, 0.5, 0.2, 0),
        plot=True,
    ):
        if window is None:
            window = (0, len(self.delay_times))
        popt_T2, pcov_T2 = curve_fit(
            self.T2_func,
            self.delay_times[window[0] : window[1]],
            ramsey_result[window[0] : window[1]],
            p0=p0,
            maxfev=6000,
            bounds=((100, -2, -2, -2, -np.pi), (10**15, 2, 2, 2, np.pi)),
        )
        if plot:
            self.plot_ramsey(ramsey_result, popt_T2)
        return (1 / popt_T2[0]) * 10**6 / (2 * np.pi), popt_T2, pcov_T2


class CoherentDephasing(RamseyExperiment):
    def __init__(
        self,
        interference=True,
        omega_tmon=2.0 * np.pi * 5.0,
        omega_cavs=(2.0 * np.pi * 4.0,),
        chi_cavstmon=(2.0 * np.pi * 0.001,),
        kappa_cavs=(2.0 * np.pi * 0.04,),
        temp=0.1,
        tmon_dim=2,
        cavity_dim=4,
        num_cavs=1,
        thermal_time=200.0,
        delay_times=np.linspace(0.0, 2000, 301),
        omega_d_tmon=2.0 * np.pi * (5.7423 - 0.0071),
        tmon_drive_amp: float = 2.0 * np.pi * 0.01,
        omega_d_cav=2.0 * np.pi * 3.0,
        epsilon_array=(2.0 * np.pi * 0.001, ),
        nsteps=1000,
        atol=1e-8,
        rtol=1e-6,
    ):
        super().__init__(
            interference,
            omega_tmon,
            omega_cavs,
            chi_cavstmon,
            kappa_cavs,
            temp,
            tmon_dim,
            cavity_dim,
            num_cavs,
            thermal_time,
            delay_times,
            omega_d_tmon,
            tmon_drive_amp,
            nsteps,
            atol,
            rtol,
        )
        self.omega_d_cav = omega_d_cav
        self.epsilon_array = epsilon_array

    def hamiltonian(self):
        H = super().hamiltonian()
        for (eps, a) in zip(self.epsilon_array, self.annihilation_ops()):
            H[0] += -self.omega_d_cav * a.dag() * a
            H[0] += eps * a + np.conj(eps) * a.dag()
        return H
