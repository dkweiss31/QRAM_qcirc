import copy

import dynamiqs as dq
import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from dynamiqs.solver import Tsit5, Expm
from quantum_utils import write_to_h5
from scipy.constants import hbar, k
from scipy.optimize import curve_fit

dq.set_precision("double")


class RamseyExperiment:
    def __init__(
        self,
        interference=True,
        omega_tmon=2.0 * np.pi * 5.0,
        omega_cavs=(2.0 * np.pi * 4.0, ),
        chi_cavstmon=(2.0 * np.pi * 0.001, ),
        chi_crosscav=(2.0 * np.pi * 0.0, ),
        kappa_cavs=(2.0 * np.pi * 0.04, ),
        EJ=2.0 * np.pi * 32.69,
        alpha=2.0 * np.pi * (-0.124),
        temp=0.1,
        tmon_dim=2,
        cavity_dim=4,
        num_cavs=1,
        thermal_time=200.0,
        delay_times=np.linspace(0.0, 2000, 301),
        omega_d_tmon=2.0 * np.pi * (5.7423 - 0.0071),
        omega_d_cav: float = 2.0 * np.pi * 4.5,
        tmon_drive_amp: float = 2.0 * np.pi * 0.01,
        nsteps=1000,
        atol=1e-8,
        rtol=1e-6,
        destructive_interference=1,
        interference_scale=1.0,
        exp_type='ramsey',
    ):
        assert len(omega_cavs) == len(chi_cavstmon) == len(kappa_cavs) == num_cavs
        self.interference = interference
        self.omega_tmon = omega_tmon
        self.omega_cavs = omega_cavs
        self.chi_cavstmon = chi_cavstmon
        self.chi_crosscav = chi_crosscav
        self.kappa_cavs = kappa_cavs
        self.EJ = EJ
        self.alpha = alpha
        self.temp = temp
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim
        self.num_cavs = num_cavs
        self.thermal_time = thermal_time
        self.delay_times = delay_times
        self.omega_d_tmon = omega_d_tmon
        self.omega_d_cav = omega_d_cav
        self.tmon_drive_amp = tmon_drive_amp
        self.nsteps = nsteps
        self.atol = atol
        self.rtol = rtol
        self.destructive_interference = destructive_interference
        self.interference_scale = interference_scale
        self.exp_type = exp_type
        self.truncated_dims = num_cavs * [cavity_dim] + [tmon_dim]

    def tmon_ops(self):
        ids = self.num_cavs * [dq.eye(self.cavity_dim)]
        q = dq.tensor(*ids, dq.destroy(2))
        return q

    def annihilation_ops(self):
        if self.num_cavs == 1:
            return [dq.tensor(dq.destroy(self.cavity_dim), dq.eye(2)), ]
        elif self.num_cavs == 2:
            return [dq.tensor(dq.destroy(self.cavity_dim), dq.eye(self.cavity_dim), dq.eye(2)),
                    dq.tensor(dq.eye(self.cavity_dim), dq.destroy(self.cavity_dim), dq.eye(2))]
        else:
            raise ValueError("number of cavities needs to be 1 or 2")

    def bare_labels(self):
        return np.array(list(np.ndindex(*self.truncated_dims)))

    def rotating_frame_frequencies(self):
        return self.num_cavs * [self.omega_d_cav] + [self.omega_d_tmon]

    def nths(self):
        # omegas = np.array(len(self.omega_cavs) * [np.max(self.omega_cavs) + 0.2 * 2.0 * np.pi])
        return 1.0 / (np.exp(hbar * self.omega_cavs * 10**9 / (k * self.temp)) - 1)

    def phi_cav(self, idx=0):
        return (self.chi_cavstmon[idx]**2 / (2 * np.abs(self.alpha) * self.EJ))**(1/4)

    def phi_q(self):
        return (2 * np.abs(self.alpha) / self.EJ)**(1/4)

    def cos_normal_ordered(self, phi, a_op):
        dim = a_op.shape[0] + 1
        H = 0.0 * a_op
        for n in range(dim):
            for m in range(dim):
                if (n + m) % 2 == 0:
                    H += self._cosine_normal_ordered_term(phi, a_op, n, m)
        return H

    def sin_normal_ordered(self, phi, a_op):
        dim = a_op.shape[0] + 1
        H = 0.0 * a_op
        for n in range(dim):
            for m in range(dim):
                if (n + m) % 2 == 1:
                    H += self._sine_normal_ordered_term(phi, a_op, n, m)
        return H

    def _cosine_normal_ordered_term(self, phi, a_op, n, m):
        overall_pref = np.exp(-0.5 * phi ** 2)
        pref = ((-phi ** 2) ** ((n + m) / 2)
                / sp.special.factorial(n)
                / sp.special.factorial(m)
                )
        return overall_pref * pref * dq.powm(dq.dag(a_op), n) @ dq.powm(a_op, m)

    def _sine_normal_ordered_term(self, phi, a_op, n, m):
        overall_pref = np.exp(-0.5 * phi ** 2) * phi
        pref = ((-phi ** 2) ** ((n + m - 1) / 2)
                / sp.special.factorial(n)
                / sp.special.factorial(m)
                )
        return overall_pref * pref * dq.powm(dq.dag(a_op), n) @ dq.powm(a_op, m)

    def hamiltonian_full(self):
        a_ops = self.annihilation_ops()
        q = self.tmon_ops()
        H0 = self._static_hamiltonian()
        if len(a_ops) == 1:
            phi_a, phi_q = self.phi_cav(0), self.phi_q()
            H0 += (-self.EJ
                   * (self.cos_normal_ordered(phi_a, a_ops[0])
                      @ self.cos_normal_ordered(phi_q, q)
                      - self.sin_normal_ordered(phi_a, a_ops[0])
                      @ self.sin_normal_ordered(phi_q, q)
                      )
                   )
            harm_indices = list(np.ndindex(2, 2, 2, 2))
            for (j, k, l, m) in harm_indices:
                if (j + k) % 2 == 0 and (l + m) % 2 == 0 and j + k + l + m == 2:
                    H0 += self.EJ * self._cosine_normal_ordered_term(
                        phi_a, a_ops[0], j, k
                    ) @ self._cosine_normal_ordered_term(phi_q, q, l, m)
                if (j + k) % 2 == 1 and (l + m) % 2 == 1 and j + k + l + m == 0:
                    H0 += (-self.EJ
                           @ self._sine_normal_ordered_term(
                                phi_a, a_ops[0], j, k
                            )
                           @ self._sine_normal_ordered_term(phi_q, q, l, m)
                           )
        elif len(a_ops) == 2:
            a, b = a_ops[0], a_ops[1]
            phi_a, phi_b, phi_q = self.phi_cav(0), self.phi_cav(1), self.phi_q()
            term_1 = -self.EJ * (
                self.cos_normal_ordered(phi_a, a)
                @ self.cos_normal_ordered(phi_b, b)
                * np.cos(phi_q)
            )
            term_2 = self.EJ * (
                self.cos_normal_ordered(phi_a, a)
                @ self.sin_normal_ordered(phi_b, b)
                @ (np.sin(phi_q) * (q + dq.dag(q)))
            )
            term_3 = self.EJ * (
                    self.sin_normal_ordered(phi_a, a)
                    @ self.cos_normal_ordered(phi_b, b)
                    @ (np.sin(phi_q) * (q + dq.dag(q)))
            )
            term_4 = self.EJ * (
                    self.sin_normal_ordered(phi_a, a)
                    @ self.sin_normal_ordered(phi_b, b)
                    * np.cos(phi_q)
            )
            # TODO harmonic subtraction
            H0 += term_1 + term_2 + term_3 + term_4 # + harm_term
        else:
            raise ValueError("a_ops can have one or two operators")
        bare_labels = self.bare_labels()
        omega_ds = self.rotating_frame_frequencies()
        rot_frame_diag = np.einsum("nj,j->n", bare_labels, omega_ds)
        rot_frame = np.reshape(rot_frame_diag, (-1, 1)) - rot_frame_diag
        H0 = H0 - H0[0, 0] * dq.eye(*self.truncated_dims)
        Ht = jnp.where(
            jnp.abs(rot_frame) > 2.0 * np.pi * 0.0, 0.0, H0
        )
        # note elementwise multiplication below
        # Ht = dq.timecallable(lambda t: jnp.exp(1j * t * rot_frame) * H0)
        return Ht

    def hamiltonian(self):
        a_ops = self.annihilation_ops()
        q = self.tmon_ops()
        H0 = self._static_hamiltonian()
        for idx, a_op in enumerate(a_ops):
            H0 += self.chi_cavstmon[idx] * dq.dag(a_op) @ a_op @ dq.dag(q) @ q
        if len(a_ops) == 1:
            a = a_ops[0]
            phi_a, phi_q = self.phi_cav(0), self.phi_q()
            H0 += (-self.EJ / 24) * (
                    12 * phi_a ** 2 * (phi_a ** 2 + phi_q ** 2) * dq.dag(a) @ a
            )
        if len(a_ops) == 2:
            a, b = a_ops[0], a_ops[1]
            H0 += self.chi_crosscav[0] * dq.dag(a) @ a @ dq.dag(b) @ b
            phi_a, phi_b, phi_q = self.phi_cav(0), self.phi_cav(1), self.phi_q()
            # check that the expressions for phi_a and phi_q are correct
            assert np.allclose(self.chi_cavstmon[0], -self.EJ * phi_a ** 2 * phi_q ** 2)
            assert np.allclose(self.chi_cavstmon[1], -self.EJ * phi_b ** 2 * phi_q ** 2)
            assert np.allclose(self.chi_crosscav[0], -self.EJ * phi_a**2 * phi_b**2)
            H0 += (-self.EJ / 24) * (
                    self.interference_scale * 24 * phi_a * phi_b * phi_q**2
                    * dq.dag(q) @ q @ (dq.dag(a) @ b + dq.dag(b) @ a)
            )
        return H0

    def _static_hamiltonian(self):
        a_ops = self.annihilation_ops()
        q = self.tmon_ops()
        H0 = sum(
            (omega - self.omega_d_cav) * dq.dag(a_op) @ a_op
            for (omega, a_op) in zip(self.omega_cavs, a_ops)
        )
        if self.exp_type == "ramsey":
            H0 += (self.omega_tmon - self.omega_d_tmon) * dq.dag(q) @ q
        elif self.exp_type == "T1":
            # pulse should be resonant
            H0 += 0.0 * dq.dag(q) @ q
        else:
            raise ValueError(f"exp_type needs to be ramsey or T1 but got {self.exp_type}")
        return H0

    def construct_c_ops_interference(self):
        a_ops = self.annihilation_ops()
        if len(a_ops) == 1:
            return self.construct_c_ops_no_interference()
        elif len(a_ops) == 2:
            kappa_cavs = self.kappa_cavs
            nths = self.nths()
            lowering_1 = (np.sqrt(kappa_cavs[0] * (1 + nths[0])) * a_ops[0]
                          - self.destructive_interference * np.sqrt(kappa_cavs[1] * (1 + nths[1])) * a_ops[1])
            raising_1 = (np.sqrt(kappa_cavs[0] * nths[0]) * dq.dag(a_ops[0])
                         - self.destructive_interference * np.sqrt(kappa_cavs[1] * nths[1]) * dq.dag(a_ops[1]))
            return [lowering_1, raising_1]
        else:
            raise RuntimeError("more than two cavities not supported")

    def construct_c_ops_no_interference(self):
        individual_lowering = [
            np.sqrt(kappa * (1 + nth)) * a_op
            for (kappa, nth, a_op) in zip(
                self.kappa_cavs, self.nths(), self.annihilation_ops()
            )
        ]
        individual_raising = [
            np.sqrt(kappa * nth) * dq.dag(a_op)
            for (kappa, nth, a_op) in zip(
                self.kappa_cavs, self.nths(), self.annihilation_ops()
            )
        ]
        return individual_lowering + individual_raising

    @staticmethod
    def T2_func(t, t2, omega, a, b, phi):
        return a * np.exp(-t / t2) * np.cos(omega * t + phi) + b

    @staticmethod
    def T1_func(t, t1, a, b):
        return a * np.exp(-t / t1) + b

    @staticmethod
    def T1_gauss_func(t, t1, a, b):
        return a * np.exp(-0.5 * (t / t1)**2) + b

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

    def mesolve_for_final_state(self, H, init_dm, t, solver=Tsit5()):
        if self.interference:
            c_ops = self.construct_c_ops_interference()
        else:
            c_ops = self.construct_c_ops_no_interference()
        e_ops = [dq.dag(a) @ a for a in self.annihilation_ops()]
        options = dq.Options(
            save_states=True, progress_meter=None,
        )
        result = dq.mesolve(
            H,
            c_ops,
            init_dm,
            (0, t),
            exp_ops=e_ops,
            solver=solver,
            options=options,
        )
        # take occupation of first cavity as representative
        print("expectation value of n_1: ", np.max(result.expects[0]))
        return result.final_state

    def readout_proj(self, tmon_idx=0):
        op_list = self.num_cavs * [dq.eye(self.cavity_dim)] + [
            dq.basis(self.tmon_dim, tmon_idx) @ dq.dag(dq.basis(self.tmon_dim, tmon_idx))
        ]
        return dq.tensor(*op_list)

    def decoherence_experiment(self, full_cosine=False):
        final_prob = np.zeros_like(self.delay_times)
        q = self.tmon_ops()
        initial_state = dq.fock(
            self.truncated_dims, len(self.truncated_dims) * [0]
        )
        if self.exp_type == "ramsey":
            t_drive = np.pi / (2 * self.tmon_drive_amp)
        elif self.exp_type == "T1":
            t_drive = np.pi / self.tmon_drive_amp
        else:
            raise ValueError(f"exp_type needs to be ramsey or T1 but got {self.exp_type}")
        if full_cosine:
            H = self.hamiltonian_full()
            solver = Tsit5(max_steps=self.nsteps, atol=self.atol, rtol=self.rtol)
        else:
            H = self.hamiltonian()
            solver = Expm()
        thermal_state = self.mesolve_for_final_state(
            H, initial_state, self.thermal_time
        )
        # Hamiltonian for pi/2 pulses
        H_with_drive = copy.deepcopy(H)
        H_with_drive += 0.5 * self.tmon_drive_amp * (q + dq.dag(q))
        readout = self.readout_proj()
        # pi/2 or pi pulses
        state_after_prev_delay = self.mesolve_for_final_state(
            H_with_drive, thermal_state, t_drive, solver=solver,
        )
        final_state = self.mesolve_for_final_state(
            H_with_drive, state_after_prev_delay, t_drive, solver=solver,
        )
        final_prob[0] = np.real(np.trace(final_state @ readout))
        delay_dif = self.delay_times[1] - self.delay_times[0]
        for idx in range(1, len(self.delay_times)):
            # run the delay
            state_after_delay = self.mesolve_for_final_state(
                H, state_after_prev_delay, delay_dif, solver=solver,
            )
            final_state = self.mesolve_for_final_state(
                H_with_drive, state_after_delay, t_drive, solver=solver,
            )
            state_after_prev_delay = state_after_delay
            final_prob[idx] = np.real(np.trace(final_state @ readout))
        return final_prob

    def plot_ramsey(self, ramsey_result, popt_T2, filename=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.delay_times, ramsey_result, "o")
        plot_times = np.linspace(0.0, self.delay_times[-1], 2000)
        ax.plot(plot_times, self.T2_func(plot_times, *popt_T2), linestyle="-")
        ax.set_ylabel(r"$P(|1\rangle)$", fontsize=12)
        ax.set_xlabel("time [ns]", fontsize=12)
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_T1(self, ramsey_result, popt_T1, filename=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.delay_times, ramsey_result, "o")
        plot_times = np.linspace(0.0, self.delay_times[-1], 2000)
        ax.plot(plot_times, self.T1_func(plot_times, *popt_T1), linestyle="-")
        ax.set_ylabel(r"$P(|1\rangle)$", fontsize=12)
        ax.set_xlabel("time [ns]", fontsize=12)
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def main_ramsey(
        self,
        filepath,
        p0=(6 * 10 ** 4, 0.045, 0.5, 0.5, -1.7),
        full_cosine=False,
    ):
        print(f"Saving to filepath {filepath}. Running sim with params")
        print(self.__dict__)
        naive_gamma_phi = sum(
            self.gamma_phi_full_func()
        )
        ramsey_result = self.decoherence_experiment(full_cosine=full_cosine)
        write_to_h5(filepath, {"ramsey_result": ramsey_result}, self.__dict__)
        if self.exp_type == "ramsey":
            gamma_phi, popt, pcov = self.extract_gammaphi(
                ramsey_result, p0=p0
            )
            # write this separately in case the fit fails
            with h5py.File(filepath, "a") as f:
                written_data = f.create_dataset("gamma_phi", data=gamma_phi)
            print(f"naive gamma_phi = {naive_gamma_phi}")
            print(f"extracted gamma_phi = {gamma_phi}")
            print(f"optimized parameters {popt}")
        elif self.exp_type == "T1":
            gamma_1, popt, pcov = self.extract_gamma1(
                ramsey_result, p0=p0
            )
            gamma_1_gauss, popt_gauss, pcov_gauss = self.extract_gamma1_gauss(
                ramsey_result, p0=p0
            )
            # write this separately in case the fit fails
            with h5py.File(filepath, "a") as f:
                written_data = f.create_dataset("gamma_1", data=gamma_1)
                written_data = f.create_dataset("gamma_1_gauss", data=gamma_1_gauss)
            print(f"extracted gamma_1 = {gamma_1}")
            print(f"extracted gamma_1_gauss = {gamma_1_gauss}")
            print(f"optimized parameters {popt}")
            print(f"optimized parameters gauss {popt_gauss}")

    def extract_gamma1(
        self,
        ramsey_result,
        window=None,
        p0=(6 * 10 ** 4, 1.0, 0.0),
        plot=True,
    ):
        if window is None:
            window = (0, len(self.delay_times))
        popt_T1, pcov_T1 = curve_fit(
            self.T1_func,
            self.delay_times[window[0]: window[1]],
            ramsey_result[window[0]: window[1]],
            p0=p0,
            maxfev=6000,
            bounds=((100, -2, -2), (10**15, 2, 2)),
        )
        if plot:
            self.plot_T1(ramsey_result, popt_T1)
        print("popt: ", popt_T1)
        print("pcov: ", pcov_T1)
        return (1 / popt_T1[0]) * 10**6 / (2 * np.pi), popt_T1, pcov_T1

    def extract_gamma1_gauss(
        self,
        ramsey_result,
        window=None,
        p0=(6 * 10 ** 4, 1.0, 0.0),
        plot=True,
    ):
        if window is None:
            window = (0, len(self.delay_times))
        popt_T1, pcov_T1 = curve_fit(
            self.T1_gauss_func,
            self.delay_times[window[0]: window[1]],
            ramsey_result[window[0]: window[1]],
            p0=p0,
            maxfev=6000,
            bounds=((100, -2, -2), (10**15, 2, 2)),
        )
        if plot:
            self.plot_T1(ramsey_result, popt_T1)
        print("popt gauss: ", popt_T1)
        print("pcov gauss: ", pcov_T1)
        return (1 / popt_T1[0]) * 10**6 / (2 * np.pi), popt_T1, pcov_T1

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
            self.delay_times[window[0]: window[1]],
            ramsey_result[window[0]: window[1]],
            p0=p0,
            maxfev=6000,
            bounds=((100, -2, -2, -2, -np.pi), (10**15, 2, 2, 2, np.pi)),
        )
        if plot:
            self.plot_ramsey(ramsey_result, popt_T2)
        print("popt: ", popt_T2)
        print("pcov: ", pcov_T2)
        print("condition_num: ", np.linalg.cond(pcov_T2))
        return (1 / popt_T2[0]) * 10**6 / (2 * np.pi), popt_T2, pcov_T2


class CoherentDephasing(RamseyExperiment):
    def __init__(
        self,
        epsilon_array=(2.0 * np.pi * 0.001,),
        include_cr=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon_array = epsilon_array
        self.include_cr = include_cr

    def hamiltonian_full(self):
        H = super().hamiltonian_full()
        a_ops = self.annihilation_ops()
        a = a_ops[0]
        eps_a = self.epsilon_array[0]
        H += eps_a * a + np.conj(eps_a) * dq.dag(a)
        if len(a_ops) == 2:
            b = a_ops[1]
            eps_b = self.epsilon_array[1]
            H += eps_b * b + np.conj(eps_b) * dq.dag(b)
        return H

    def hamiltonian(self):
        H = super().hamiltonian()
        a_ops = self.annihilation_ops()
        if len(a_ops) == 1:
            a = a_ops[0]
            eps = self.epsilon_array[0]
            H += eps * a + np.conj(eps) * dq.dag(a)
        elif len(a_ops) == 2:
            a, b = a_ops[0], a_ops[1]
            eps_a, eps_b = self.epsilon_array[0], self.epsilon_array[1]
            H += -1j * (b * eps_b - a * eps_a)
            H += 1j * (dq.dag(b) * np.conj(eps_b) - dq.dag(a) * np.conj(eps_a))
        else:
            raise RuntimeError("only one or two cavities supported")
        return H
