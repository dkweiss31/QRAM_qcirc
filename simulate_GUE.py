from qutip import (
    destroy,
    mesolve,
    Options, Qobj, tensor,
)
import numpy as np
from scipy.special import erf

from utils import id_wrap_ops, construct_basis_states_list


class SimulateGUE:
    """compute the fidelity of state transfer for GUEs"""

    def __init__(
            self,
            gamma_b_avg,
            gamma_c_avg,
            gamma_b_dev,
            gamma_c_dev,
            cavity_dim: int = 2,
    ):
        self.cavity_dim = cavity_dim
        # this is for state transfer one way. think about including two way
        self.truncated_dims = 8 * [cavity_dim]
        b1_idx = 0
        b2_idx = 1
        c1_idx = 2
        c2_idx = 3
        b1_r_idx = 4
        b2_r_idx = 5
        c1_r_idx = 6
        c2_r_idx = 7
        self.phi = -np.pi / 2

        self.b1 = id_wrap_ops(destroy(cavity_dim), b1_idx, self.truncated_dims)
        self.b2 = id_wrap_ops(destroy(cavity_dim), b2_idx, self.truncated_dims)
        self.c1 = id_wrap_ops(destroy(cavity_dim), c1_idx, self.truncated_dims)
        self.c2 = id_wrap_ops(destroy(cavity_dim), c2_idx, self.truncated_dims)

        self.b1_r = id_wrap_ops(destroy(cavity_dim), b1_r_idx, self.truncated_dims)
        self.b2_r = id_wrap_ops(destroy(cavity_dim), b2_r_idx, self.truncated_dims)
        self.c1_r = id_wrap_ops(destroy(cavity_dim), c1_r_idx, self.truncated_dims)
        self.c2_r = id_wrap_ops(destroy(cavity_dim), c2_r_idx, self.truncated_dims)

        self.gamma_b_1 = gamma_b_avg + 0.5 * gamma_b_dev
        self.gamma_b_2 = gamma_b_avg - 0.5 * gamma_b_dev
        self.gamma_c_1 = gamma_c_avg + 0.5 * gamma_c_dev
        self.gamma_c_2 = gamma_c_avg - 0.5 * gamma_c_dev

    def collective_loss_ops(self):
        L_R_b = (
                np.sqrt(self.gamma_b_1) * self.b1_r
                - 1j * np.sqrt(self.gamma_b_2) * self.b2_r
        )
        L_R_c = (
                np.exp(-1j * self.phi)
                * (-1j)
                * (
                        np.sqrt(self.gamma_c_1) * self.c1_r
                        - 1j * np.sqrt(self.gamma_c_2) * self.c2_r
                )
        )
        L_L_b = (
                np.sqrt(self.gamma_b_1) * self.b1_r
                + 1j * np.sqrt(self.gamma_b_2) * self.b2_r
        )
        L_L_c = (
                np.exp(1j * self.phi)
                * 1j
                * (
                        np.sqrt(self.gamma_c_1) * self.c1_r
                        + 1j * np.sqrt(self.gamma_c_2) * self.c2_r
                )
        )
        return L_R_b, L_R_c, L_L_b, L_L_c

    def construct_c_ops(self, Gamma_1_cav=0.0, Gamma_1_transfer_nr=0.0, nth=0.0):
        L_R_b, L_R_c, L_L_b, L_L_c = self.collective_loss_ops()
        return [
            L_R_b + L_R_c,
            L_L_b + L_L_c,
            np.sqrt(Gamma_1_cav) * self.b1,
            np.sqrt(Gamma_1_cav) * self.b2,
            np.sqrt(Gamma_1_cav) * self.c1,
            np.sqrt(Gamma_1_cav) * self.c2,
            np.sqrt(nth * Gamma_1_cav) * self.b1.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.b2.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.c1.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.c2.dag(),
            np.sqrt(Gamma_1_transfer_nr) * self.b1_r,
            np.sqrt(Gamma_1_transfer_nr) * self.b2_r,
            np.sqrt(Gamma_1_transfer_nr) * self.c1_r,
            np.sqrt(Gamma_1_transfer_nr) * self.c2_r,
            np.sqrt(nth * Gamma_1_transfer_nr) * self.b1_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.b2_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.c1_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.c2_r.dag(),
        ]

    def gamma_b_func(self, t, args=None):
        c = args["c"]
        B = args["B"]
        gamma_b = args["gamma_b_avg"]
        t_half = args["t_half"]
        scale_b = args["scale_b"]
        return (
                scale_b
                * np.sqrt(gamma_b)
                * np.sqrt(
            (
                    0.5
                    * np.exp(-c * (t - t_half) ** 2)
                    / (
                            (1 / B)
                            - np.sqrt(np.pi / (4 * c)) * erf(np.sqrt(c) * (t - t_half))
                    )
            )
        )
        )

    def gamma_c_func(self, t, args=None):
        t_half = args["t_half"]
        scale_c = args["scale_c"]
        return scale_c * self.gamma_b_func(-t + 2 * t_half, args=args)

    def hamiltonian(self):
        L_R_b, L_R_c, L_L_b, L_L_c = self.collective_loss_ops()
        H0_r_half = -0.5 * 1j * (L_R_c.dag() * L_R_b + L_L_b.dag() * L_L_c)
        H0_r = H0_r_half + H0_r_half.dag()
        H_int_b_1 = self.b1 * self.b1_r.dag() + self.b1.dag() * self.b1_r
        H_int_b_2 = self.b2 * self.b2_r.dag() + self.b2.dag() * self.b2_r
        H_int_c_1 = self.c1 * self.c1_r.dag() + self.c1.dag() * self.c1_r
        H_int_c_2 = self.c2 * self.c2_r.dag() + self.c2.dag() * self.c2_r
        return H0_r, H_int_b_1, H_int_b_2, H_int_c_1, H_int_c_2

    def run_state_transfer(
            self,
            init_state,
            args,
            c_ops=None,
            e_ops=None,
    ):
        if e_ops is None:
            e_ops = []
        if c_ops is None:
            c_ops = []
        t_half = args["t_half"]
        tlist = np.linspace(0.0, 2 * t_half, 800)
        H0_r, H_int_b_1, H_int_b_2, H_int_c_1, H_int_c_2 = self.hamiltonian()
        H = [
            H0_r,
            [H_int_b_1, self.gamma_b_func],
            [H_int_b_2, self.gamma_b_func],
            [H_int_c_1, self.gamma_c_func],
            [H_int_c_2, self.gamma_c_func],
        ]
        return mesolve(
            H,
            init_state,
            tlist,
            c_ops=c_ops,
            args=args,
            e_ops=e_ops,
            options=Options(store_final_state=True, store_states=True),
            progress_bar=True,
        )


class SimulateGUETwoWay:
    """compute the fidelity of state transfer for GUEs"""

    def __init__(
            self,
            gamma_a_avg,
            gamma_b_avg,
            gamma_c_avg,
            gamma_a_dev,
            gamma_b_dev,
            gamma_c_dev,
            cavity_dim: int = 2,
    ):
        self.cavity_dim = cavity_dim
        # this is for state transfer one way. think about including two way
        self.truncated_dims = 12 * [cavity_dim]
        [a1_idx, a2_idx, b1_idx, b2_idx, c1_idx, c2_idx] = np.arange(6)
        [a1_r_idx, a2_r_idx, b1_r_idx, b2_r_idx, c1_r_idx, c2_r_idx] = np.arange(6, 12)
        self.phiab = -np.pi / 2
        self.phibc = -np.pi / 2

        self.a1 = id_wrap_ops(destroy(cavity_dim), a1_idx, self.truncated_dims)
        self.a2 = id_wrap_ops(destroy(cavity_dim), a2_idx, self.truncated_dims)
        self.b1 = id_wrap_ops(destroy(cavity_dim), b1_idx, self.truncated_dims)
        self.b2 = id_wrap_ops(destroy(cavity_dim), b2_idx, self.truncated_dims)
        self.c1 = id_wrap_ops(destroy(cavity_dim), c1_idx, self.truncated_dims)
        self.c2 = id_wrap_ops(destroy(cavity_dim), c2_idx, self.truncated_dims)

        self.a1_r = id_wrap_ops(destroy(cavity_dim), a1_r_idx, self.truncated_dims)
        self.a2_r = id_wrap_ops(destroy(cavity_dim), a2_r_idx, self.truncated_dims)
        self.b1_r = id_wrap_ops(destroy(cavity_dim), b1_r_idx, self.truncated_dims)
        self.b2_r = id_wrap_ops(destroy(cavity_dim), b2_r_idx, self.truncated_dims)
        self.c1_r = id_wrap_ops(destroy(cavity_dim), c1_r_idx, self.truncated_dims)
        self.c2_r = id_wrap_ops(destroy(cavity_dim), c2_r_idx, self.truncated_dims)

        self.gamma_a_1 = gamma_a_avg + 0.5 * gamma_a_dev
        self.gamma_a_2 = gamma_a_avg - 0.5 * gamma_a_dev
        self.gamma_b_1 = gamma_b_avg + 0.5 * gamma_b_dev
        self.gamma_b_2 = gamma_b_avg - 0.5 * gamma_b_dev
        self.gamma_c_1 = gamma_c_avg + 0.5 * gamma_c_dev
        self.gamma_c_2 = gamma_c_avg - 0.5 * gamma_c_dev

    def collective_loss_ops(self):
        L_R_a = (
                np.sqrt(self.gamma_a_1) * self.a1_r
                - 1j * np.sqrt(self.gamma_a_2) * self.a2_r
        )
        L_R_b = (np.exp(-1j * self.phiab)
                 * (-1j)
                 * (
                         np.sqrt(self.gamma_b_1) * self.b1_r
                         - 1j * np.sqrt(self.gamma_b_2) * self.b2_r
                 )
                 )
        L_R_c = (
                np.exp(-1j * self.phiab - 1j * self.phibc)
                * (-1j)**2
                * (
                        np.sqrt(self.gamma_c_1) * self.c1_r
                        - 1j * np.sqrt(self.gamma_c_2) * self.c2_r
                )
        )
        L_L_a = (
                np.sqrt(self.gamma_a_1) * self.a1_r
                + 1j * np.sqrt(self.gamma_a_2) * self.a2_r
        )
        L_L_b = (np.exp(1j * self.phiab)
                 * 1j
                 * (
                np.sqrt(self.gamma_b_1) * self.b1_r
                + 1j * np.sqrt(self.gamma_b_2) * self.b2_r
        ))
        L_L_c = (
                np.exp(1j * (self.phiab + self.phibc))
                * 1j ** 2
                * (
                        np.sqrt(self.gamma_c_1) * self.c1_r
                        + 1j * np.sqrt(self.gamma_c_2) * self.c2_r
                )
        )
        return L_R_a, L_R_b, L_R_c, L_L_a, L_L_b, L_L_c

    def construct_c_ops(self, Gamma_1_cav=0.0, Gamma_1_transfer_nr=0.0, nth=0.0):
        L_R_a, L_R_b, L_R_c, L_L_a, L_L_b, L_L_c = self.collective_loss_ops()
        return [
            L_R_a + L_R_b + L_R_c,
            L_L_a + L_L_b + L_L_c,
            np.sqrt(Gamma_1_cav) * self.a1,
            np.sqrt(Gamma_1_cav) * self.a2,
            np.sqrt(Gamma_1_cav) * self.b1,
            np.sqrt(Gamma_1_cav) * self.b2,
            np.sqrt(Gamma_1_cav) * self.c1,
            np.sqrt(Gamma_1_cav) * self.c2,
            np.sqrt(nth * Gamma_1_cav) * self.a1.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.a2.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.b1.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.b2.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.c1.dag(),
            np.sqrt(nth * Gamma_1_cav) * self.c2.dag(),
            np.sqrt(Gamma_1_transfer_nr) * self.a1_r,
            np.sqrt(Gamma_1_transfer_nr) * self.a2_r,
            np.sqrt(Gamma_1_transfer_nr) * self.b1_r,
            np.sqrt(Gamma_1_transfer_nr) * self.b2_r,
            np.sqrt(Gamma_1_transfer_nr) * self.c1_r,
            np.sqrt(Gamma_1_transfer_nr) * self.c2_r,
            np.sqrt(nth * Gamma_1_transfer_nr) * self.a1_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.a2_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.b1_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.b2_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.c1_r.dag(),
            np.sqrt(nth * Gamma_1_transfer_nr) * self.c2_r.dag(),
        ]

    def gamma_b_func(self, t, args=None):
        c = args["c"]
        B = args["B"]
        gamma_b = args["gamma_b_avg"]
        t_half = args["t_half"]
        scale_b = args["scale_b"]
        return (
                scale_b
                * np.sqrt(gamma_b)
                * np.sqrt(
            (
                    0.5
                    * np.exp(-c * (t - t_half) ** 2)
                    / (
                            (1 / B)
                            - np.sqrt(np.pi / (4 * c)) * erf(np.sqrt(c) * (t - t_half))
                    )
            )
        )
        )

    def gamma_c_func(self, t, args=None):
        t_half = args["t_half"]
        scale_c = args["scale_c"]
        return scale_c * self.gamma_b_func(-t + 2 * t_half, args=args)

    def gamma_a_func(self, t, args=None):
        t_half = args["t_half"]
        scale_a = args["scale_a"]
        return scale_a * self.gamma_b_func(-t + 2 * t_half, args=args)

    def hamiltonian(self):
        L_R_a, L_R_b, L_R_c, L_L_a, L_L_b, L_L_c = self.collective_loss_ops()
        H0_r_half = -0.5 * 1j * (L_L_b.dag() * L_L_c + L_R_c.dag() * L_R_b
                                 + L_L_a.dag() * L_L_b + L_R_b.dag() * L_R_a
                                 + L_L_a.dag() * L_L_c + L_R_c.dag() * L_R_a)
        H0_r = H0_r_half + H0_r_half.dag()
        H_int_a_1 = self.a1 * self.a1_r.dag() + self.a1.dag() * self.a1_r
        H_int_a_2 = self.a2 * self.a2_r.dag() + self.a2.dag() * self.a2_r
        H_int_b_1 = self.b1 * self.b1_r.dag() + self.b1.dag() * self.b1_r
        H_int_b_2 = self.b2 * self.b2_r.dag() + self.b2.dag() * self.b2_r
        H_int_c_1 = self.c1 * self.c1_r.dag() + self.c1.dag() * self.c1_r
        H_int_c_2 = self.c2 * self.c2_r.dag() + self.c2.dag() * self.c2_r
        return H0_r, H_int_a_1, H_int_a_2, H_int_b_1, H_int_b_2, H_int_c_1, H_int_c_2

    def run_state_transfer(
            self,
            init_state,
            args,
            c_ops=None,
            e_ops=None,
    ):
        if e_ops is None:
            e_ops = []
        if c_ops is None:
            c_ops = []
        t_half = args["t_half"]
        tlist = np.linspace(0.0, 2 * t_half, 800)
        H0_r, H_int_a_1, H_int_a_2, H_int_b_1, H_int_b_2, H_int_c_1, H_int_c_2 = self.hamiltonian()
        H = [
            H0_r,
            [H_int_a_1, self.gamma_a_func],
            [H_int_a_2, self.gamma_a_func],
            [H_int_b_1, self.gamma_b_func],
            [H_int_b_2, self.gamma_b_func],
            [H_int_c_1, self.gamma_c_func],
            [H_int_c_2, self.gamma_c_func],
        ]
        return mesolve(
            H,
            init_state,
            tlist,
            c_ops=c_ops,
            args=args,
            e_ops=e_ops,
            options=Options(store_final_state=True, store_states=True),
            progress_bar=True,
        )
