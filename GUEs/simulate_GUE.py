import warnings

import numpy as np
from qutip import (
    destroy,
    mesolve,
    Options,
    tensor,
    basis,
    Qobj,
)
from scipy.interpolate import CubicSpline
from scipy.special import erf

from QRAM_utils.dual_rail import DualRailMixin
from QRAM_utils.hashing import Hashing
from QRAM_utils.quantum_helpers import operator_basis_lidar, apply_gate_to_states
from QRAM_utils.utils import id_wrap_ops, construct_basis_states_list, extract_controls_QOGS


class SimulateGUE:
    """
    compute the fidelity of state transfer for GUEs
    Parameters
    ----------
    gamma_b_avg, gamma_c_avg, gamma_b_dev: float

    """

    def __init__(
        self,
        gamma_b_avg: float,
        gamma_c_avg: float,
        gamma_b_dev: float,
        gamma_c_dev: float,
        cav_idx_dict: dict,
        tran_res_idx_dict: dict,
        cavity_dim: int = 2,
        scale_b: float = 1.018,
        scale_c: float = 1.017,
        t_half: float = 600.0,
        B: float = 0.006,
        c: float = 2.8284e-5,
        Gamma_1_cav: float = 0.0,
        Gamma_phi_cav: float = 0.0,
        Gamma_1_transfer_nr: float = 0.0,
        Gamma_phi_transfer: float = 0.0,
        nth: float = 0.0,
        nsteps: int = 2000,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        num_cpus: int = 8,
        phi=-np.pi/2,
    ):
        self.gamma_b_1 = gamma_b_avg + 0.5 * gamma_b_dev
        self.gamma_b_2 = gamma_b_avg - 0.5 * gamma_b_dev
        self.gamma_c_1 = gamma_c_avg + 0.5 * gamma_c_dev
        self.gamma_c_2 = gamma_c_avg - 0.5 * gamma_c_dev
        self.gamma_b_avg = gamma_b_avg
        self.gamma_c_avg = gamma_c_avg
        self.gamma_b_dev = gamma_b_dev
        self.gamma_c_dev = gamma_c_dev
        self.cav_idx_dict = cav_idx_dict
        self.tran_res_idx_dict = tran_res_idx_dict
        for label, idx in cav_idx_dict.items():
            setattr(self, label, idx)
        for label, idx in tran_res_idx_dict.items():
            setattr(self, label, idx)
        self.cavity_dim = cavity_dim
        self.scale_b = scale_b
        self.scale_c = scale_c
        self.t_half = t_half
        self.B = B
        self.c = c
        self.Gamma_1_cav = Gamma_1_cav
        self.Gamma_phi_cav = Gamma_phi_cav
        self.Gamma_1_transfer_nr = Gamma_1_transfer_nr
        self.Gamma_phi_transfer = Gamma_phi_transfer
        self.nth = nth
        self.nsteps = nsteps
        self.atol = atol
        self.rtol = rtol
        self.truncated_dims = 8 * [cavity_dim]
        self.phi = phi
        self.num_cpus = num_cpus
        self.options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )

        self.b1 = id_wrap_ops(destroy(self.cavity_dim), cav_idx_dict["b1_idx"], self.truncated_dims)
        self.b2 = id_wrap_ops(destroy(self.cavity_dim), cav_idx_dict["b2_idx"], self.truncated_dims)
        self.c1 = id_wrap_ops(destroy(self.cavity_dim), cav_idx_dict["c1_idx"], self.truncated_dims)
        self.c2 = id_wrap_ops(destroy(self.cavity_dim), cav_idx_dict["c2_idx"], self.truncated_dims)

        self.b1_r = id_wrap_ops(destroy(self.cavity_dim), tran_res_idx_dict["b1_r_idx"], self.truncated_dims)
        self.b2_r = id_wrap_ops(destroy(self.cavity_dim), tran_res_idx_dict["b2_r_idx"], self.truncated_dims)
        self.c1_r = id_wrap_ops(destroy(self.cavity_dim), tran_res_idx_dict["c1_r_idx"], self.truncated_dims)
        self.c2_r = id_wrap_ops(destroy(self.cavity_dim), tran_res_idx_dict["c2_r_idx"], self.truncated_dims)

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

    def construct_c_ops(self):
        L_R_b, L_R_c, L_L_b, L_L_c = self.collective_loss_ops()
        return [
            L_R_b + L_R_c,
            L_L_b + L_L_c,
            np.sqrt(self.Gamma_1_cav) * self.b1,
            np.sqrt(self.Gamma_1_cav) * self.b2,
            np.sqrt(self.Gamma_1_cav) * self.c1,
            np.sqrt(self.Gamma_1_cav) * self.c2,
            np.sqrt(self.nth * self.Gamma_1_cav) * self.b1.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.b2.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.c1.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.c2.dag(),
            np.sqrt(self.Gamma_phi_cav) * self.b1.dag() * self.b1,
            np.sqrt(self.Gamma_phi_cav) * self.b2.dag() * self.b2,
            np.sqrt(self.Gamma_phi_cav) * self.c1.dag() * self.c1,
            np.sqrt(self.Gamma_phi_cav) * self.c2.dag() * self.c2,
            np.sqrt(self.Gamma_1_transfer_nr) * self.b1_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.b2_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.c1_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.c2_r,
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.b1_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.b2_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.c1_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.c2_r.dag(),
            np.sqrt(self.Gamma_phi_transfer) * self.b1_r.dag() * self.b1_r,
            np.sqrt(self.Gamma_phi_transfer) * self.b2_r.dag() * self.b2_r,
            np.sqrt(self.Gamma_phi_transfer) * self.c1_r.dag() * self.c1_r,
            np.sqrt(self.Gamma_phi_transfer) * self.c2_r.dag() * self.c2_r,
        ]

    def gamma_b_func(self, t, args=None):
        return (
            self.scale_b
            * np.sqrt(self.gamma_b_avg)
            * np.sqrt(
                (
                    0.5
                    * np.exp(-self.c * (t - self.t_half) ** 2)
                    / (
                        (1 / self.B)
                        - np.sqrt(np.pi / (4 * self.c))
                        * erf(np.sqrt(self.c) * (t - self.t_half))
                    )
                )
            )
        )

    def gamma_c_func(self, t, args=None):
        return (self.scale_c
                * self.gamma_b_func(-t + 2 * self.t_half, args=args)
                )

    def reduced_rightward_state(self):
        right_state = (
            tensor(basis(2, 1), basis(2, 0))
            + 1j * tensor(basis(2, 0), basis(2, 1))
        ).unit()
        return right_state

    def reduced_zero_state(self):
        return tensor(*[basis(dim, 0) for dim in [self.cavity_dim, self.cavity_dim]])

    def vacuum_state(self):
        (state_0000,) = construct_basis_states_list([8 * (0,), ], self.truncated_dims)
        return state_0000

    def hamiltonian(self):
        L_R_b, L_R_c, L_L_b, L_L_c = self.collective_loss_ops()
        H0_r_half = -0.5 * 1j * (L_R_c.dag() * L_R_b + L_L_b.dag() * L_L_c)
        H0_r = H0_r_half + H0_r_half.dag()
        H_int_b = self.b1 * self.b1_r.dag() + self.b2 * self.b2_r.dag()
        H_int_c = self.c1 * self.c1_r.dag() + self.c2 * self.c2_r.dag()
        return H0_r, H_int_b, H_int_c

    def _setup_H_for_mesolve(self):
        tlist = np.linspace(0.0, 2 * self.t_half, 800)
        H0_r, H_int_b, H_int_c = self.hamiltonian()
        H = [
            H0_r,
            [H_int_b + H_int_b.dag(), self.gamma_b_func],
            [H_int_c + H_int_c.dag(), self.gamma_c_func],
        ]
        return tlist, H

    def run_state_transfer(
        self,
        init_state,
        e_ops=None,
        final_state_only=True,
    ) -> Qobj:
        if e_ops is None:
            e_ops = []
        tlist, H = self._setup_H_for_mesolve()
        result = mesolve(
            H,
            init_state,
            tlist,
            c_ops=self.construct_c_ops(),
            e_ops=e_ops,
            options=self.options,
        )
        if final_state_only:
            return result.final_state
        else:
            return result

    @staticmethod
    def state_transfer_fidelity(
        real_final_states: dict,
        ideal_final_cardinal_states: dict,
        measurement_op: Qobj = None,
    ):
        fidel = 0.0
        total_prob = 0.0
        num_states = len(ideal_final_cardinal_states)
        for (real_final_state, ideal_final_state) in zip(
            real_final_states.values(), ideal_final_cardinal_states.values()
        ):
            norm = np.trace(real_final_state)
            real_final_state = real_final_state / norm
            if measurement_op is not None:
                real_final_state = (
                    measurement_op * real_final_state * measurement_op.dag()
                )
                prob = np.trace(real_final_state)
                real_final_state = real_final_state / prob
                total_prob += prob
            fidel += np.trace(real_final_state * ideal_final_state)
        return fidel / num_states, total_prob / num_states

    @staticmethod
    def trace_out_dict(state_dict, keep_idxs):
        return {
            label: final_state.ptrace(keep_idxs)
            for label, final_state in state_dict.items()
        }

    def overall_state_transfer_fidelity(self, initial_basis_states, label_list, ideal_final_basis_states, keep_idxs):
        op_dict_SR, initial_cardinal_states = operator_basis_lidar(
            initial_basis_states, label_list=label_list
        )
        _, ideal_final_cardinal_states = operator_basis_lidar(
            ideal_final_basis_states, label_list=label_list
        )
        final_SR_states = apply_gate_to_states(
            self.run_state_transfer, initial_cardinal_states, self.num_cpus
        )
        final_SR_states = self.trace_out_dict(final_SR_states, keep_idxs)
        fidel_SR, _ = self.state_transfer_fidelity(
            final_SR_states, ideal_final_cardinal_states
        )
        return fidel_SR, final_SR_states


class SimulateGUEDR(SimulateGUE, DualRailMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def DR_basis(SR_comp_bas_states):
        raise NotImplementedError("shouldn't need to call this function")


class SimulateGUEHashing(Hashing, SimulateGUE):
    def __init__(
            self,
            gamma_b_avg: float,
            gamma_c_avg: float,
            gamma_b_dev: float,
            gamma_c_dev: float,
            cav_idx_dict: dict,
            tran_res_idx_dict: dict,
            num_exc: int = 1,
            number_degrees_freedom: int = 8,
            **kwargs,
    ):
        Hashing.__init__(self, num_exc=num_exc, number_degrees_freedom=number_degrees_freedom)
        SimulateGUE.__init__(self, gamma_b_avg, gamma_c_avg, gamma_b_dev, gamma_c_dev,
                             cav_idx_dict, tran_res_idx_dict, cavity_dim=1, **kwargs)
        self.b1 = self.a_operator(cav_idx_dict["b1_idx"])
        self.b2 = self.a_operator(cav_idx_dict["b2_idx"])
        self.c1 = self.a_operator(cav_idx_dict["c1_idx"])
        self.c2 = self.a_operator(cav_idx_dict["c2_idx"])
        self.b1_r = self.a_operator(tran_res_idx_dict["b1_r_idx"])
        self.b2_r = self.a_operator(tran_res_idx_dict["b2_r_idx"])
        self.c1_r = self.a_operator(tran_res_idx_dict["c1_r_idx"])
        self.c2_r = self.a_operator(tran_res_idx_dict["c2_r_idx"])

    def vacuum_state(self):
        """vacuum state of the full system"""
        vac = np.zeros(self.hilbert_dim())
        vac[0] = 1.0
        return Qobj(vac)

    def _reduced_hash(self):
        return Hashing(number_degrees_freedom=2, num_exc=self.num_exc)

    def reduced_zero_state(self):
        """vacuum state used for fidelity calcs (trace out all irrelevant states)"""
        red_hilbert_dim = self._reduced_hash().hilbert_dim()
        vac = np.zeros(red_hilbert_dim)
        vac[0] = 1.0
        return Qobj(vac)

    def reduced_rightward_state(self):
        new_hash = self._reduced_hash()
        red_c1 = new_hash.a_operator(0)
        red_c2 = new_hash.a_operator(1)
        vac = self.reduced_zero_state()
        return ((red_c1.dag() + 1j * red_c2.dag()) * vac).unit()


class SimulateGUEHashingOptControl(SimulateGUEHashing):
    def __init__(self, gamma_b_avg: float, gamma_c_avg: float, gamma_b_dev: float, gamma_c_dev: float,
                 cav_idx_dict: dict, tran_res_idx_dict: dict, control_file_location: str, **kwargs):
        super().__init__(gamma_b_avg, gamma_c_avg, gamma_b_dev, gamma_c_dev,
                         cav_idx_dict, tran_res_idx_dict, **kwargs)
        self.control_file_location = control_file_location

    def _setup_H_for_mesolve(self):
        tlist, controls = extract_controls_QOGS(self.control_file_location)
        if tlist[-1] != 2 * self.t_half:
            warnings.WarningMessage("Supplied pulse time does not match that extracted from"
                                    "optimal control. Proceeding with optimal control pulse time.")
        H0_r, H_int_b, H_int_c = self.hamiltonian()
        H = [
            H0_r,
            # controls[0] and controls[2] contain the I quadrature, don't need Q
            [H_int_b + H_int_b.dag(), controls[0]],
            [H_int_c + H_int_c.dag(), controls[2]],
        ]
        return tlist, H

    def gamma_b_func(self, t, args=None):
        tlist, controls = extract_controls_QOGS(self.control_file_location)
        return CubicSpline(tlist, controls[0], bc_type="natural")

    def gamma_c_func(self, t, args=None):
        tlist, controls = extract_controls_QOGS(self.control_file_location)
        return CubicSpline(tlist, controls[2], bc_type="natural")


class SimulateGUEHashingDR(SimulateGUEHashing, DualRailMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimulateGUETwoWay(SimulateGUEHashing):
    """compute the fidelity of state transfer for GUEs"""

    def __init__(
        self,
        gamma_a_avg: float,
        gamma_b_avg: float,
        gamma_c_avg: float,
        gamma_a_dev: float,
        gamma_b_dev: float,
        gamma_c_dev: float,
        cav_idx_dict: dict,
        tran_res_idx_dict: dict,
        scale_a: float = 1.017,
        scale_b: float = 1.018,
        scale_c: float = 1.017,
        t_half: float = 600.0,
        B: float = 0.006,
        c: float = 2.8284e-5,
        Gamma_1_cav: float = 0.0,
        Gamma_phi_cav: float = 0.0,
        Gamma_1_transfer_nr: float = 0.0,
        Gamma_phi_transfer: float = 0.0,
        nth: float = 0.0,
        nsteps: int = 2000,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        num_cpus: int = 8,
        num_exc: int = 1,
    ):
        Hashing.__init__(self, num_exc=num_exc, number_degrees_freedom=12)
        self.gamma_b_1 = gamma_b_avg + 0.5 * gamma_b_dev
        self.gamma_b_2 = gamma_b_avg - 0.5 * gamma_b_dev
        self.gamma_c_1 = gamma_c_avg + 0.5 * gamma_c_dev
        self.gamma_c_2 = gamma_c_avg - 0.5 * gamma_c_dev
        self.gamma_b_avg = gamma_b_avg
        self.gamma_c_avg = gamma_c_avg
        self.gamma_b_dev = gamma_b_dev
        self.gamma_c_dev = gamma_c_dev
        self.cav_idx_dict = cav_idx_dict
        self.tran_res_idx_dict = tran_res_idx_dict
        for label, idx in cav_idx_dict.items():
            setattr(self, label, idx)
        for label, idx in tran_res_idx_dict.items():
            setattr(self, label, idx)
        self.scale_b = scale_b
        self.scale_c = scale_c
        self.t_half = t_half
        self.B = B
        self.c = c
        self.Gamma_1_cav = Gamma_1_cav
        self.Gamma_phi_cav = Gamma_phi_cav
        self.Gamma_1_transfer_nr = Gamma_1_transfer_nr
        self.Gamma_phi_transfer = Gamma_phi_transfer
        self.nth = nth
        self.nsteps = nsteps
        self.atol = atol
        self.rtol = rtol
        self.options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )
        # new items below
        self.gamma_a_1 = gamma_a_avg + 0.5 * gamma_a_dev
        self.gamma_a_2 = gamma_a_avg - 0.5 * gamma_a_dev
        self.gamma_a_avg = gamma_a_avg
        self.gamma_a_dev = gamma_a_dev
        self.scale_a = scale_a
        self.phiab = -np.pi/2
        self.phibc = -np.pi/2
        self.num_exc = num_exc
        self.num_cpus = num_cpus
        self.a1 = self.a_operator(cav_idx_dict["a1_idx"])
        self.a2 = self.a_operator(cav_idx_dict["a2_idx"])
        self.b1 = self.a_operator(cav_idx_dict["b1_idx"])
        self.b2 = self.a_operator(cav_idx_dict["b2_idx"])
        self.c1 = self.a_operator(cav_idx_dict["c1_idx"])
        self.c2 = self.a_operator(cav_idx_dict["c2_idx"])
        self.a1_r = self.a_operator(tran_res_idx_dict["a1_r_idx"])
        self.a2_r = self.a_operator(tran_res_idx_dict["a2_r_idx"])
        self.b1_r = self.a_operator(tran_res_idx_dict["b1_r_idx"])
        self.b2_r = self.a_operator(tran_res_idx_dict["b2_r_idx"])
        self.c1_r = self.a_operator(tran_res_idx_dict["c1_r_idx"])
        self.c2_r = self.a_operator(tran_res_idx_dict["c2_r_idx"])

    def collective_loss_ops(self):
        L_R_a = (
            np.sqrt(self.gamma_a_1) * self.a1_r
            - 1j * np.sqrt(self.gamma_a_2) * self.a2_r
        )
        L_R_b = (
            np.exp(-1j * self.phiab)
            * (-1j)
            * (
                np.sqrt(self.gamma_b_1) * self.b1_r
                - 1j * np.sqrt(self.gamma_b_2) * self.b2_r
            )
        )
        L_R_c = (
            np.exp(-1j * self.phiab - 1j * self.phibc)
            * (-1j) ** 2
            * (
                np.sqrt(self.gamma_c_1) * self.c1_r
                - 1j * np.sqrt(self.gamma_c_2) * self.c2_r
            )
        )
        L_L_a = (
            np.sqrt(self.gamma_a_1) * self.a1_r
            + 1j * np.sqrt(self.gamma_a_2) * self.a2_r
        )
        L_L_b = (
            np.exp(1j * self.phiab)
            * 1j
            * (
                np.sqrt(self.gamma_b_1) * self.b1_r
                + 1j * np.sqrt(self.gamma_b_2) * self.b2_r
            )
        )
        L_L_c = (
            np.exp(1j * (self.phiab + self.phibc))
            * 1j**2
            * (
                np.sqrt(self.gamma_c_1) * self.c1_r
                + 1j * np.sqrt(self.gamma_c_2) * self.c2_r
            )
        )
        return L_R_a, L_R_b, L_R_c, L_L_a, L_L_b, L_L_c

    def construct_c_ops(self):
        L_R_a, L_R_b, L_R_c, L_L_a, L_L_b, L_L_c = self.collective_loss_ops()
        return [
            L_R_a + L_R_b + L_R_c,
            L_L_a + L_L_b + L_L_c,
            np.sqrt(self.Gamma_1_cav) * self.a1,
            np.sqrt(self.Gamma_1_cav) * self.a2,
            np.sqrt(self.Gamma_1_cav) * self.b1,
            np.sqrt(self.Gamma_1_cav) * self.b2,
            np.sqrt(self.Gamma_1_cav) * self.c1,
            np.sqrt(self.Gamma_1_cav) * self.c2,
            np.sqrt(self.nth * self.Gamma_1_cav) * self.a1.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.a2.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.b1.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.b2.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.c1.dag(),
            np.sqrt(self.nth * self.Gamma_1_cav) * self.c2.dag(),
            np.sqrt(self.Gamma_phi_cav) * self.a1.dag() * self.a1,
            np.sqrt(self.Gamma_phi_cav) * self.a2.dag() * self.a2,
            np.sqrt(self.Gamma_phi_cav) * self.b1.dag() * self.b1,
            np.sqrt(self.Gamma_phi_cav) * self.b2.dag() * self.b2,
            np.sqrt(self.Gamma_phi_cav) * self.c1.dag() * self.c1,
            np.sqrt(self.Gamma_phi_cav) * self.c2.dag() * self.c2,
            np.sqrt(self.Gamma_1_transfer_nr) * self.a1_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.a2_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.b1_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.b2_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.c1_r,
            np.sqrt(self.Gamma_1_transfer_nr) * self.c2_r,
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.a1_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.a2_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.b1_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.b2_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.c1_r.dag(),
            np.sqrt(self.nth * self.Gamma_1_transfer_nr) * self.c2_r.dag(),
            np.sqrt(self.Gamma_phi_transfer) * self.a1_r.dag() * self.a1_r,
            np.sqrt(self.Gamma_phi_transfer) * self.a2_r.dag() * self.a2_r,
            np.sqrt(self.Gamma_phi_transfer) * self.b1_r.dag() * self.b1_r,
            np.sqrt(self.Gamma_phi_transfer) * self.b2_r.dag() * self.b2_r,
            np.sqrt(self.Gamma_phi_transfer) * self.c1_r.dag() * self.c1_r,
            np.sqrt(self.Gamma_phi_transfer) * self.c2_r.dag() * self.c2_r,
        ]

    def gamma_a_func(self, t, args=None):
        return self.scale_a * self.gamma_b_func(-t + 2 * self.t_half, args=args)

    def hamiltonian(self):
        L_R_a, L_R_b, L_R_c, L_L_a, L_L_b, L_L_c = self.collective_loss_ops()
        H0_r_half = (
            -0.5
            * 1j
            * (
                L_L_b.dag() * L_L_c
                + L_R_c.dag() * L_R_b
                + L_L_a.dag() * L_L_b
                + L_R_b.dag() * L_R_a
                + L_L_a.dag() * L_L_c
                + L_R_c.dag() * L_R_a
            )
        )
        H0_r = H0_r_half + H0_r_half.dag()
        H_int_a = self.a1 * self.a1_r.dag() + self.a2 * self.a2_r.dag()
        H_int_b = self.b1 * self.b1_r.dag() + self.b2 * self.b2_r.dag()
        H_int_c = self.c1 * self.c1_r.dag() + self.c2 * self.c2_r.dag()
        return H0_r, H_int_a, H_int_b, H_int_c,

    def _setup_H_for_mesolve(self):
        tlist = np.linspace(0.0, 2 * self.t_half, 800)
        (
            H0_r,
            H_int_a,
            H_int_b,
            H_int_c,
        ) = self.hamiltonian()
        H = [
            H0_r,
            [H_int_a, self.gamma_a_func],
            [H_int_a.dag(), lambda t, a: np.conj(self.gamma_a_func(t, a))],
            [H_int_b, self.gamma_b_func],
            [H_int_b.dag(), self.gamma_b_func],
            [H_int_c, self.gamma_c_func],
            [H_int_c.dag(), self.gamma_c_func],
        ]
        return tlist, H


class SimulateGUETwoWayOptControl(SimulateGUETwoWay, SimulateGUEHashingOptControl):
    def __init__(self, gamma_a_avg: float, gamma_b_avg: float, gamma_c_avg: float, gamma_a_dev: float,
                 gamma_b_dev: float, gamma_c_dev: float, cav_idx_dict: dict, tran_res_idx_dict: dict,
                 control_file_location: str, **kwargs):
        SimulateGUETwoWay.__init__(self, gamma_a_avg, gamma_b_avg, gamma_c_avg, gamma_a_dev, gamma_b_dev, gamma_c_dev,
                                   cav_idx_dict, tran_res_idx_dict, **kwargs)
        self.control_file_location = control_file_location

    def _setup_H_for_mesolve(self):
        tlist, controls = extract_controls_QOGS(self.control_file_location)
        if tlist[-1] != 2 * self.t_half:
            warnings.WarningMessage("Supplied pulse time does not match that extracted from"
                                    "optimal control. Proceeding with optimal control pulse time.")
        H0_r, H_int_a, H_int_b, H_int_c = self.hamiltonian()
        H = [
            H0_r,
            # controls[0] and controls[2] contain the I quadrature, don't need Q
            [H_int_a + H_int_a.dag(), controls[2]],
            [H_int_b + H_int_b.dag(), controls[0]],
            [H_int_c + H_int_c.dag(), controls[2]],
        ]
        return tlist, H

    def gamma_a_func(self, t, args=None):
        tlist, controls = extract_controls_QOGS(self.control_file_location)
        return CubicSpline(tlist, controls[2], bc_type="natural")


class SimulateGUETwoWayDR(SimulateGUETwoWay, DualRailMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
