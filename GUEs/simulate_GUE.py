import copy

import numpy as np
from qutip import (
    destroy,
    mesolve,
    Options,
    tensor,
    qeye,
    basis,
    Qobj,
)
from scipy.special import erf

from utils.dual_rail import DualRailMixin
from utils.utils import id_wrap_ops


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
        additional_label: bool = False,
        nsteps: int = 2000,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        num_cpus: int = 8,  # only included to allow it to be passed to this class
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
        self.additional_label = additional_label
        self.nsteps = nsteps
        self.atol = atol
        self.rtol = rtol
        self.truncated_dims = 8 * [cavity_dim]
        self.phi = -np.pi / 2

        if not additional_label:
            for label, idx in cav_idx_dict.items():
                setattr(
                    self,
                    str(label[0:2]),
                    id_wrap_ops(destroy(cavity_dim), idx, self.truncated_dims),
                )
            for label, idx in tran_res_idx_dict.items():
                setattr(
                    self,
                    str(label[0:4]),
                    id_wrap_ops(destroy(cavity_dim), idx, self.truncated_dims),
                )
        else:
            for label, idx in cav_idx_dict.items():
                setattr(
                    self,
                    str(label[0:2]),
                    tensor(
                        id_wrap_ops(destroy(cavity_dim), idx, self.truncated_dims),
                        qeye(2),
                    ),
                )
            for label, idx in tran_res_idx_dict.items():
                setattr(
                    self,
                    str(label[0:4]),
                    tensor(
                        id_wrap_ops(destroy(cavity_dim), idx, self.truncated_dims),
                        qeye(2),
                    ),
                )

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
        return self.scale_c * self.gamma_b_func(-t + 2 * self.t_half, args=args)

    def rightward_state(self, idx_0, idx_1):
        assert idx_1 == idx_0 + 1
        dim_0 = self.truncated_dims[idx_0]
        dim_1 = self.truncated_dims[idx_1]
        right_state = (
            tensor(basis(dim_0, 1), basis(dim_1, 0))
            + 1j * tensor(basis(dim_0, 0), basis(dim_1, 1))
        ).unit()
        return right_state

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
        e_ops=None,
        final_state_only=True,
    ) -> Qobj:
        if e_ops is None:
            e_ops = []
        tlist = np.linspace(0.0, 2 * self.t_half, 800)
        H0_r, H_int_b_1, H_int_b_2, H_int_c_1, H_int_c_2 = self.hamiltonian()
        H = [
            H0_r,
            [H_int_b_1, self.gamma_b_func],
            [H_int_b_2, self.gamma_b_func],
            [H_int_c_1, self.gamma_c_func],
            [H_int_c_2, self.gamma_c_func],
        ]
        options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )
        result = mesolve(
            H,
            init_state,
            tlist,
            c_ops=self.construct_c_ops(),
            e_ops=e_ops,
            options=options,
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


class SimulateGUEDR(SimulateGUE, DualRailMixin):
    def __init__(
        self,
        gamma_b_avg,
        gamma_c_avg,
        gamma_b_dev,
        gamma_c_dev,
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
        additional_label: bool = False,
        nsteps: int = 2000,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        num_cpus: int = 8,
    ):

        super().__init__(
            gamma_b_avg,
            gamma_c_avg,
            gamma_b_dev,
            gamma_c_dev,
            cav_idx_dict,
            tran_res_idx_dict,
            cavity_dim=cavity_dim,
            scale_b=scale_b,
            scale_c=scale_c,
            t_half=t_half,
            B=B,
            c=c,
            Gamma_1_cav=Gamma_1_cav,
            Gamma_phi_cav=Gamma_phi_cav,
            Gamma_1_transfer_nr=Gamma_1_transfer_nr,
            Gamma_phi_transfer=Gamma_phi_transfer,
            nth=nth,
            additional_label=additional_label,
            nsteps=nsteps,
            atol=atol,
            rtol=rtol,
            num_cpus=num_cpus,
        )

    def measurement_op_DR(self, idx_0, idx_1):
        # assume below that we've traced out transfer res and initial data cavs
        assert idx_1 == idx_0 + 1
        dim_0 = self.truncated_dims[idx_0]
        dim_1 = self.truncated_dims[idx_1]
        right_state = self.rightward_state(idx_0, idx_1)
        right_state_proj = right_state * right_state.dag()
        left_state_proj = copy.deepcopy(right_state_proj)
        if self.additional_label:
            right_state_proj = tensor(right_state_proj, basis(2, 0) * basis(2, 0).dag())
            left_state_proj = tensor(left_state_proj, basis(2, 1) * basis(2, 1).dag())
            vac_proj = tensor(
                *[basis(dim, 0) * basis(dim, 0).dag() for dim in [dim_0, dim_1, 2]]
            )
            return (
                tensor(right_state_proj, vac_proj)
                + tensor(vac_proj, right_state_proj)
                + tensor(left_state_proj, vac_proj)
                + tensor(vac_proj, left_state_proj)
            )
        else:
            vac_proj = tensor(
                *[basis(dim, 0) * basis(dim, 0).dag() for dim in [dim_0, dim_1]]
            )
            return tensor(right_state_proj, vac_proj) + tensor(
                vac_proj, right_state_proj
            )

    def DR_basis(self, SR_comp_bas_states):
        if self.additional_label:
            # assumption is that SR_comp_bas_states = [|0>, |1>_{R}, |1>_{L}] in terms of logical states
            return [
                tensor(SR_comp_bas_states[1], SR_comp_bas_states[0]),
                tensor(SR_comp_bas_states[2], SR_comp_bas_states[0]),
                tensor(SR_comp_bas_states[0], SR_comp_bas_states[1]),
                tensor(SR_comp_bas_states[0], SR_comp_bas_states[2]),
            ]
        else:
            return [
                tensor(SR_comp_bas_states[1], SR_comp_bas_states[0]),
                tensor(SR_comp_bas_states[0], SR_comp_bas_states[1]),
            ]

    def _DR_op_from_SR_ops(self, DR_label_1, DR_label_2, final_SR_ops):
        """construct DR op given DR labels and dictionary of SR operators.
        see overrided function for more info
        """
        if self.additional_label:
            # in this case, DR labels are of the form e.g. "0R1L" so of the same form
            # as the bosonic case
            return super()._DR_op_from_SR_ops(DR_label_1, DR_label_2, final_SR_ops)
        else:
            # DR labels are now simply "10" or "01"
            SR_label_1 = DR_label_1[0] + DR_label_2[0]
            SR_label_2 = DR_label_1[1] + DR_label_2[1]
            return tensor(final_SR_ops[SR_label_1], final_SR_ops[SR_label_2])


class SimulateGUETwoWay(SimulateGUE):
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
        cavity_dim: int = 2,
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
        # new items below
        self.gamma_a_1 = gamma_a_avg + 0.5 * gamma_a_dev
        self.gamma_a_2 = gamma_a_avg - 0.5 * gamma_a_dev
        self.gamma_a_avg = gamma_a_avg
        self.gamma_a_dev = gamma_a_dev
        self.scale_a = scale_a
        self.truncated_dims = 12 * [cavity_dim]
        self.phiab = -np.pi / 2
        self.phibc = -np.pi / 2

        self.a1 = id_wrap_ops(
            destroy(cavity_dim), cav_idx_dict["a1_idx"], self.truncated_dims
        )
        self.a2 = id_wrap_ops(
            destroy(cavity_dim), cav_idx_dict["a2_idx"], self.truncated_dims
        )
        self.b1 = id_wrap_ops(
            destroy(cavity_dim), cav_idx_dict["b1_idx"], self.truncated_dims
        )
        self.b2 = id_wrap_ops(
            destroy(cavity_dim), cav_idx_dict["b2_idx"], self.truncated_dims
        )
        self.c1 = id_wrap_ops(
            destroy(cavity_dim), cav_idx_dict["c1_idx"], self.truncated_dims
        )
        self.c2 = id_wrap_ops(
            destroy(cavity_dim), cav_idx_dict["c2_idx"], self.truncated_dims
        )

        self.a1_r = id_wrap_ops(
            destroy(cavity_dim), tran_res_idx_dict["a1_r_idx"], self.truncated_dims
        )
        self.a2_r = id_wrap_ops(
            destroy(cavity_dim), tran_res_idx_dict["a2_r_idx"], self.truncated_dims
        )
        self.b1_r = id_wrap_ops(
            destroy(cavity_dim), tran_res_idx_dict["b1_r_idx"], self.truncated_dims
        )
        self.b2_r = id_wrap_ops(
            destroy(cavity_dim), tran_res_idx_dict["b2_r_idx"], self.truncated_dims
        )
        self.c1_r = id_wrap_ops(
            destroy(cavity_dim), tran_res_idx_dict["c1_r_idx"], self.truncated_dims
        )
        self.c2_r = id_wrap_ops(
            destroy(cavity_dim), tran_res_idx_dict["c2_r_idx"], self.truncated_dims
        )

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
        e_ops=None,
        final_state_only=True,
    ) -> Qobj:
        if e_ops is None:
            e_ops = []
        tlist = np.linspace(0.0, 2 * self.t_half, 800)
        (
            H0_r,
            H_int_a_1,
            H_int_a_2,
            H_int_b_1,
            H_int_b_2,
            H_int_c_1,
            H_int_c_2,
        ) = self.hamiltonian()
        H = [
            H0_r,
            [H_int_a_1, self.gamma_a_func],
            [H_int_a_2, self.gamma_a_func],
            [H_int_b_1, self.gamma_b_func],
            [H_int_b_2, self.gamma_b_func],
            [H_int_c_1, self.gamma_c_func],
            [H_int_c_2, self.gamma_c_func],
        ]
        options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )
        result = mesolve(
            H,
            init_state,
            tlist,
            c_ops=self.construct_c_ops(),
            e_ops=e_ops,
            options=options,
        )
        if final_state_only:
            return result.final_state
        else:
            return result


class SimulateGUETwoWayDR(SimulateGUETwoWay, DualRailMixin):
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
        cavity_dim: int = 2,
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
    ):
        super().__init__(
            gamma_a_avg,
            gamma_b_avg,
            gamma_c_avg,
            gamma_a_dev,
            gamma_b_dev,
            gamma_c_dev,
            cav_idx_dict,
            tran_res_idx_dict,
            cavity_dim=cavity_dim,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_c=scale_c,
            t_half=t_half,
            B=B,
            c=c,
            Gamma_1_cav=Gamma_1_cav,
            Gamma_phi_cav=Gamma_phi_cav,
            Gamma_1_transfer_nr=Gamma_1_transfer_nr,
            Gamma_phi_transfer=Gamma_phi_transfer,
            nth=nth,
            nsteps=nsteps,
            atol=atol,
            rtol=rtol,
            num_cpus=num_cpus,
        )

    def _DR_op_from_SR_ops(self, DR_label_1, DR_label_2, final_SR_ops):
        """construct DR op given DR labels and dictionary of SR operators.
        see overrided function for more info
        """
        # DR labels are assumed to be "R0", "L0", "0R", "0L"
        SR_label_1 = DR_label_1[0] + DR_label_2[0]
        SR_label_2 = DR_label_1[1] + DR_label_2[1]
        return tensor(final_SR_ops[SR_label_1], final_SR_ops[SR_label_2])
