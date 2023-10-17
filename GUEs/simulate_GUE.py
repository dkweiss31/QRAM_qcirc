import numpy as np
from qutip import (
    destroy,
    mesolve,
    Options,
    tensor,
    basis,
    Qobj,
)
from scipy.special import erf

from QRAM_utils.dual_rail import DualRailMixin
from QRAM_utils.hashing import Hashing
from QRAM_utils.quantum_helpers import operator_basis_lidar, apply_gate_to_states
from QRAM_utils.utils import id_wrap_ops, construct_basis_states_list


class SimulateGUE:
    """
    performs simulations in https://arxiv.org/abs/2310.08288
    compute the fidelity of state transfer for GUEs
    Parameters
    ----------
    gamma_b_avg, gamma_c_avg, gamma_b_dev, gamma_c_dev: float
        average decay rates of the GUEs into the waveguide and
        their asymmetry: gamma_b1/2 = gamma_b_avg \pm 0.5 * gamma_b_dev, etc.
    cav_idx_dict: dict
        dictionary of ints corresponding to the indices of the data cavities
    tran_res_idx_dict: dict
        dictionary of ints corresponding to the indices of the transfer resonators
    cavity_dim: int
        Hilbert-space dimension used for the cavities and transfer res
    scale_b: float
        multiplying factor on the drive on GUE b (optimized for state transfer)
    scale_c: float
        multiplying factor on the drive on GUE c (optimized for state transfer)
    t_half: float
        2 * t_half = time for state transfer protocol
    xi, zeta: float
        parameters defining the state-transfer pulse
    Gamma_1_cav: float
        T1 of cavities
    Gamma_phi_cav: float
        Tphi of cavities
    Gamma_1_transfer_nr: float
        non-radiative T1 of transfer resonators
    Gamma_phi_transfer: float
        Tphi of transfer resonators
    nth: float
        number of thermal photons in all elements
    nsteps, atol, rtol: int, float, float
        QuTiP solver parameters
    num_cpus: int
        number of cpus used to perform fidelity calculations
    phi=-np.pi/2
        phase associated with the distance between GUEs
    number_degrees_freedom: int
        how many degrees of freedom we are simulating in total
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
        xi: float = 0.006,
        zeta: float = 2.8284e-5,
        Gamma_1_cav: float = 0.0,
        Gamma_phi_cav: float = 0.0,
        Gamma_1_transfer_nr: float = 0.0,
        Gamma_phi_transfer: float = 0.0,
        nth: float = 0.0,
        nsteps: int = 2000,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        num_cpus: int = 8,
        phi=-np.pi/2,  # np.pi/2 for op control, -np.pi / 2 for analytic
        number_degrees_freedom: int = 8,
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
        self.xi = xi
        self.zeta = zeta
        self.Gamma_1_cav = Gamma_1_cav
        self.Gamma_phi_cav = Gamma_phi_cav
        self.Gamma_1_transfer_nr = Gamma_1_transfer_nr
        self.Gamma_phi_transfer = Gamma_phi_transfer
        self.nth = nth
        self.nsteps = nsteps
        self.atol = atol
        self.rtol = rtol
        self.phi = phi
        self.number_degrees_freedom = number_degrees_freedom
        self.truncated_dims = self.number_degrees_freedom * [cavity_dim]
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
        """construct the collective loss operators associated with each GUE"""
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
        """construct all collapse operators to be passed to mesolve"""
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
        """state-transfer pulse applied to GUE b to facilitate state transfer to GUE c"""
        return (
            self.scale_b
            * np.sqrt(self.gamma_b_avg)
            * np.sqrt(
                (
                    0.5
                    * np.exp(-self.zeta * (t - self.t_half) ** 2)
                    / (
                        (1 / self.xi)
                        - np.sqrt(np.pi / (4 * self.zeta))
                        * erf(np.sqrt(self.zeta) * (t - self.t_half))
                    )
                )
            )
        )

    def gamma_c_func(self, t, args=None):
        """state-transfer pulse applied to GUE c to facilitate state transfer from GUE c"""
        return (self.scale_c
                * self.gamma_b_func(-t + 2 * self.t_half, args=args)
                )

    def reduced_rightward_state(self):
        """helper function for performing fidelity calculations: we trace out the
        initial cavities and the transfer resonators, and all we are left with
        is the cavity we are performing state transfer towards"""
        right_state = (
            tensor(basis(2, 1), basis(2, 0))
            + 1j * tensor(basis(2, 0), basis(2, 1))
        ).unit()
        return right_state

    def reduced_zero_state(self):
        """vacuum state associated with e.g. reduced_rightward_state"""
        return tensor(*[basis(dim, 0) for dim in [self.cavity_dim, self.cavity_dim]])

    def vacuum_state(self):
        """vacuum state in the full Hilbert space"""
        (state_0000,) = construct_basis_states_list(
            [self.number_degrees_freedom * (0,), ], self.truncated_dims
        )
        return state_0000

    def hamiltonian(self):
        """the non-Hermitian effective Hamiltonian is H0_r, with drive
        terms H_int_b + H_int_b.dag(), H_int_c + H_int_c.dag()"""
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
        """
        run the state-transfer procedure for a given initial state
        Parameters
        ----------
        init_state: Qobj
            initial state for state transfer
        e_ops: List
            operators to be passed to mesolve for calculating expectation values
        final_state_only: Bool
            True -> return only the final state
            False -> return the mesolve result
        """
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
        """
        calculate the state-transfer fidelity by averaging over
        all possible initial cardinal states
        Parameters
        ----------
        real_final_states: dict
            dictionary of the final states associated with simulating the
            state-transfer protocol
        ideal_final_cardinal_states: dict
            dictionary of the final states of the ideal state-transfer,
            in the same order as real_final_states
        measurement_op: Qobj
            optional, measurement operator used for a projective measurement
            e.g. onto dual-rail basis states to simulate a parity measurement

        Returns
        -------
            state-transfer fidelity, success probability
        """
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

    def overall_state_transfer_fidelity(
            self,
            initial_basis_states: list,
            label_list: list,
            ideal_final_basis_states: list,
            keep_idxs: list
    ):
        """
        given initial basis states, their labels, associated ideal final basis states and indices to keep
        in the partial trace, calculate the state-transfer fidelity and simulated final states
        by first calculating the initial cardinal states, and simulating the state-transfer procedure for each
        Parameters
        ----------
        initial_basis_states: list
            list of Qobjs of initial basis states in the full Hilbert space
        label_list: list
            list of labels associated with the initial states
        ideal_final_basis_states: list
            list of Qobjs of the ideal final states in reduced Hilbert space
        keep_idxs: list
            list of indices to keep in partial transfer after simulating state
            transfer to yield the reduced Hilbert space

        Returns
        -------
        fidelity, final simulated states
        """
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
    """added dual-rail functionality for GUE calculations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def DR_basis(SR_comp_bas_states):
        raise NotImplementedError("shouldn't need to call this function")


class SimulateGUEHashing(Hashing, SimulateGUE):
    """
    Use the Hashing algorithm introduced in https://arxiv.org/abs/1102.4006 to
    simulate a reduced Hilbert space associated with a global-excitation-number cutoff.
    All parameters are as in SimulateGUE except for
    num_exc: int
        global-excitation number cutoff
    """
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
        # below we have to pass cavity_dim=1, a hack
        SimulateGUE.__init__(self, gamma_b_avg, gamma_c_avg, gamma_b_dev, gamma_c_dev,
                             cav_idx_dict, tran_res_idx_dict, cavity_dim=1,
                             number_degrees_freedom=number_degrees_freedom, **kwargs)
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
        """Performing the partial trace is now tricky in this basis. First
        step is to create an instance of Hashing with only 2 degrees of freedom
        which will be the result after tracing out
        """
        return Hashing(number_degrees_freedom=2, num_exc=self.num_exc)

    def reduced_zero_state(self):
        """vacuum state used for fidelity calcs (trace out all irrelevant states). The vacuum state
        is always the first one in the list"""
        red_hilbert_dim = self._reduced_hash().hilbert_dim()
        vac = np.zeros(red_hilbert_dim)
        vac[0] = 1.0
        return Qobj(vac)

    def reduced_rightward_state(self):
        """as above, in the new basis associated with hashing"""
        new_hash = self._reduced_hash()
        red_c1 = new_hash.a_operator(0)
        red_c2 = new_hash.a_operator(1)
        vac = self.reduced_zero_state()
        return ((red_c1.dag() + 1j * red_c2.dag()) * vac).unit()


class SimulateGUEHashingDR(SimulateGUEHashing, DualRailMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimulateGUETwoWay(SimulateGUEHashing):
    """Now we want to include a GUE a to the left of GUE b to better simulate
    the whole process which includes emitting simultaneously to the left and right. This
    simulation includes 12 quantum elements (6 data cavities, 6 transfer resonators)"""

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
        xi: float = 0.006,
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
        self.xi = xi
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
        """collective loss ops for all three GUEs"""
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

    def _reduced_hash(self):
        return Hashing(number_degrees_freedom=4, num_exc=self.num_exc)

    def reduced_rightward_state(self):
        vac = self.reduced_zero_state()
        red_hash = self._reduced_hash()
        (red_c1, red_c2) = (red_hash.a_operator(idx) for idx in range(2, 4))
        return ((red_c1.dag() + 1j * red_c2.dag()) * vac).unit()

    def reduced_leftward_state(self):
        vac = self.reduced_zero_state()
        red_hash = self._reduced_hash()
        (red_a1, red_a2) = (red_hash.a_operator(idx) for idx in range(0, 2))
        return ((red_a1.dag() - 1j * red_a2.dag()) * vac).unit()


class SimulateGUETwoWayDR(SimulateGUETwoWay, DualRailMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
