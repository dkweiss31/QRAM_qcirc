from typing import List

import numpy as np
from qutip import (
    basis,
    Qobj,
    to_super,
    Options,
)

from QRAM_utils.dual_rail import DualRailMixin
from QRAM_utils.quantum_helpers import prop_or_mesolve_factory, SWAP_op
from QRAM_utils.utils import id_wrap_ops, construct_basis_states_list


class SimulateBosonicOperations:
    """
    Simulate time evolution of a system of two cavities with one coupled to a transmon
    Parameters
    ----------
    gf_tmon: bool
        True: we use a gf tmon
        False: we use a ge tmon
    tmon_dim: int
        transmon Hilbert space dimension
    cavity_dim: int
        cavity Hilbert space dimensions
    nsteps, atol, rtol: int, float, float
        options for mesolve
    """

    def __init__(
        self, gf_tmon=True, tmon_dim=3, cavity_dim=3, nsteps=2000, atol=1e-8, rtol=1e-6
    ):
        self.gf_tmon = gf_tmon
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim
        self.truncated_dims = [cavity_dim, cavity_dim, tmon_dim]
        self.nsteps = nsteps
        self.atol = atol
        self.rtol = rtol
        # below we define the s_ops for the transmon
        self.sx = None
        self.sy = None
        self.sz = None
        self.s_gg = None
        self.s_ee = None
        self.s_ff = None
        self.sminus_ge = None
        self.sminus_ef = None
        if self.gf_tmon:
            self.s_ops_gi_tmon(2)
        else:
            self.s_ops_gi_tmon(1)

    def s_ops_gi_tmon(self, i):
        """construct gi tmon where i can be e or f"""
        sx_gi = (
            basis(self.tmon_dim, 0) * basis(self.tmon_dim, i).dag()
            + basis(self.tmon_dim, i) * basis(self.tmon_dim, 0).dag()
        )
        sy_gi = (
            -1j * basis(self.tmon_dim, 0) * basis(self.tmon_dim, i).dag()
            + 1j * basis(self.tmon_dim, i) * basis(self.tmon_dim, 0).dag()
        )
        sz_gi = (
            basis(self.tmon_dim, 0) * basis(self.tmon_dim, 0).dag()
            - basis(self.tmon_dim, i) * basis(self.tmon_dim, i).dag()
        )
        # below assumes that tmon is in position 2 (zero indexed) in truncated_dims
        tmon_idx = 2
        self.sx = id_wrap_ops(sx_gi, tmon_idx, self.truncated_dims)
        self.sy = id_wrap_ops(sy_gi, tmon_idx, self.truncated_dims)
        self.sz = id_wrap_ops(sz_gi, tmon_idx, self.truncated_dims)
        # define below ops for collapse operators
        self.sminus_ge = id_wrap_ops(
            basis(3, 0) * basis(3, 1).dag(), tmon_idx, self.truncated_dims
        )
        self.s_gg = id_wrap_ops(
            basis(3, 0) * basis(3, 0).dag(), tmon_idx, self.truncated_dims
        )
        self.s_ee = id_wrap_ops(
            basis(3, 1) * basis(3, 1).dag(), tmon_idx, self.truncated_dims
        )
        if i == 2:
            self.sminus_ef = id_wrap_ops(
                basis(3, 1) * basis(3, 2).dag(), tmon_idx, self.truncated_dims
            )
            self.s_ff = id_wrap_ops(
                basis(3, 2) * basis(3, 2).dag(), tmon_idx, self.truncated_dims
            )

    def construct_c_ops(
        self,
        a: Qobj,
        b: Qobj,
        Gamma_1_ge=0.0,
        Gamma_1_ef=0.0,
        Gamma_phi_gg=0.0,
        Gamma_phi_ee=0.0,
        Gamma_phi_ff=0.0,
        Gamma_1_res=0.0,
        Gamma_phi_res=0.0,
        nth=0.0,
        **kwargs
    ):
        c_ops = [
            np.sqrt(Gamma_1_ge) * self.sminus_ge,
            np.sqrt(Gamma_1_ef) * self.sminus_ef,
            np.sqrt(nth * Gamma_1_ge) * self.sminus_ge.dag(),
            np.sqrt(nth * Gamma_1_ef) * self.sminus_ef.dag(),
            np.sqrt(Gamma_phi_gg) * self.s_gg,
            np.sqrt(Gamma_phi_ee) * self.s_ee,
            np.sqrt(Gamma_phi_ff) * self.s_ff,
            np.sqrt(Gamma_1_res) * a,
            np.sqrt(Gamma_1_res) * b,
            np.sqrt(nth * Gamma_1_res) * a.dag(),
            np.sqrt(nth * Gamma_1_res) * b.dag(),
            np.sqrt(Gamma_phi_res) * a.dag() * a,
            np.sqrt(Gamma_phi_res) * b.dag() * b,
        ]
        return c_ops

    def cZZU(
        self,
        a_op: Qobj,
        b_op: Qobj,
        chi: float,
        c_ops: List[Qobj] = None,
        state: Qobj = None,
    ):
        """
        Parameters
        ----------
        a_op: Qobj
            lowering operator for the first bosonic mode, coupled to the transmon
        b_op: Qobj
            lowering operator for the second bosonic mode
        chi: float
            dispersive shift between cavity a and its transmon
        c_ops: List[Qobj]
            collapse operators
        state: Qobj
            if None, we compute the propagator. if instead we pass a Qobj, the
            time evolution of that specific state is computed

        Returns
        -------
        propagator corresponding to the cZZ unitary as described in
        Tsunoda et al. arXiv:2212.11196 (2022) and Teoh et al. arXiv:2212.12077 (2022)
        """
        g = np.sqrt(3) * chi / 2
        H = -0.5 * chi * self.sz * a_op.dag() * a_op + 0.5 * g * (
            a_op.dag() * b_op + a_op * b_op.dag()
        )
        Omega = np.sqrt(g**2 + (chi / 2) ** 2)
        options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )
        return prop_or_mesolve_factory(
            H, 2.0 * np.pi / Omega, c_ops, state, options=options
        )

    @staticmethod
    def R_osc(a_op: Qobj, phi: float, c_ops: List[Qobj] = None, state: Qobj = None):
        """
        this gate is done in software and thus can be done with unit fidelity
        (hence why the collapse operators are not used even in the case when they are passed)
        Parameters
        ----------
        a_op: Qobj
            lowering operator of the mode we want to apply a
            phase correction to
        phi: float
            parameter describing the phase correction
        c_ops: List[Qobj]
            unused collapse operators
        state: Qobj
            if None, we compute the propagator. if instead we pass a Qobj, the
            time evolution of that specific state is computed
        Returns
        -------
        propagator corresponding to a single-cavity rotation
        """
        cav_rotation = (-1j * phi * a_op.dag() * a_op).expm()
        if c_ops is None:
            if state is None:
                return cav_rotation
            else:
                assert state.isket
                return cav_rotation * state
        else:
            if state is None:
                return to_super(cav_rotation)
            else:
                assert state.isoper
                return cav_rotation * state * cav_rotation.dag()

    def R_tmon(
        self,
        g: float,
        t: float,
        direction: str = "X",
        c_ops: List[Qobj] = None,
        state: Qobj = None,
    ):
        r"""
        Parameters
        ----------
        g: float
            drive strength: the Hamiltonian is :math:`H = 0.5 * g * \sigma_{i}, i\in\{x,y,z\}`
        t: float
            time for which the interaction should act
        direction: str
            either "X", "Y", or "Z" depending on the rotation direction
        c_ops: List[Qobj]
            collapse operators
        state: Qobj
            if None, we compute the propagator. if instead we pass a Qobj, the
            time evolution of that specific state is computed

        Returns
        -------
        propagator corresponding to a transmon rotation
        """
        if direction == "X":
            s_op = self.sx
        elif direction == "Y":
            s_op = self.sy
        elif direction == "Z":
            s_op = self.sz
        else:
            raise RuntimeError("specified direction must be 'X', 'Y', or 'Z'")
        options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )
        return prop_or_mesolve_factory(0.5 * g * s_op, t, c_ops, state, options=options)

    def beamsplitter(
        self, a_op: Qobj, b_op: Qobj, g: float, t: float, c_ops=None, state: Qobj = None
    ):
        H = 0.5 * g * a_op.dag() * b_op + 0.5 * np.conjugate(g) * b_op.dag() * a_op
        options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )
        return prop_or_mesolve_factory(H, t, c_ops, state, options=options)

    def cZU(self, a_op: Qobj, chi: float, c_ops=None, state: Qobj = None):
        H = 0.5 * chi * self.sz * a_op.dag() * a_op
        t = np.pi / chi
        options = Options(
            store_final_state=True, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps
        )
        return prop_or_mesolve_factory(H, t, c_ops, state, options=options)

    def cZZZU(
        self,
        a_op: Qobj,
        b_op: Qobj,
        c_op: Qobj,
        chi: float,
        c_ops=None,
        state: Qobj = None,
    ):
        if state is None:
            U1 = self.cZZU(b_op, a_op, chi, c_ops=c_ops)
            U2 = self.cZZU(b_op, c_op, chi, c_ops=c_ops)
            U3 = self.cZU(b_op, chi, c_ops=c_ops)
            return U1 * U2 * U3
        else:
            U1_result = self.cZZU(b_op, a_op, chi, c_ops=c_ops, state=state)
            U2_result = self.cZZU(
                b_op, c_op, chi, c_ops=c_ops, state=U1_result.final_state
            )
            return self.cZU(b_op, chi, c_ops=c_ops, state=U2_result.final_state)

    def U_eJP_func(
        self, a_op: Qobj, b_op: Qobj, params, c_ops=None, state: Qobj = None
    ):
        """
        exponentiated JP operation. consists of hadamard, JP, X_{theta}, JP, hadamard
        Parameters
        ----------
        a_op: Qobj
            lowering operator for the first bosonic mode, coupled to the transmon
        b_op: Qobj
            lowering operator for the second bosonic mode
        params: tuple
            (tmon_d_strength, chi)
        c_ops: List[Qobj]
            collapse operators
        state: Qobj
            if None, we compute the propagator. if instead we pass a Qobj, the
            time evolution of that specific state is computed

        Returns
        -------
            propagator or state as described above
        """
        (tmon_d_strength, chi) = params
        tmon_d_time = np.pi / (2 * tmon_d_strength)
        if state is None:
            JPP = self.cZZU(a_op, b_op, chi, c_ops=c_ops)
            U_tmon_y = self.R_tmon(
                tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops
            )
            # don't want to use dagger here as it can be a superoperator (need to understand better)
            U_tmon_y_min = self.R_tmon(
                -tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops, state=state
            )
            U_tmon_x = self.R_tmon(
                tmon_d_strength, tmon_d_time, direction="X", c_ops=c_ops
            )
            U_a = self.R_osc(a_op, -np.pi / 2, c_ops=c_ops)
            U_b = self.R_osc(b_op, -np.pi / 2, c_ops=c_ops)
            return U_a * U_b * U_tmon_y_min * JPP * U_tmon_x * JPP * U_tmon_y
        else:
            # track the time evolution of the starting state, passing the final state
            # from each operation as input for the next operation
            U_tmon_y = self.R_tmon(
                tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops, state=state
            )
            JPP_1 = self.cZZU(a_op, b_op, chi, c_ops=c_ops, state=U_tmon_y)
            U_tmon_x = self.R_tmon(
                tmon_d_strength, tmon_d_time, direction="X", c_ops=c_ops, state=JPP_1
            )
            JPP_2 = self.cZZU(a_op, b_op, chi, c_ops=c_ops, state=U_tmon_x)
            U_tmon_y_min = self.R_tmon(
                -tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops, state=JPP_2
            )
            U_b = self.R_osc(b_op, -np.pi / 2, c_ops=c_ops, state=U_tmon_y_min)
            U_a = self.R_osc(a_op, -np.pi / 2, c_ops=c_ops, state=U_b)
            return U_a

    def U_erasure_check(self, a_op: Qobj, b_op: Qobj, params, c_ops=None):
        raise NotImplementedError("not implemented yet in the case of using mesolve")
        # (tmon_d_strength, chi) = params
        # tmon_d_time = np.pi / (2 * tmon_d_strength)
        # JPP = self.cZZU(a_op, b_op, chi, c_ops=c_ops)
        # U_tmon_y = self.R_tmon(tmon_d_strength, tmon_d_time, direction="Y", c_ops=c_ops)
        # U_a = self.R_osc(a_op, np.pi / 2, c_ops=c_ops)
        # U_b = self.R_osc(b_op, np.pi / 2, c_ops=c_ops)
        # return U_tmon_y * U_a * U_b * JPP * U_tmon_y

    def measurement_op_tmon_projector(self, idx):
        """project onto a specific tmon eigenstate"""
        Fock_states_spec = [
            (i, j, idx) for i in range(self.cavity_dim) for j in range(self.cavity_dim)
        ]
        Fock_states = construct_basis_states_list(Fock_states_spec, self.truncated_dims)
        return sum([Fock_state * Fock_state.dag() for Fock_state in Fock_states])


class SimulateBosonicOperationsDR(SimulateBosonicOperations, DualRailMixin):
    def __init__(
        self, gf_tmon=True, tmon_dim=3, cavity_dim=3, nsteps=2000, atol=1e-8, rtol=1e-6
    ):
        super().__init__(
            gf_tmon=gf_tmon,
            tmon_dim=tmon_dim,
            cavity_dim=cavity_dim,
            nsteps=nsteps,
            atol=atol,
            rtol=rtol,
        )

    def measurement_op_DR_parity(self):
        """measurement operator projecting onto the DR basis. Assumes identity operation
        on the transmon"""
        measurement_op = 0.0
        for idx in range(self.tmon_dim):
            Fock_states_spec = [(i, j, idx) for i in range(2) for j in range(2)]
            # TODO confusing in the below that self.truncated_dims refers to the SR system...
            Fock_states = construct_basis_states_list(
                Fock_states_spec, self.truncated_dims
            )
            Fock_states_DR = self.DR_basis(Fock_states)
            measurement_op += sum(
                [
                    detected_state * detected_state.dag()
                    for detected_state in Fock_states_DR
                ]
            )
        return measurement_op

    def V_2_op(self):
        """see https://arxiv.org/abs/1111.6950 , operator that SWAPs the internal indices
        to properly order a tensor product of superoperators"""
        return SWAP_op(1, 2, 4 * [self.truncated_dims])
