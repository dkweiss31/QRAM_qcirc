from collections import OrderedDict
from functools import partial
from itertools import product
from typing import List

from qutip import (
    tensor,
    basis,
    Qobj,
    operator_to_vector,
    vector_to_operator,
    sigmaz,
    destroy,
    liouvillian,
    to_super,
    qeye,
    mesolve,
    Options,
)
import numpy as np
from utils import id_wrap_ops, project_U, construct_basis_states_list, get_map


class SimulateBosonicOperations:
    """ """

    def __init__(self, gf_tmon=True, tmon_dim=3, cavity_dim=3, control_dt=1.0):
        self.gf_tmon = gf_tmon
        self.tmon_dim = tmon_dim
        self.cavity_dim = cavity_dim
        self.truncated_dims = [cavity_dim, cavity_dim, tmon_dim]
        self.control_dt = control_dt
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
        c_ops: List[Qobj]
            collapse operators

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
        return self._prop_or_mesolve_factory(H, 2.0 * np.pi / Omega, c_ops, state)

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

    def _prop_or_mesolve_factory(self, H, t, c_ops, state):
        if state is None:
            return self._propagator(H, t, c_ops=c_ops)
        else:
            tlist = np.linspace(0.0, t, int(t / self.control_dt))
            result = mesolve(
                H, state, tlist, c_ops=c_ops, options=Options(store_final_state=True)
            )
            return result.final_state

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
        return self._prop_or_mesolve_factory(0.5 * g * s_op, t, c_ops, state)

    @staticmethod
    def _propagator(H: Qobj, t: float, c_ops: List[Qobj] = None):
        """
        Parameters
        ----------
        H: Qobj
            Hamiltonian
        t: float
            time for which the Hamiltonian acts
        c_ops: List[Qobj]
            list of collapse operators

        Returns
        -------
        propagator corresponding to the Hamiltonian acting for a time :math:`t`.
        If c_ops are not passed, we assume the Hamiltonian is time
        independent and exponentiate it. If c_ops are passed, we construct the
        Liouvillian and exponentiate that instead.
        """
        if c_ops is None:
            return (-1j * H * t).expm()
        return (liouvillian(H, c_ops) * t).expm()

    def beamsplitter(
        self, a_op: Qobj, b_op: Qobj, g: float, t: float, c_ops=None, state: Qobj = None
    ):
        H = 0.5 * g * a_op.dag() * b_op + 0.5 * np.conjugate(g) * b_op.dag() * a_op
        return self._prop_or_mesolve_factory(H, t, c_ops, state)

    def cZU(self, a_op: Qobj, chi: float, c_ops=None, state: Qobj = None):
        H = 0.5 * chi * self.sz * a_op.dag() * a_op
        t = np.pi / chi
        return self._prop_or_mesolve_factory(H, t, c_ops, state)

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

    def cZZ_time(self, chi):
        g = np.sqrt(3) * chi / 2
        Omega = np.sqrt(g**2 + (chi / 2) ** 2)
        return 2.0 * np.pi / Omega

    def U_eJP_func(
        self, a_op: Qobj, b_op: Qobj, params, c_ops=None, state: Qobj = None
    ):
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

    def apply_gate_to_states(self, gate, args, states_dict, num_cpus=1):
        labels_list, states_list = [], []
        for (label, state) in states_dict.items():
            labels_list.append(label)
            states_list.append(state)
        target_map = get_map(num_cpus)
        gate_func = partial(getattr(self, gate), *args)
        # only want to apply the costly function to unique states. below combine the
        # results as appropriate for the propagated states
        mapped_states = list(target_map(gate_func, states_list))
        return dict(zip(labels_list, mapped_states))

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

    def measurement_op_DR_parity(self):
        measurement_op = 0.0
        for idx in range(self.tmon_dim):
            Fock_states_spec = [(i, j, idx) for i in range(2) for j in range(2)]
            Fock_states = construct_basis_states_list(
                Fock_states_spec, self.truncated_dims
            )
            Fock_states_DR, _ = self.DR_basis(Fock_states)
            measurement_op += sum(
                [
                    detected_state * detected_state.dag()
                    for detected_state in Fock_states_DR
                ]
            )
        return measurement_op

    @staticmethod
    def _Fock_prods(dim):
        if type(dim) == int:
            return range(dim)
        else:
            return list(product(*map(range, dim)))

    def SWAP_op(self, idx_0, idx_1, dims=None):
        """SWAP between two subsystems"""
        if dims is None:
            dims = self.truncated_dims
        dim_0 = dims[idx_0]
        dim_1 = dims[idx_1]
        Fock_prods = product(*map(self._Fock_prods, [dim_0, dim_1]))
        result = 0.0
        for Fock_prod in Fock_prods:
            id_list = [qeye(dim) for dim in dims]
            (Fock_0,) = construct_basis_states_list(
                [
                    Fock_prod[0],
                ],
                dim_0,
            )
            (Fock_1,) = construct_basis_states_list(
                [
                    Fock_prod[1],
                ],
                dim_1,
            )
            id_list[idx_0] = Fock_1 * Fock_0.dag()
            id_list[idx_1] = Fock_0 * Fock_1.dag()
            result += tensor(*id_list)
        return result

    def V_2_op(self):
        """see https://arxiv.org/abs/1111.6950 , operator that SWAPs the internal indices
        to properly order a tensor product of superoperators"""
        return self.SWAP_op(1, 2, dims=4 * [self.truncated_dims])

    @staticmethod
    def DR_basis(SR_comp_bas_states):
        # express logical DR states in terms of the basis states of the cavities
        # basis states originally in |router, input, tmon=0>
        # ordered as router, input, router, input
        # |00>_{L} = |10>_{r}|10>_{i} --> |1>_{r}|1>_{i}|0>_{r}|0>_{i}
        # |01>_{L} = |10>|01> --> |1>|0>|0>|1>
        # |10>_{L} = |01>|10> --> |0>|1>|1>|0>
        # |11>_{L} = |01>|01> --> |0>|0>|1>|1>
        basis_state_DR = [
            tensor(SR_comp_bas_states[3], SR_comp_bas_states[0]),
            tensor(SR_comp_bas_states[2], SR_comp_bas_states[1]),
            tensor(SR_comp_bas_states[1], SR_comp_bas_states[2]),
            tensor(SR_comp_bas_states[0], SR_comp_bas_states[3]),
        ]
        labels_basis_states = ["1100", "1001", "0110", "0011"]
        return basis_state_DR, labels_basis_states

    def operator_basis_lidar(
        self, basis_states: list, label_list: list = None
    ) -> (dict, dict):
        """
        Parameters
        ----------
        basis_states: list
            list of the basis states with which to construct the Lidar basis (coherences)
            see 10.1103/PhysRevA.77.032322 for more detail (note typo in the paper, missing an i)
        label_list: list
            list of labels that apply to the basis states. If not provided, we provide
            on for you free of charge
        Returns
        -------
            a tuple of dictionaries. The first dictionary contains information on the operators
            whose evolution we want to track. The keys correspond to the coherence, e.g. "12"
            for the coherence |1><2| or 11 for the "coherence" |1><1|. The values are tuples containing
            four pieces of information. first the operator in question, next a tuple of coefficients
            in the state decomposition (really density matrices, but call them "states" to differentiate from the
            operator coherences which are not density matrices) of the operator, next is those states,
            and finally a tuple of labels corresponding to the states.
        """
        if label_list is None:
            label_list = range(len(basis_states))
        op_dict = {}
        unique_state_dict = {}
        for i, ket_0 in enumerate(basis_states):
            for j, ket_1 in enumerate(basis_states):
                if i == j:
                    op_dict[label_list[i] + label_list[i]] = (
                        ket_0 * ket_0.dag(),
                        (1.0,),
                        (ket_0 * ket_0.dag(),),
                        ((label_list[i],),),
                    )
                    if (label_list[i],) not in unique_state_dict:
                        unique_state_dict[(label_list[i],)] = ket_0 * ket_0.dag()
                else:
                    # slight inefficiency rn is that |ij> + |kl> and |ij> + |kl> get recorded as different states
                    pl_state = (ket_0 + ket_1).unit()
                    min_state = (ket_0 + 1j * ket_1).unit()
                    new_states = (
                        pl_state * pl_state.dag(),
                        min_state * min_state.dag(),
                        ket_0 * ket_0.dag(),
                        ket_1 * ket_1.dag(),
                    )
                    alpha_coeffs = (1, 1j, -0.5 * (1 + 1j), -0.5 * (1 + 1j))
                    label_0 = label_list[i]
                    label_1 = label_list[j]
                    new_labels = (
                        (
                            label_0,
                            1,
                            label_1,
                        ),
                        (
                            label_0,
                            1j,
                            label_1,
                        ),
                        (label_0,),
                        (label_1,),
                    )
                    op_dict[label_list[i] + label_list[j]] = (
                        ket_0 * ket_1.dag(),
                        alpha_coeffs,
                        new_states,
                        new_labels,
                    )
                    unique_state_dict.update(
                        {
                            new_labels[k]: state
                            for k, state in enumerate(new_states)
                            if k not in unique_state_dict
                        }
                    )
                    assert ket_0 * ket_1.dag() == sum(
                        [
                            alpha_coeffs[i] * new_states[i]
                            for i in range(len(alpha_coeffs))
                        ]
                    )
        return op_dict, unique_state_dict

    def construct_final_SR_ops(
        self, SR_op_dict: dict, final_unique_states_dict: dict
    ) -> dict:
        """
        Parameters
        ----------
        SR_op_dict: dict
            op dictionary as returned by operator_basis_lidar
        final_unique_states_dict
            state dictionary in the same form as returned by operator_basis_lidar

        Returns
        -------
            final SR operators according to how the final unique states transform
        """
        final_SR_ops = {}
        for key in SR_op_dict.keys():
            op, coeffs, rhos, labels = SR_op_dict[key]
            final_SR_ops[key] = sum(
                [
                    coeffs[idx] * final_unique_states_dict[label]
                    for idx, label in enumerate(labels)
                ]
            )
        return final_SR_ops

    def construct_final_unique_DR_states(self, unique_DR_state_dict, final_SR_op_dict):
        final_DR_state_dict = {}
        for (label, state) in unique_DR_state_dict.items():
            final_DR_state_dict[label] = self.DR_state_from_SR_ops(
                label, final_SR_op_dict
            )
        return final_DR_state_dict

    def DR_state_from_SR_ops(self, DR_label: tuple, final_SR_op_dict: dict) -> Qobj:
        """
        Parameters
        ----------
        DR_label: tuple
            tuple either of length 1, signifying not a superposition state,
            or of length 3 signifying a superposition state. In this case, the
            first entry is the label of the first state, the third entry is the coefficient
            of the second state and the second entry is the coefficient of the second state
        final_SR_op_dict: dict
            dictionary of the final SR operators. labels are of the form "1100" which indicates
            how the operator |11><00| transforms
        Returns
        -------
            final DR state constructed from final SR ops
        """
        if len(DR_label) == 1:
            return self._DR_state_from_SR_ops(
                DR_label[0], DR_label[0], final_SR_op_dict
            )
        elif len(DR_label) == 3:
            coeff = DR_label[1]
            return (
                self._DR_state_from_SR_ops(DR_label[0], DR_label[0], final_SR_op_dict)
                + self._DR_state_from_SR_ops(DR_label[2], DR_label[2], final_SR_op_dict)
                + np.conj(coeff)
                * self._DR_state_from_SR_ops(DR_label[0], DR_label[2], final_SR_op_dict)
                + coeff
                * self._DR_state_from_SR_ops(DR_label[2], DR_label[0], final_SR_op_dict)
            ).unit()
        else:
            raise RuntimeError("DR_label should have length 1 or 3")

    @staticmethod
    def _DR_state_from_SR_ops(DR_label_1, DR_label_2, final_SR_ops):
        """"""
        SR_label_1 = DR_label_1[0:2] + DR_label_2[0:2]
        SR_label_2 = DR_label_1[2:4] + DR_label_2[2:4]
        return tensor(final_SR_ops[SR_label_1], final_SR_ops[SR_label_2])

    @staticmethod
    def measurement_channel(rho, measurement_op):
        assert measurement_op.type == "oper" and rho.type == "oper"
        new_rho = measurement_op * rho * measurement_op.dag()
        return new_rho, np.trace(new_rho)

    @staticmethod
    def process_fidelity_nielsen(entanglement_fidelity, num_qubits=2):
        dim = num_qubits**2
        return (dim * entanglement_fidelity + 1) / (dim + 1)

    def entanglement_fidelity_nielsen(
        self,
        prop_or_final_states_dict,
        U_ideal,
        basis_states_labels_tuple,
        measurement_op=None,
        ptrace_idxs=None,
        num_qubits=2,
    ) -> (float, float):
        """
        Parameters
        ----------
        prop_or_final_states_dict: Qobj or dict
            either the propogator (superoperator) corresponding to the real time evolution
            or a dictionary of how the basis states of interest evolve
        U_ideal: Qobj
            propogator of the ideal evolution
        basis_states_labels_tuple: tuple
            tuple of the basis states of interest and their labels
        measurement_op: Qobj
            measurement operator, if any
        ptrace_idxs: tuple
            indices to keep if we want to trace over a subsystem
        num_qubits: int
            number of qubits, usually here 2

        Returns
        -------
            returns a tuple of floats corresponding to the entanglement fidelity
            according the Nielsen's formula together with the success probability

        """
        dim = 2**num_qubits
        basis_states, label_list = basis_states_labels_tuple
        op_dict, unique_state_dict = self.operator_basis_lidar(
            basis_states=basis_states, label_list=label_list
        )
        overall_contr = 0.0
        total_prob = 0.0
        # want to change num_states indexing so that we only sum over unique states
        num_states = 0
        for op_key in op_dict.keys():
            op, coeffs, rhos, labels = op_dict[op_key]
            for (coeff, pauli_rho, label) in zip(coeffs, rhos, labels):
                if type(prop_or_final_states_dict) == Qobj:
                    rho = operator_to_vector(pauli_rho)
                    propagated_rho = vector_to_operator(prop_or_final_states_dict * rho)
                else:
                    propagated_rho = prop_or_final_states_dict[label]
                state_contr, prob = self._fidel_individual_state(
                    propagated_rho,
                    op,
                    U_ideal,
                    basis_states,
                    measurement_op,
                    ptrace_idxs,
                )
                overall_contr += coeff * state_contr
                total_prob += prob
                num_states += 1
        return overall_contr / dim**2, total_prob / num_states

    def _fidel_individual_state(
        self,
        propagated_rho,
        op,
        U_ideal,
        basis_states,
        measurement_op=None,
        ptrace_idxs=None,
    ):
        if measurement_op is not None:
            propagated_rho, prob = self.measurement_channel(
                propagated_rho, measurement_op
            )
        else:
            prob = 0.0
        if ptrace_idxs is not None:
            propagated_rho = propagated_rho.ptrace(ptrace_idxs)
            op = op.ptrace(ptrace_idxs)
        projected_rho = project_U(propagated_rho, basis_states=basis_states)
        projected_op = project_U(op, basis_states=basis_states)
        state_contr = np.trace(
            U_ideal * projected_op.dag() * U_ideal.dag() * projected_rho
        )
        return state_contr, prob
