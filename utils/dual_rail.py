import numpy as np
from qutip import tensor, Qobj

from utils.quantum_helpers import operator_basis_lidar, operators_from_states


class DualRailMixin:
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
        return basis_state_DR

    def DR_state_from_SR_ops(self, DR_label: tuple, final_SR_op_dict: dict) -> Qobj:
        """
        Parameters
        ----------
        DR_label: tuple
            tuple either of length 1, signifying not a superposition state,
            or of length 3 signifying a superposition state. In this case, the
            first entry is the label of the first state, the third entry is the label
            of the second state and the second entry is the coefficient of the second state
        final_SR_op_dict: dict
            dictionary of the final SR operators. labels are of the form "1100" which indicates
            how the operator |11><00| transforms
        Returns
        -------
            final DR state constructed from final SR ops
        """
        if len(DR_label) == 1:
            return self._DR_op_from_SR_ops(DR_label[0], DR_label[0], final_SR_op_dict)
        elif len(DR_label) == 3:
            coeff = DR_label[1]
            return (
                self._DR_op_from_SR_ops(DR_label[0], DR_label[0], final_SR_op_dict)
                + self._DR_op_from_SR_ops(DR_label[2], DR_label[2], final_SR_op_dict)
                + np.conj(coeff)
                * self._DR_op_from_SR_ops(DR_label[0], DR_label[2], final_SR_op_dict)
                + coeff
                * self._DR_op_from_SR_ops(DR_label[2], DR_label[0], final_SR_op_dict)
            ).unit()
        else:
            raise RuntimeError("DR_label should have length 1 or 3")

    @staticmethod
    def _DR_op_from_SR_ops(DR_label_1, DR_label_2, final_SR_ops):
        """construct DR op given DR labels and dictionary of SR operators. the DR labels are assumed to be of the
        form e.g. '1100' to signify the ket |11>|00>. The keys of the SR dict correspond to operators:
        Ex:
            DR_label_1 = "1100" -> |11>|00> (order of router, input, router, input as opposed to logical ordering)
            DR_label_1 = "1001" -> <10|<01|
         -> SR_label_1 = "1110" -> |11><10|
            SR_label_1 = "0001" -> |00><01|
            return |1100><1001|
        """
        len_DR_label_2 = len(DR_label_1) // 2
        SR_label_1 = DR_label_1[0:len_DR_label_2] + DR_label_2[0:len_DR_label_2]
        SR_label_2 = DR_label_1[len_DR_label_2:2*len_DR_label_2] + DR_label_2[len_DR_label_2:2*len_DR_label_2]
        return tensor(final_SR_ops[SR_label_1], final_SR_ops[SR_label_2])

    def DR_final_states(self, SR_basis_states, SR_labels, DR_ideal_final_basis_states, DR_labels,
                        SR_final_cardinal_states):
        op_dict_SR, initial_cardinal_states = operator_basis_lidar(
            SR_basis_states, label_list=SR_labels
        )
        _, ideal_final_cardinal_states_DR = operator_basis_lidar(
            DR_ideal_final_basis_states, label_list=DR_labels
        )
        final_SR_ops = operators_from_states(op_dict_SR, SR_final_cardinal_states)
        final_DR_states = {
            label: self.DR_state_from_SR_ops(label, final_SR_ops)
            for label, state in ideal_final_cardinal_states_DR.items()
        }
        return final_DR_states, ideal_final_cardinal_states_DR
