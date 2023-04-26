import numpy as np
from numpy import ndarray
from itertools import product


class Qcirc:
    """
    name_dict
        dictionary of the involved subsystems, assumed to
        be in the form of name: index
    gate_list
        list of lists, where the elements are pairs of
        gates and the involved subsystems
    """
    def __init__(self, name_dict: dict, gate_list: list):
        self.name_dict = name_dict
        self.gate_list = gate_list

    def SWAP(self, qvec: ndarray, name0: str, name1: str):
        """

        Parameters
        ----------
        qvec
            state vector to be updated by the SWAP gate
        name0
            name of the first qubit
        name1
            name of the second qubit

        Returns
        -------
            updated state vector

        """
        idx0 = self.name_dict[name0]
        idx1 = self.name_dict[name1]
        val0 = qvec[idx0]
        val1 = qvec[idx1]
        qvec[idx0] = val1
        qvec[idx1] = val0
        return qvec

    def CNOT(
        self, qvec: ndarray, name0: str, name1: str, control_in_0: bool = False
    ):
        """

        Parameters
        ----------
        qvec
            state vector to be updated by the CNOT gate
        name0
            name of the control qubit
        name1
            name of the target qubit
        control_in_0
            bool representing if the gate should be applied if the control
            is in zero (opposite of the usual case)

        Returns
        -------
            updated state vector

        """
        if control_in_0:
            control_bool = 1
        else:
            control_bool = 0
        idx0 = self.name_dict[name0]
        val0 = qvec[idx0]
        if (val0 + control_bool) % 2:
            idx1 = self.name_dict[name1]
            qvec[idx1] = (qvec[idx1] + 1) % 2
            return qvec
        else:
            return qvec

    def CSWAP(
        self,
        qvec: ndarray,
        name0: str,
        name1: str,
        name2: str,
        control_in_0: bool = False,
    ):
        """

        Parameters
        ----------
        qvec
            state vector to be updated by the CSWAP gate
        name0
            name of the control qubit
        name1
            name of the first target qubit
        name2
            name of the second target qubit
        control_in_0
            bool representing if the gate should be applied if the control
            is in zero (opposite of the usual case)

        Returns
        -------
            updated state vector

        """
        if control_in_0:
            control_bool = 1
        else:
            control_bool = 0
        idx0 = self.name_dict[name0]
        val0 = qvec[idx0]
        if (val0 + control_bool) % 2:
            idx1 = self.name_dict[name1]
            idx2 = self.name_dict[name2]
            val1 = qvec[idx1]
            val2 = qvec[idx2]
            qvec[idx1] = val2
            qvec[idx2] = val1
            return qvec
        else:
            return qvec

    def CCNOT(
        self,
        qvec: ndarray,
        name0: str,
        name1: str,
        name2: str,
        control_0_in_0: bool = False,
        control_1_in_0: bool = False,
    ):
        """
        
        Parameters
        ----------
        qvec
            state vector to be updated by the CCNOT gate
        name0
            name of the first control qubit
        name1
            name of the second control qubit
        name2
            name of the target qubit
        control_0_in_0
            bool representing if the gate should be applied if the first control
            is in zero (opposite of the usual case)
        control_1_in_0
            bool representing if the gate should be applied if the second control
            is in zero (opposite of the usual case)

        Returns
        -------
            updated state vector

        """
        if control_0_in_0:
            control_0_bool = 1
        else:
            control_0_bool = 0
        if control_1_in_0:
            control_1_bool = 1
        else:
            control_1_bool = 0
        idx0 = self.name_dict[name0]
        val0 = qvec[idx0]
        idx1 = self.name_dict[name1]
        val1 = qvec[idx1]
        if (val0 + control_0_bool) % 2 and (val1 + control_1_bool) % 2:
            idx2 = self.name_dict[name2]
            qvec[idx2] = (qvec[idx2] + 1) % 2
            return qvec
        else:
            return qvec

    def reset(self, qvec: ndarray, name0):
        """reset a specific subsystem to the ground state"""
        qvec[self.name_dict[name0]] = 0
        return qvec

    def ideal_qram(self, qvec: ndarray):
        """
        given an initial state vector with classical data specified, output
        the ideal result of a qram query
        Parameters
        ----------
        qvec
            initial state vector

        Returns
        -------
            final state vector
        """
        address_str = ""
        result_qvec = np.copy(qvec)
        for i, key in enumerate(self.name_dict.keys()):
            if key[0:4] == "addr":
                address_str = address_str + str(qvec[self.name_dict[key]])
        data_idx = int(address_str, 2)
        result_qvec[self.name_dict["bus"]] = qvec[
            self.name_dict["data" + str(data_idx)]
        ]
        return result_qvec

    def run_single(self, qvec: ndarray):
        """

        Parameters
        ----------
        qvec
            input qvec, assumed for now to not be in a superposition state

        Returns
        -------
            updated qvec
        """
        for i, (gate_name, gate_params) in enumerate(self.gate_list):
            gate_func = getattr(self, gate_name)
            qvec = gate_func(qvec, *gate_params)
        return qvec

    def loop_over_all_inputs_test(self):
        """
        check that the qram works over all possible inputs (addresses
        and classical data)
        """
        num_subsystems = len(self.name_dict)
        addr_data_dict = {}
        loop_idx = 0
        for i, key in enumerate(self.name_dict.keys()):
            if key[0:4] == "addr" or key[0:4] == "data":
                addr_data_dict[i] = loop_idx
                loop_idx += 1
        init_qvecs = product(range(2), repeat=loop_idx)
        for init_qvec in init_qvecs:
            init_qvec_w_zeros = np.zeros(num_subsystems, dtype=int)
            init_qvec_w_zeros[list(addr_data_dict.keys())] = init_qvec
            original_init_qvec_w_zeros = np.copy(init_qvec_w_zeros)
            result = self.run_single(init_qvec_w_zeros)
            ideal_result = self.ideal_qram(original_init_qvec_w_zeros)
            assert np.allclose(result, ideal_result)
