from numpy import ndarray


class Qcirc:
    def __init__(self, name_dict: dict):
        self.name_dict = name_dict

    def SWAP(self, qvec: ndarray, name0: str, name1: str):
        idx0 = self.name_dict[name0]
        idx1 = self.name_dict[name1]
        val0 = qvec[idx0]
        val1 = qvec[idx1]
        qvec[idx0] = val1
        qvec[idx1] = val0
        return qvec

    def CNOT(self, qvec: ndarray, name0: str, name1: str, control_in_zero: bool = False):
        """Assumption is that name0 corresponds to the control"""
        if control_in_zero:
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

    def CSWAP(self, qvec: ndarray, name0: str, name1: str, name2: str, control_in_zero: bool = False):
        if control_in_zero:
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

    def CCNOT(self, qvec: ndarray, name0: str, name1: str, name2: str, control_0_in_0: bool = False,
              control_1_in_0: bool = False):
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

    def ideal_qram(self, qvec: ndarray):
        address_str = ""
        for i, key in enumerate(self.name_dict.keys()):
            if key[0:4] == "addr":
                address_str = address_str + str(qvec[self.name_dict[key]])
        data_idx = int(address_str, 2)
        qvec[self.name_dict["bus"]] = qvec[self.name_dict["data"+str(data_idx)]]
        return qvec

    def run(self, qvec: ndarray, gate_list: list):
        """

        Parameters
        ----------
        qvec: input qvec, assumed for now to not be in a superposition state
        dict_of_gates
            dictionary where the keys are gates and the values are
            the names of the involved subsystems

        Returns
        -------
            updated qvec
        """
        for i, (gate_name, gate_params) in enumerate(gate_list):
            gate_func = getattr(self, gate_name)
            qvec = gate_func(qvec, *gate_params)
        return qvec
