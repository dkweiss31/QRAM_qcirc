from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from qutip import Qobj
from scipy.special import comb


class Hashing:
    """Helper class for efficiently constructing raising and lowering operators
    using a global excitation cutoff scheme, as opposed to the more commonly used
    number of excitations per mode cutoff, which can be easily constructed
    using kronecker product. The ideas herein are based on the excellent
    paper
    [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).
    """

    num_exc: int  # up to and including the number of global excitations to keep
    number_degrees_freedom: int  # number of degrees of freedom of the system

    def __init__(self, num_exc, number_degrees_freedom) -> None:
        self.num_exc = num_exc
        self.number_degrees_freedom = number_degrees_freedom
        self.sqrt_prime_list = np.sqrt(
            [
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                97,
                101,
                103,
                107,
                109,
                113,
                127,
                131,
                137,
                139,
                149,
                151,
                157,
                163,
                167,
                173,
                179,
                181,
                191,
                193,
                197,
                199,
                211,
                223,
                227,
                229,
                233,
                239,
                241,
                251,
                257,
                263,
                269,
                271,
                277,
                281,
                283,
                293,
                307,
                311,
                313,
                317,
                331,
                337,
                347,
                349,
                353,
                359,
                367,
                373,
                379,
                383,
                389,
                397,
                401,
                409,
                419,
                421,
                431,
                433,
                439,
                443,
                449,
                457,
                461,
                463,
                467,
                479,
                487,
                491,
                499,
                503,
                509,
                521,
                523,
                541,
                547,
                557,
                563,
                569,
                571,
                577,
                587,
                593,
                599,
                601,
                607,
                613,
                617,
                619,
                631,
                641,
                643,
                647,
                653,
                659,
                661,
                673,
                677,
                683,
                691,
                701,
                709,
                719,
                727,
                733,
                739,
                743,
                751,
                757,
                761,
                769,
                773,
                787,
                797,
                809,
                811,
                821,
                823,
                827,
                829,
                839,
                853,
                857,
                859,
                863,
                877,
                881,
                883,
                887,
                907,
                911,
                919,
                929,
                937,
                941,
                947,
                953,
                967,
                971,
                977,
                983,
                991,
                997,
            ]
        )

    def gen_basis_vectors(self, number_degrees_freedom=None, num_exc=None) -> ndarray:
        """Generate basis vectors using Zhang algorithm. `func` allows for inclusion of other vectors,
        such as those with negative entries (see CurrentMirrorGlobal)"""
        if number_degrees_freedom is None:
            number_degrees_freedom = self.number_degrees_freedom
        if num_exc is None:
            num_exc = self.num_exc
        vector_list = [np.zeros(number_degrees_freedom)]
        for total_exc in range(
            1, num_exc + 1
        ):  # No excitation number conservation as in [1]
            previous_vector = np.zeros(number_degrees_freedom)
            previous_vector[0] = total_exc
            vector_list.append(previous_vector)
            while (
                previous_vector[-1] != total_exc
            ):  # step through until the last entry is total_exc
                next_vector = self.generate_next_vector(previous_vector, total_exc)
                vector_list.append(next_vector)
                previous_vector = next_vector
        return np.array(vector_list)

    def ptrace(self, density_matrix, keep_idxs):
        basis_vectors = self.gen_basis_vectors()
        num_keep_idxs = len(keep_idxs)
        remove_idxs = np.array(
            [idx for idx in range(self.number_degrees_freedom) if idx not in keep_idxs]
        )
        red_dim = int(comb(self.num_exc + num_keep_idxs, num_keep_idxs))
        new_dm = np.zeros((red_dim, red_dim), dtype=complex)
        new_basis_vecs = self.gen_basis_vectors(num_keep_idxs, self.num_exc)
        new_tags, new_index_array = self._gen_tags(new_basis_vecs)
        for row_idx in range(self.hilbert_dim()):
            for col_idx in range(row_idx, self.hilbert_dim()):
                ket = basis_vectors[row_idx, :]
                bra = basis_vectors[col_idx, :]
                # only if the two masks are the same does this
                # information survive the partial trace
                if np.allclose(ket.take(remove_idxs), bra.take(remove_idxs)):
                    new_ket_tag = self.hash(ket.take(keep_idxs))
                    new_bra_tag = self.hash(bra.take(keep_idxs))
                    ket_index = np.searchsorted(new_tags, new_ket_tag)
                    bra_index = np.searchsorted(new_tags, new_bra_tag)
                    new_dm[
                        new_index_array[ket_index], new_index_array[bra_index]
                    ] += density_matrix[row_idx, col_idx]
                    if row_idx != col_idx:
                        new_dm[
                            new_index_array[bra_index], new_index_array[ket_index]
                        ] += density_matrix[col_idx, row_idx]
        return Qobj(new_dm)

    @staticmethod
    def generate_next_vector(prev_vec: ndarray, radius: int) -> ndarray:
        """Algorithm for generating all vectors with positive entries of a given Manhattan length, specified in
        [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010)"""
        k = 0
        for num in range(len(prev_vec) - 2, -1, -1):
            if prev_vec[num] != 0:
                k = num
                break
        next_vec = np.zeros_like(prev_vec)
        next_vec[0:k] = prev_vec[0:k]
        next_vec[k] = prev_vec[k] - 1
        next_vec[k + 1] = radius - np.sum([next_vec[i] for i in range(k + 1)])
        return next_vec

    def a_operator(self, i: int) -> Qobj:
        """Construct the lowering operator for mode `i`.

        Parameters
        ----------
        i: int
            integer specifying the mode whose annihilation operator we would like to construct

        Returns
        -------
        Qobj
        """
        basis_vectors = self.gen_basis_vectors()
        tags, index_array = self._gen_tags(basis_vectors)
        dim = self.hilbert_dim()
        a = np.zeros((dim, dim))
        for w, vec in enumerate(basis_vectors):
            if vec[i] >= 1:
                temp_coefficient = np.sqrt(vec[i])
                basis_index = self._find_lowered_vector(vec, i, tags, index_array)
                if (
                    basis_index is not None
                ):  # Should not be the case here, only an issue for charge basis
                    a[basis_index, w] = temp_coefficient
        return Qobj(a)

    def trace_out_dict(self, state_dict, keep_idxs):
        return {
            label: self.ptrace(final_state, keep_idxs)
            for label, final_state in state_dict.items()
        }

    def _find_lowered_vector(
        self,
        vector: ndarray,
        i: int,
        tags: ndarray,
        index_array: ndarray,
        raised_or_lowered="lowered",
    ) -> Optional[int]:
        if raised_or_lowered == "lowered":
            pm_1 = -1
        elif raised_or_lowered == "raised":
            pm_1 = +1
        else:
            raise ValueError("only raised or lowered recognized")
        temp_vector = np.copy(vector)
        temp_vector[i] = vector[i] + pm_1
        temp_vector_tag = self.hash(temp_vector)
        index = np.searchsorted(tags, temp_vector_tag)
        if not np.allclose(tags[index], temp_vector_tag):
            return None
        basis_index = index_array[index]
        return basis_index

    def hilbert_dim(self) -> int:
        """Using the global excitation scheme the total number of states
        is given by the hockey-stick identity"""
        return int(
            comb(
                self.num_exc + self.number_degrees_freedom, self.number_degrees_freedom
            )
        )

    def hash(self, vector: ndarray) -> ndarray:
        """Generate the (unique) identifier for a given vector `vector`"""
        dim = len(vector)
        return np.sum(self.sqrt_prime_list[0:dim] * vector)

    def _gen_tags(self, basis_vectors: ndarray) -> Tuple[ndarray, ndarray]:
        """Generate the identifiers for all basis vectors `basis_vectors`"""
        dim = basis_vectors.shape[0]
        tag_list = np.array([self.hash(basis_vectors[i, :]) for i in range(dim)])
        index_array = np.argsort(tag_list)
        tag_list = tag_list[index_array]
        return tag_list, index_array
