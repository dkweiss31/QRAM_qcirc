import numpy as np

from utils.hashing import Hashing


class TestHashing:
    @classmethod
    def setup_class(cls):
        cls.hashing = Hashing(num_exc=1, number_degrees_freedom=2)

    def test_ptrace(self):
        dm = np.array([[0.1, 0.3, 0.4],
                       [0.3, 0.5, 0.2],
                       [0.4, 0.2, 0.4]])
        new_dm = self.hashing.ptrace(dm, keep_idxs=[0,])
        analytic_dm = np.array([[dm[0, 0] + dm[2, 2], dm[0, 1]],
                                [dm[1, 0], dm[1, 1]]])
        assert np.allclose(new_dm, analytic_dm)
