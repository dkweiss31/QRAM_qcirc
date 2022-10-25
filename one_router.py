import numpy as np

from main import Qcirc

name_dict = {"address": 0, "bus": 1, "router": 2, "data0": 3, "data1": 4}
gate_list = (("SWAP", ["address", "router"]),
             ("CCNOT", ["data0", "router", "bus", False, True]),
             ("CCNOT", ["data1", "router", "bus", False, False]),
             ("SWAP", ["address", "router"]))
test_qcirc = Qcirc(name_dict=name_dict, gate_list=gate_list)
test_qcirc.loop_over_all_inputs()

# for addr in range(2):
#     for d1 in range(2):
#         for d2 in range(2):
#             qvec = np.array([addr, 0, 0, d1, d2])
#             init_qvec = np.copy(qvec)
#             result = test_qcirc.run_single(qvec)
#             ideal_result = test_qcirc.ideal_qram(init_qvec)
#             assert np.allclose(result, ideal_result)
#             print(init_qvec, result, ideal_result)
