import warnings

import numpy as np

from main import Qcirc
name_list = ["address0", "address1", "bus",
             "router0", "router1", "router2", "input",
             "output0", "output1", "data0", "data1", "data2", "data3"]

name_dict = dict(zip(name_list, np.arange(0, len(name_list))))
gate_list = (("CNOT", ["address0", "router0"]),
             ("SWAP", ["address1", "input"]),
             ("CSWAP", ["router0", "input", "output0", True]),
             ("CSWAP", ["router0", "input", "output1", False]),
             ("SWAP", ["output0", "router1"]),
             ("SWAP", ["output1", "router2"]),
             ("CCNOT", ["router1", "data0", "output0", True, False]),
             ("CCNOT", ["router1", "data1", "output0", False, False]),
             ("CCNOT", ["router2", "data2", "output1", True, False]),
             ("CCNOT", ["router2", "data3", "output1", False, False]),
             # ("CCNOT", ["router0", "output0", "input", True, False]),  # change to CSWAP?
             # ("CCNOT", ["router0", "output1", "input", False, False]),
             ("CSWAP", ["router0", "output0", "input", True]),  # change to CSWAP?
             ("CSWAP", ["router0", "output1", "input", False]),
             ("CNOT", ["input", "bus"]),  # change to SWAP?
             # ("CCNOT", ["router0", "output0", "input", True, False]),  # change to CSWAP?
             # ("CCNOT", ["router0", "output1", "input", False, False]),
             ("CSWAP", ["router0", "output0", "input", True]),  # change to CSWAP?
             ("CSWAP", ["router0", "output1", "input", False]),
             ("CCNOT", ["router2", "data2", "output1", True, False]),
             ("CCNOT", ["router2", "data3", "output1", False, False]),
             ("CCNOT", ["router1", "data0", "output0", True, False]),
             ("CCNOT", ["router1", "data1", "output0", False, False]),
             ("SWAP", ["output0", "router1"]),
             ("SWAP", ["output1", "router2"]),
             ("CSWAP", ["router0", "input", "output0", True]),
             ("CSWAP", ["router0", "input", "output1", False]),
             ("SWAP", ["address1", "input"]),
             ("CNOT", ["address0", "router0"]),
             )
test_qcirc = Qcirc(name_dict=name_dict)

for d0 in range(2):
    for d1 in range(2):
        for d2 in range(2):
            for d3 in range(2):
                for addr0 in range(2):
                    for addr1 in range(2):
                        qvec = np.zeros(len(name_list), dtype=int)
                        qvec[name_dict["address0"]] = addr0
                        qvec[name_dict["address1"]] = addr1
                        qvec[name_dict["data0"]] = d0
                        qvec[name_dict["data1"]] = d1
                        qvec[name_dict["data2"]] = d2
                        qvec[name_dict["data3"]] = d3
                        init_qvec = np.copy(qvec)
                        result = test_qcirc.run(qvec, gate_list)
                        ideal_result = test_qcirc.ideal_qram(init_qvec)
                        print(init_qvec, result, ideal_result)
                        if not np.allclose(result, ideal_result):
                            print("not ideal")

