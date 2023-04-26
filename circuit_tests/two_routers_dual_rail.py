import warnings

import numpy as np

from circuit import Qcirc
name_list = ["address0", "address1", "bus",
             "router0", "router1", "router2", "input",
             "output0", "output1", "data0", "data1", "data2", "data3"]

name_dict = dict(zip(name_list, np.arange(0, len(name_list))))
gate_list = (("CNOT", ["address1", "router0"]),
             ("SWAP", ["router0", "input"]),
             ("SWAP", ["address0", "address1"]),
             ("CNOT", ["address1", "router0"]),
             ("CSWAP", ["router0", "input", "output0", True]),
             ("CSWAP", ["router0", "input", "output1", False]),
             ("SWAP", ["output0", "router1"]),
             ("SWAP", ["output1", "router2"]),
             ("CCNOT", ["router1", "data0", "output0", True, False]),
             ("CCNOT", ["router1", "data1", "output0", False, False]),
             ("CCNOT", ["router2", "data2", "output1", True, False]),
             ("CCNOT", ["router2", "data3", "output1", False, False]),
             ("CSWAP", ["router0", "output0", "input", True]),  # change to CSWAP?
             ("CSWAP", ["router0", "output1", "input", False]),
             ("SWAP", ["input", "router0"]),
             ("CNOT", ["router0", "bus"]),
             ("SWAP", ["input", "router0"]),
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
             ("CNOT", ["address1", "router0"]),
             ("SWAP", ["address0", "address1"]),
             ("SWAP", ["router0", "input"]),
             ("CNOT", ["address1", "router0"]),
             )
test_qcirc = Qcirc(name_dict=name_dict, gate_list=gate_list)
test_qcirc.loop_over_all_inputs_test()
