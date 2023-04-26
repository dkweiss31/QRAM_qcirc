import numpy as np

from circuit import Qcirc

name_dict = {"address": 0, "bus": 1, "router": 2, "data0": 3, "data1": 4}
gate_list = (("SWAP", ["address", "router"]),
             ("CCNOT", ["data0", "router", "bus", False, True]),
             ("CCNOT", ["data1", "router", "bus", False, False]),
             ("SWAP", ["address", "router"]))
test_qcirc = Qcirc(name_dict=name_dict, gate_list=gate_list)
test_qcirc.loop_over_all_inputs_test()
