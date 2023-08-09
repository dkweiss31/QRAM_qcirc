from QRAM_utils.utils import extract_info_from_h5
from Ramsey.ramsey_Yao import RamseyExperiment

filepath = "/Users/danielweiss/PycharmProjects/QRAM_qcirc/Ramsey/out/00000_cohere_Ramsey_cav_5_interfer_True.hdf5"
filepath = "/Users/danielweiss/PycharmProjects/QRAM_qcirc/Ramsey/out/00000_Ramsey_cav_8_interfer_True.hdf5"
data_dict, param_dict = extract_info_from_h5(filepath)
ramsey_result = data_dict["ramsey_result"]
# param_dict.pop("truncated_dims")
# ramsey_inst = CoherentDephasing(**param_dict)
# p0 = (6 * 10 ** 4, 0.045, -0.5, -0.2, -2)
ramsey_inst = RamseyExperiment(**param_dict)
p0 = (6 * 10 ** 4, 0.045, 0.5, 0.2, -1)
gamma_phi_indep, popt, pcov = ramsey_inst.extract_gammaphi(ramsey_result, p0=p0)
naive_gamma_phi = sum(
    ramsey_inst.gamma_phi_full_func()
)
print(f"naive gamma_phi = {naive_gamma_phi}")
print(f"indep gamma_phi = {gamma_phi_indep}")