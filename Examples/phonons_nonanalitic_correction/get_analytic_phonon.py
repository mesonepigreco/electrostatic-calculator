import cellconstructor as CC, cellconstructor.Phonons
import pyelectrostatic, pyelectrostatic.calculator as calculator

import sys, os
import numpy as np

DYNNAME = "force_expression_30_11_2022/dyn_cc_3x3x3_"

if len(sys.argv) == 2:
    DYNNAME = int(sys.argv[1])

OUTNAME = DYNNAME + "analytical_"
OUTNAME2 = DYNNAME + "fcsuper_"

nqirr = len([x for x in os.listdir(os.path.dirname(DYNNAME)) if x.startswith(os.path.basename(DYNNAME)) and not x == os.path.basename(DYNNAME) + "0" and x.replace(os.path.basename(DYNNAME), "").isnumeric()]) 

calc = calculator.ElectrostaticCalculator()
calc.eta = 8
dyn = CC.Phonons.Phonons(DYNNAME, nqirr)
dyn_gamma  = CC.Phonons.Phonons("../compute_energy_forces/BaTiO3_")
calc.init_from_phonons(dyn_gamma)

print("k-points: ", len(calc.kpoints))

for iq, q in enumerate(dyn.q_tot):
    #myq = q * 2 * np.pi / CC.Units.A_TO_BOHR
    dyn.dynmats[iq] = calc.get_longrange_phonons(q, dyn.structure)

dyn.save_qe(OUTNAME)
dyn.effective_charges = dyn_gamma.effective_charges
dyn.dielectric_tensor = dyn_gamma.dielectric_tensor
calc.init_from_phonons(dyn)

super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
fc_total = calc.get_supercell_fc(super_struct)
dynq = CC.Phonons.GetDynQFromFCSupercell(fc_total, np.array(dyn.q_tot), dyn.structure, super_struct)

for iq, q in enumerate(dyn.q_tot):
    dyn.dynmats[iq] = dynq[iq, :, :]

dyn.save_qe(OUTNAME2)


