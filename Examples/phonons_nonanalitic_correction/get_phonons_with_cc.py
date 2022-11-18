import cellconstructor as CC, cellconstructor.Phonons
import pyelectrostatic, pyelectrostatic.calculator as calculator

import sys, os


# Now create the calculator
SCELL = 2

if len(sys.argv) == 2:
    SCELL = int(sys.argv[1])

supercell = (SCELL, SCELL, SCELL)

calc = calculator.ElectrostaticCalculator()
dyn = CC.Phonons.Phonons("../compute_energy_forces/BaTiO3_")
calc.init(dyn.structure, dyn.effective_charges, dyn.dielectric_tensor, supercell)

final_dyn_name = os.path.join("electrostatic_dyn", "dyn_cc_{}x{}x{}_".format(SCELL, SCELL, SCELL))
#calc.cutoff = 30
#calc.eta = 5
#calc.debug = False

final_dyn = CC.Phonons.compute_phonons_finite_displacements(dyn.structure, calc, progress_bar = True, progress = 1, supercell = supercell)

final_dyn.Symmetrize()
final_dyn.save_qe(final_dyn_name)

