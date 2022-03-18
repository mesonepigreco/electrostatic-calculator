import cellconstructor as CC, cellconstructor.Phonons
import pyelectrostatic, pyelectrostatic.fourier_calculator as calculator

import sys, os


# Now create the calculator
SCELL = 2

if len(sys.argv) == 2:
    SCELL = int(sys.argv[1])

calc = calculator.FourierCalculator()
dyn = CC.Phonons.Phonons("../compute_energy_forces/BaTiO3_")
calc.init_from_dyn(dyn)

final_dyn_name = os.path.join("fourier_dyn", "dyn_cc_{}x{}x{}_".format(SCELL, SCELL, SCELL))
#calc.cutoff = 30
#calc.eta = 5
#calc.debug = False

final_dyn = CC.Phonons.compute_phonons_finite_displacements(dyn.structure, calc, progress_bar = True, progress = 1, supercell = (SCELL, SCELL, SCELL))

final_dyn.Symmetrize()
final_dyn.save_qe(final_dyn_name)

