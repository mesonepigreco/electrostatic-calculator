import cellconstructor as CC, cellconstructor.Phonons
import pyelectrostatic, pyelectrostatic.fourier_calculator as calculator

# Now create the calculator
SCELL = 2
MODE = "remove"

calc = calculator.FourierCalculator(mode = MODE)
dyn = CC.Phonons.Phonons("../../compute_energy_forces/BaTiO3_")
calc.init_from_dyn(dyn)


#calc.cutoff = 30
#calc.eta = 5
#calc.debug = False

final_dyn = CC.Phonons.compute_phonons_finite_displacements(dyn.structure, calc, progress_bar = True, progress = 1, supercell = (SCELL, SCELL, SCELL))

final_dyn.Symmetrize()
final_dyn.save_qe("dyn_cc_{}x{}x{}_".format(SCELL, SCELL, SCELL))

