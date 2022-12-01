import cellconstructor as CC, cellconstructor.Phonons, cellconstructor.ForceTensor
import sys, os
import numpy as np
import pyelectrostatic, pyelectrostatic.calculator as calculator



def test_espresso():
    calc = calculator.ElectrostaticCalculator()
    calc.eta = 8
    dyn  = CC.Phonons.Phonons("BaTiO3_")
    calc.init_from_phonons(dyn)

    # Setup the dynamical matrix
    t2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure, dyn.GetSupercell())
    t2.SetupFromPhonons(dyn)

    q_direction = np.array([1,0,0], dtype = np.double)
    epsil = 1e-6
    q_vector = q_direction * epsil

    dynq_espresso = t2.Interpolate(np.array([0,0,0], dtype = np.double), q_direct = q_direction)
    dynq_electrostatic = calc.get_longrange_phonons(q_vector, dyn.structure)

    dist1 = np.linalg.norm(dynq_espresso - dynq_electrostatic)
    dist2 = np.linalg.norm(dynq_espresso - np.conj(dynq_electrostatic))

    assert dist1 < 1e-5 and dist2 < 1e-5, "Distance between dynamical matrices: {} {}".format(dist1, dist2)


if __name__ == "__main__":
    test_espresso()

