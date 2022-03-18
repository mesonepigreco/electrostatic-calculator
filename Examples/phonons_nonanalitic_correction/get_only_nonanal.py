"""
Get the nonanalitical contribution from the effective charges and compare with the one given by the electrostatic term.
"""

import numpy as np
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.ForceTensor

DYN_TOTAL = '../compute_energy_forces/BaTiO3_'
dyn_0 = CC.Phonons.Phonons(DYN_TOTAL, 1)

dyn1 = CC.Phonons.Phonons('electrostatic_gamma/dyn_cc_4x4x4_', 10)

# Set to zero the dynamical matrix
for i, q in enumerate(dyn_0.q_tot):
    dyn_0.dynmats[i][:,:] = 0


# Prepare here the interpolated dynamical matrix
t2 = CC.ForceTensor.Tensor2(dyn_0.structure, dyn_0.structure.generate_supercell(dyn_0.GetSupercell()), dyn_0.GetSupercell())
t2.SetupFromPhonons(dyn_0)


for iq, q in enumerate(dyn1.q_tot):
    dyn1.dynmats[iq] = t2.Interpolate(-q)

dyn1.save_qe('only_nonanal/dyn_only_effective_charges')

