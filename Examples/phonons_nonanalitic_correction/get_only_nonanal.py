"""
Get the nonanalitical contribution from the effective charges and compare with the one given by the electrostatic term.
"""

import numpy as np
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.ForceTensor
import sys, os

DIR = "only_nonanal"

if not os.path.exists(DIR):
    os.makedirs(DIR)

DYN_TOTAL = '../compute_energy_forces/BaTiO3_'
dyn_0 = CC.Phonons.Phonons(DYN_TOTAL, 1)
dyn1 = CC.Phonons.Phonons('dyn', 4)

# Set to zero the dynamical matrix
for i, q in enumerate(dyn1.q_tot):
    dyn1.dynmats[i][:,:] = 0

dyn1.Symmetrize()



# Prepare here the interpolated dynamical matrix
t2 = CC.ForceTensor.Tensor2(dyn1.structure, dyn1.structure.generate_supercell(dyn1.GetSupercell()), dyn1.GetSupercell())
t2.SetupFromPhonons(dyn1)
#t2.Center(Far = 3)
#t2.Apply_ASR()

for iq, q in enumerate(dyn1.q_tot):
    dyn1.dynmats[iq] = t2.Interpolate(-q)

dyn1.save_qe(os.path.join(DIR, "nonac"))


dyn1.effective_charges = dyn_0.effective_charges
dyn1.dielectric_tensor = dyn_0.dielectric_tensor

# Prepare here the interpolated dynamical matrix
t2 = CC.ForceTensor.Tensor2(dyn1.structure, dyn1.structure.generate_supercell(dyn1.GetSupercell()), dyn1.GetSupercell())
t2.SetupFromPhonons(dyn1)
#t2.Center(Far = 3)
#t2.Apply_ASR()

for iq, q in enumerate(dyn1.q_tot):
    dyn1.dynmats[iq] = t2.Interpolate(-q)

dyn1.save_qe(os.path.join(DIR, "nac"))
t2.GeneratePhonons(dyn1.GetSupercell()).save_qe(os.path.join(DIR, "nacc"))





ef = t2.effective_charges
t2.effective_charges = None
for iq, q in enumerate(dyn1.q_tot):
    dyn1.dynmats[iq] = t2.Interpolate(-q)

dyn1.save_qe(os.path.join(DIR, "remove_mat"))


t2.tensor[:,:,:] = 0
new_dyn = t2.GeneratePhonons(dyn1.GetSupercell())
new_dyn.save_qe(os.path.join(DIR, "zero_mat"))

t2.effective_charges = ef
new_dyn = t2.GeneratePhonons(dyn1.GetSupercell())
new_dyn.save_qe(os.path.join(DIR, "add_mat"))
