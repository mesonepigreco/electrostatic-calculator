import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.ForceTensor
import ase, ase.dft

import numpy as np
import matplotlib.pyplot as plt

DYN_GOOD = "../compute_energy_forces/BaTiO3_"

DYN_222 = "dyn"
ELE_222 = "electrostatic_dyn/dyn_cc_2x2x2_"#"4x4x4_electrostatic/dyn""electrostatic_dyn/dyn"
NQIRR = 4

ELE_444 = "electrostatic_dyn/dyn_cc_4x4x4_"
NQIRR_BIG = 10


ELE_666 = None
#ELE_666 = "fourier_dyn/dyn_cc_6x6x6_"#"4x4x4_electrostatic/dyn"
#NQIRR_BIGG = 20

PATH= "GX"
N_POINTS = 1000



# -------- HERE THE CORE SCRIPT ------------
#dyn = CC.Phonons.Phonons(DYN_GOOD, NQIRR)
dyn = CC.Phonons.Phonons(DYN_222, NQIRR)


band_path = ase.dft.kpoints.bandpath(PATH,
                                     dyn.structure.unit_cell,
                                     N_POINTS)

# Get the q points of the path
q_path = band_path.cartesian_kpts()

# Get the values of x axis for plotting the band path
x_axis, xticks, xlabels = band_path.get_linear_kpoint_axis()


t2_noBEC =  CC.ForceTensor.Tensor2(dyn.structure,
                            dyn.structure.generate_supercell(dyn.GetSupercell()),
                            dyn.GetSupercell())
t2_noBEC.SetupFromPhonons(dyn)
t2_noBEC.Center(Far = 3)
t2_noBEC.Apply_ASR()

# Prepare the tensor with the born effective charges
dyn_bec = CC.Phonons.Phonons(DYN_GOOD, 1)
dyn.dielectric_tensor = dyn_bec.dielectric_tensor
dyn.effective_charges = dyn_bec.effective_charges
t2_BEC =  CC.ForceTensor.Tensor2(dyn.structure,
                                 dyn.structure.generate_supercell(dyn.GetSupercell()),
                                 dyn.GetSupercell())
t2_BEC.SetupFromPhonons(dyn)
t2_BEC.Center(Far = 3)
t2_BEC.Apply_ASR()


# Prepare the tensor with the born effective charges
dyn = CC.Phonons.Phonons(ELE_222, NQIRR)
t2_ELE =  CC.ForceTensor.Tensor2(dyn.structure,
                                 dyn.structure.generate_supercell(dyn.GetSupercell()),
                                 dyn.GetSupercell())
t2_ELE.SetupFromPhonons(dyn)
t2_ELE.Center(Far = 3)
t2_ELE.Apply_ASR()


# Prepare the tensor with the born effective charges
dyn = CC.Phonons.Phonons(ELE_444, NQIRR_BIG)
t2_ELEbig =  CC.ForceTensor.Tensor2(dyn.structure,
                                 dyn.structure.generate_supercell(dyn.GetSupercell()),
                                 dyn.GetSupercell())
t2_ELEbig.SetupFromPhonons(dyn)
t2_ELEbig.Center(Far = 3)
t2_ELEbig.Apply_ASR()



# Prepare the tensor with the born effective charges
if ELE_666 is not None:
    dyn = CC.Phonons.Phonons(ELE_666, NQIRR_BIGG)
    t2_ELEbigg =  CC.ForceTensor.Tensor2(dyn.structure,
                                     dyn.structure.generate_supercell(dyn.GetSupercell()),
                                     dyn.GetSupercell())
    t2_ELEbigg.SetupFromPhonons(dyn)
    t2_ELEbigg.Center(Far = 3)
    t2_ELEbigg.Apply_ASR()



# Now we need to perform the interpolation, dyagonalizing the dynamical matrix for each q point of the path
n_modes = 3 * dyn.structure.N_atoms
ws = np.zeros((N_POINTS, n_modes, 4), dtype = np.double)
m = dyn.structure.get_masses_array()
m = np.tile(m, (3,1)).T.ravel()

for i in range(N_POINTS):
    # For each point in the path

    # Interpoalte the dynamical matrix
    fc_nobec = t2_noBEC.Interpolate(-q_path[i, :])
    fc = fc_nobec.copy()
    
    # Mass rescale the force constant matrix
    dynq = fc / np.outer(np.sqrt(m), np.sqrt(m))

    # Diagonalize the dynamical matrix
    w2 = np.linalg.eigvalsh(dynq)
    ws[i, :,0] = np.sqrt(np.abs(w2)) * np.sign(w2) * CC.Units.RY_TO_CM
    
    # Interpoalte the dynamical matrix
    fc = t2_BEC.Interpolate(-q_path[i, :])
    # Mass rescale the force constant matrix
    dynq = fc / np.outer(np.sqrt(m), np.sqrt(m))

    # Diagonalize the dynamical matrix
    w2 = np.linalg.eigvalsh(dynq)
    ws[i, :,1] = np.sqrt(np.abs(w2)) * np.sign(w2) * CC.Units.RY_TO_CM


    # Produce the dynq with the sum
    fc_ele222 = t2_ELE.Interpolate(-q_path[i,:])
    fc_ele444 = t2_ELEbig.Interpolate(-q_path[i,:])

    if ELE_666 is not None:
        fc_ele666 = t2_ELEbigg.Interpolate(-q_path[i,:])
    
    fc = (fc_nobec + fc_ele222 - fc_ele444)
    # Mass rescale the force constant matrix
    dynq = fc / np.outer(np.sqrt(m), np.sqrt(m))

    # Diagonalize the dynamical matrix
    w2 = np.linalg.eigvalsh(dynq)
    ws[i, :,2] = np.sqrt(np.abs(w2)) * np.sign(w2) * CC.Units.RY_TO_CM


    if ELE_666 is not None:
        fc = (fc_nobec - fc_ele222 + fc_ele666)
        # Mass rescale the force constant matrix
        dynq = fc / np.outer(np.sqrt(m), np.sqrt(m))

        # Diagonalize the dynamical matrix
        w2 = np.linalg.eigvalsh(dynq)
        ws[i, :,3] = np.sqrt(np.abs(w2)) * np.sign(w2) * CC.Units.RY_TO_CM


    
# ============= PLOT THE FIGURE =================
fig = plt.figure(dpi = 200)
ax = plt.gca()

# Plot all the modes
for i in range(n_modes):
    label1 = None
    label2 = None
    label3 = None
    label4 = None
    if i ==0 :
        label1 = "NAC"
        label2 = "no NAC"
        label3 = "4x4x4 NAC"
        label4 = "6x6x6 NAC"
        
    ax.plot(x_axis, ws[:,i,1], color = "k", label = label1)
    ax.plot(x_axis, ws[:,i,0], color = "r", ls = "dotted", label=label2)
    ax.plot(x_axis, ws[:,i,2], color = "b", ls = "dashed", label=label3)

    if ELE_666 is not None:
        ax.plot(x_axis, ws[:,i,3], color = "g", ls = "-.", label=label4)


# Plot vertical lines for each high symmetry points
for x in xticks:
    ax.axvline(x, 0, 1, color = "k", lw = 0.4, ls = ":")

# Set the x labels to the high symmetry points
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

ax.set_ylabel("Energy [cm-1]")
ax.set_xlabel("q path")
ax.legend()

fig.tight_layout()
fig.savefig("dispersion.png")
fig.savefig("dispersion.eps")
plt.show()
