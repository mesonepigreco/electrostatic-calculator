"""
This simple tutorial shows how to use pyelectrostatic to compute
the forces on a structure.
Here we will use ASE to load the structure, 
and cellconstructor to load the espresso dynamical matrix file to initialize the electrostatic model
(only dielectric tensor, centroid structure and effective charges are used)

First of all, import all the packages
"""

# Import the LongRangeInteractions ASE calculator from py-electrostatic
import pyelectrostatic, pyelectrostatic.calculator as calculator

# Import CellConstructor as ASE
import ase, ase.io
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Methods

# Misc import for math and plotting and timing
import numpy as np
import matplotlib.pyplot as plt
import time


""" 
Let's initialize the electrostatic calculator by loading the espresso dynamical matrix
"""

dyn = CC.Phonons.Phonons("BaTiO3_")
calculator = calculator.ElectrostaticCalculator()
supercell = (3,3,3)  # The size of the supercell on which the forces are computed.

# Set the cutoff for the short range forces
"""
The eta parameter is the size of each dipole. Long range forces are zero within this range and starts acting outside the eta range.

The cutoff is an adimensional parameter for the Ewald summation: 
k-points bigger than cutoff/eta are discarded. 
Note that there is a prefactor exp( 1/2 (k * eta)^2 ), 
therefore values of the cutoff between 3 and 5 should give converged results.
"""

calculator.eta = 6 
calculator.cutoff = 5  


"""
Now, we need to initialize the calculator (computing the k-points allowed for the Ewald summation) and setting the reference structure, effective charges, dielectric tensor and the supercell.
Luckily, these infrmation are all stored inside the dynamical matrix of quantum espresso format.
"""
calculator.init(dyn.structure, dyn.effective_charges, dyn.dielectric_tensor, supercell)



"""
Now we load the structure on which we want to compute energies and forces.
It is supercell.pwo, written as a quantum-espresso input file.
This is an ASE Atoms object
"""

structure = ase.io.read("supercell.pwo")

"""
We retrive the real energies and forces from the ASE calculator
"""

print("The total DFT energy is: {} eV".format(structure.get_total_energy()))
dft_forces = structure.get_forces()


"""
We assign the electrostatic calculator to the structure
"""

structure.set_calculator(calculator)


"""
Now we can compute the energy and forces!
"""
t1 = time.time()
print("The total electrostatic energy is: {} eV".format(structure.get_total_energy()))
elec_forces = structure.get_forces()
t2 = time.time()
print("Total time to compute the electrostatic energy and forces: {} s".format(t2-t1))

print("The total electrostatic forces are:  [eV/A]")
print(elec_forces)



##################################################################################

# """
# That is it!
# We now plot on screen the dependency of forces with the distance from the displaced atom.
# In this case I displaced the Ti atom in the origin.
# """

# # Lets convert the Ase atoms into cellconstructor
# # That enables calculations of distances with periodic boundary conditions
# struct = CC.Structure.Structure()
# struct.generate_from_ase_atoms(structure)

# # Prepare the array to be filled with the modulus of the forces
# distance = np.zeros(struct.N_atoms)
# force_modulus = np.zeros(struct.N_atoms)
# diff_modulus = np.zeros(struct.N_atoms)
# electric_modulus = np.zeros(struct.N_atoms)

# for i in range(len(structure)):
#     # Compute the distance between the atoms (accounting for the periodic boundary conditions)
#     distance[i] = np.linalg.norm(CC.Methods.get_closest_vector(struct.unit_cell, struct.coords[0,:]- struct.coords[i,:]))

#     force_modulus[i] = np.linalg.norm(dft_forces[i, :])
#     diff_modulus[i] = np.linalg.norm(dft_forces[i,:] - elec_forces[i,:])
#     electric_modulus[i] = np.linalg.norm(elec_forces[i,:])

# # Sort the array for the distances
# sort_mask = np.argsort(distance)
# distance = distance[sort_mask]
# force_modulus = force_modulus[sort_mask]
# diff_modulus = diff_modulus[sort_mask]
# electric_modulus = electric_modulus[sort_mask]

# # Measure the average cutoff distance for the forces
# dft_cutoff = np.sum(force_modulus * distance) #/ np.sum(force_modulus)
# diff_cutoff = np.sum(diff_modulus * distance) #/ np.sum(diff_modulus)
# electric_cutoff = np.sum(electric_modulus * distance) #/ np.sum(electric_modulus)

# print()
# print("The average distances on forces intensities are:")
# print("DFT:        {:10.4f} A".format(dft_cutoff))
# print("Electric:   {:10.4f} A".format(electric_cutoff))
# print("Difference: {:10.4f} A".format(diff_cutoff))
# print()

# np.savetxt("dist_file-{}.dat".format(calculator.cutoff),
#            np.transpose([distance, force_modulus, electric_modulus, diff_modulus]),
#            header = "Distance [A]; DFT force [eV/A]; Electrostatic force [eV/A]; Differnce [eV/A]")

# # Now plot the results
# plt.figure(dpi = 150)
# LBL_FS = 12
# plt.rcParams["font.family"] = "Liberation Serif"
# plt.plot(distance, force_modulus, lw=0, marker = "D", ls = "dotted", label = "DFT forces")
# plt.plot(distance, diff_modulus, lw=0, marker = "*", ls= "dashed", label = "DFT - long range")
# plt.plot(distance, electric_modulus, lw=0,marker = "+", ls= "-.", label = "long range")
# plt.xlabel("Distance [$\AA$]", fontsize = LBL_FS)
# plt.ylabel("Force modulus [eV/A]", fontsize = LBL_FS)
# plt.legend()
# plt.tight_layout()


# # Now plot the results
# #fig, axarr = plt.subplots(nrows = 3, ncols = 1, dpi = 150, figsize = (6,10))
# #plt.plot
# #plt.plot(distance, force_modulus, marker = "D", ls = "dotted", label = "DFT forces")
# #plt.plot(distance, diff_modulus, marker = "*", ls= "dashed", label = "DFT - long range")
# #plt.plot(distance, electric_modulus, marker = "+", ls= "-.", label = "long range")
# #plt.xlabel("Distance [$\AA$]", fontsize = LBL_FS)
# #plt.ylabel("Force modulus [eV/A]", fontsize = LBL_FS)
# #plt.legend()
# #plt.tight_layout()





# plt.show()
    












