from unittest import BaseTestSuite
import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize

import numpy as np
import matplotlib.pyplot as plt


def test_pbc(plot = False):
    np.random.seed(0)

    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    BaTiO3 = CC.Phonons.Phonons("BaTiO3_")
    nat = BaTiO3.structure.N_atoms

    #for i in range(BaTiO3.structure.N_atoms):
    #    BaTiO3.effective_charges[i, :, :] = np.eye(3) * np.trace(BaTiO3.effective_charges[i, :, :]) / 3

    calculator = calc.LongRangeInteractions()
    calculator.init_from_dyn(BaTiO3)

    #calculator.dielectric_tensor[:,:] = np.eye(3)

    struct = BaTiO3.structure.copy()
    struct.coords += np.random.normal(size = struct.coords.shape, scale = 0.05)

    SUPERCELL = (2,2,2)
    SS2 = (3,3,3)
    N_STEPS = 12
    super_struct = struct.generate_supercell(SUPERCELL)
    ss2 = struct.generate_supercell(SS2)
    

    energies = np.zeros((N_STEPS,3), dtype = np.double)
    #forces = np.zeros(N_steps, dtype = np.double)
    cutoff = np.linspace(2, 30, N_STEPS)
    
    for i, cut in enumerate(cutoff):
        calculator.reset_cutoff(cut)

        s1 = struct.get_ase_atoms()
        s1.set_calculator(calculator)
        energies[i,0] = s1.get_total_energy()

        s1 = super_struct.get_ase_atoms()
        s1.set_calculator(calculator)
        energies[i,1] = s1.get_total_energy() / np.prod(SUPERCELL)

        
        s1 = ss2.get_ase_atoms()
        s1.set_calculator(calculator)
        energies[i,2] = s1.get_total_energy() / np.prod(SS2)


    if plot:
        plt.figure(dpi = 150)
        LBL_FS = 13
        plt.rcParams["font.family"] = 'Liberation Serif'
        plt.plot(cutoff, energies[:,0], label = 'unit cell', color = 'k', ls = 'solid')
        plt.plot(cutoff, energies[:,1], ls = 'dashed', color = 'red', label = '2x2x2 cell')
        plt.plot(cutoff, energies[:,2], ls = '-.', color = 'green', label = '3x3x3 cell')
        plt.legend()
        plt.xlabel('Cutoff [A]', fontsize = LBL_FS)
        plt.ylabel('Total energy [eV]', fontsize = LBL_FS)
        plt.tight_layout()
        plt.savefig('convergence_cutoff.png')
        plt.savefig('convergence_cutoff.pdf')
        plt.savefig('convergence_cutoff.eps')



if __name__ == "__main__":
    test_pbc(True)
    plt.show()
