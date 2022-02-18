import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize

import numpy as np
import matplotlib.pyplot as plt

def test_unit_cell(plot = False):
    """
    Simple test to check if forces are derivatives of the energy
    """

    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    

    BaTiO3 = CC.Phonons.Phonons("BaTiO3_")

    calculator = calc.LongRangeInteractions()
    calculator.init_from_dyn(BaTiO3)

    struct = BaTiO3.structure.copy()

    # Move atom
    atm_id = 2
    direction = 0
    delta = 0.05
    N_steps  = 50

    energies = np.zeros(N_steps, dtype = np.double)
    forces = np.zeros(N_steps, dtype = np.double)

    xvalues = np.zeros(N_steps, dtype = np.double)
    ss = []
    for i in range(N_steps):    
        struct.coords[atm_id, direction] += delta
        xvalues[i] = struct.coords[atm_id, direction]
        
        atm = struct.get_ase_atoms()
        ss.append(atm)
        atm.set_calculator(calculator)
        energies[i] = atm.get_total_energy()
        forces[i] = atm.get_forces()[atm_id, direction]

    if plot:
        plt.plot(xvalues, -np.gradient(energies, delta), label = "Numerical diff")
        plt.plot(xvalues, forces, label = "Forces")
        plt.legend()
        plt.tight_layout()
        ase.visualize.view(ss)


if __name__ == "__main__":
    test_unit_cell(plot = True)
    plt.show()