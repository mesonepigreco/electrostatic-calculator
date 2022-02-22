from unittest import BaseTestSuite
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

    for i in range(BaTiO3.structure.N_atoms):
        BaTiO3.effective_charges[i, :, :] = np.eye(3) * np.trace(BaTiO3.effective_charges[i, :, :]) / 3

    calculator = calc.LongRangeInteractions()
    calculator.init_from_dyn(BaTiO3)
    calculator.dielectric_tensor[:,:] = np.eye(3)

    struct = BaTiO3.structure.copy()
    struct.coords += np.random.normal(size = struct.coords.shape, scale = 0.05)

    # Move atom
    atm_id = 2
    direction = 0
    delta = 0.01
    N_steps  = 50

    energies = np.zeros(N_steps, dtype = np.double)
    electric_field = np.zeros((N_steps, 3), dtype = np.double)
    de_dr = np.zeros((N_steps, 3), dtype = np.double)
    forces = np.zeros(N_steps, dtype = np.double)

    xvalues = np.zeros(N_steps, dtype = np.double)
    ss = []
    s_charges = []
    for i in range(N_steps):    
        struct.coords[atm_id, direction] += delta
        xvalues[i] = struct.coords[atm_id, direction]
        
        atm = struct.get_ase_atoms()
        ss.append(atm)
        atm.set_calculator(calculator)

        energies[i] = atm.get_total_energy()
        forces[i] = atm.get_forces()[atm_id, direction]

        electric_field[i,:] = calculator.get_electric_field(np.zeros(3))
        de_dr[i, :] = calculator.get_derivative_efield(np.zeros(3))[atm_id, direction, :]
        print("CHARGES:")
        print(calculator.charges)
        print(calculator.charge_coords)

    if plot:
        plt.plot(xvalues, energies, label = "Energy")
        plt.figure()
        plt.plot(xvalues, -np.gradient(energies, delta), label = "Numerical diff")
        plt.plot(xvalues, forces, label = "Forces")
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, -np.gradient(electric_field[:,0], delta), label = "Numerical efield diff")
        plt.plot(xvalues, de_dr[:, 0], label = "Anal efield diff") 
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, electric_field[:,0], label = "E field")
        plt.legend()
        plt.tight_layout()
        #ase.visualize.view(ss)


if __name__ == "__main__":
    test_unit_cell(plot = True)
    plt.show()
