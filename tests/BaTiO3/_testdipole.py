from unittest import BaseTestSuite
import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize

import numpy as np
import matplotlib.pyplot as plt


def test_dipole(plot = False):
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


    ATM_ID = 4
    COORD = 1
    NX = 10
    
    _x_ = np.linspace(0, 1, NX)
    dipole_zeff = np.zeros((NX, 3))
    dipole_charge = np.zeros((NX, 3))
    
    
    for i,x in enumerate(_x_):
        struct = BaTiO3.structure.copy()
        struct.coords[ATM_ID, :] += x

        atms = struct.get_ase_atoms()
        
        calculator.dipole_from_effective_charge = True
        calculator.calculate(atms)
        dipole_zeff[i, :] = calculator.results["dipole"]

        calculator.dipole_from_effective_charge = False
        calculator.calculate(atms)
        dipole_charge[i, :] = calculator.results["dipole"]


    
    if plot:
        plt.plot(_x_, dipole_zeff[:, 0], color = "r", ls = "dashed", label = "DZx", marker ="o", markersize = 8)
        plt.plot(_x_, dipole_zeff[:, 1], color = "r", ls = "dotted", label = "DZy", marker ="o", markersize = 8)
        plt.plot(_x_, dipole_zeff[:, 2], color = "r", ls = "-.", label = "DZz", marker ="o", markersize = 8)

        
        plt.plot(_x_, dipole_charge[:, 0], color = "darkblue", ls = "dashed", label = "DCx", marker ="s", markersize = 4)
        plt.plot(_x_, dipole_charge[:, 1], color = "darkblue", ls = "dotted", label = "DCy", marker ="s", markersize = 4)
        plt.plot(_x_, dipole_charge[:, 2], color = "darkblue", ls = "-.", label = "DCz", marker ="s", markersize = 4)

        plt.legend()
        plt.tight_layout()
        plt.show()
        
    assert np.max(np.abs(dipole_charge - dipole_zeff)) < 1e-8

if __name__ == "__main__":    
    test_dipole(plot = True)
