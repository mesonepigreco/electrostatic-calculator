from unittest import BaseTestSuite
import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize
import time

import numpy as np
import matplotlib.pyplot as plt

def test_supercell(nat = 5, supercell = (3,3,3)):
    s = CC.Structure.Structure(nat)
    s.unit_cell = np.eye(3) * 5
    s.has_unit_cell = True

    for i in range(nat):
        s.coords[i, :] = np.random.uniform(size = 3)

    effective_charges = np.zeros((nat,3,3), dtype = np.double)
    dielectric_tensor = np.eye(3)

    for i in range(nat):
        effective_charges[i, :, :] = np.eye(3) 
        

    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 2
    calculator.init(s.copy(), effective_charges, dielectric_tensor, 
                    unique_atom_element="H")    
    #calculator.kpoints = calculator.kpoints[:1, :]
    #print("kpts:", calculator.kpoints)

    new_s = s.copy()
    new_s.coords[0, 0] += 0.15

    atm = new_s.get_ase_atoms()
    atm.set_calculator(calculator)

    print("Original coordinates")
    print(s.coords)
    print("New coordinates")
    print(new_s.coords)
    print()
    energy_primitive_cell = atm.get_total_energy()
    sys.stdout.flush()


    # Use the supercell
    calculator.init(s, effective_charges, dielectric_tensor,
                    unique_atom_element="H",
                    supercell = supercell)

    new_s = new_s.generate_supercell(supercell)
    atm = new_s.get_ase_atoms()
    atm.set_calculator(calculator)
    print("N kpoints: ", np.shape(calculator.kpoints))
    energy_supercell = atm.get_total_energy() / np.prod(supercell)

    print(energy_supercell)
    print(energy_primitive_cell)
    
    assert abs(energy_primitive_cell -energy_supercell) < 1e-10, """
Error, the energy of the primitive cell is {}
       the energy of the supercell (rescaled) is {}
       They should match.
""".format(energy_primitive_cell, energy_supercell)


if __name__ == "__main__":
    test_supercell()
    
    

