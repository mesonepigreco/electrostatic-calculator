from unittest import BaseTestSuite
import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize
import time

import numpy as np
import matplotlib.pyplot as plt

def test_stress(nat = 5, supercell = (3,3,3), verbose=True):
    s = CC.Structure.Structure(nat)
    s.unit_cell = np.eye(3) * 5
    s.has_unit_cell = True
    s.atoms[0] = "Si"
    s.atoms[1] = "C"
    s.atoms[2] = "H"
    s.atoms[3] = "N"
    s.atoms[4] = "O"

    for i in range(nat):
        s.coords[i, :] = np.random.uniform(size = 3)

    effective_charges = np.zeros((nat,3,3), dtype = np.double)
    dielectric_tensor = np.eye(3)

    for i in range(nat):
        effective_charges[i, :, :] = np.eye(3) 
        

    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 3.0
    calculator.cutoff = 3.0

    #calculator.init(s.copy(), effective_charges, dielectric_tensor, 
    #                unique_atom_element="H")    
    #calculator.kpoints = calculator.kpoints[:1, :]
    #print("kpts:", calculator.kpoints)

    # new_s = s.copy()

    # Compute the pressure numerically and using ASE, and compare them

    # Use the supercell
    calculator.init(s, effective_charges, dielectric_tensor,
                    unique_atom_element="H",
                    supercell = supercell)

    new_s = s.generate_supercell(supercell)
    new_s.coords[0, 0] += 0.1
    new_s.coords[0, 1] += 0.05
    new_s.coords[0, 2] -= 0.02
    new_s.coords[1, 0] -= 0.02

    atm = new_s.get_ase_atoms()
    atm.set_calculator(calculator)
    print("N kpoints: ", np.shape(calculator.kpoints))

    # Now compare the numerical stress from ase
    # With the one of the calculator
    
    energy = atm.get_potential_energy()
    if verbose:
        print("Energy: ", energy)

    stress = atm.get_stress()
    if verbose:
        print("Stress: ", stress)

    # Compute the numerical stress
    eps_voigt = np.zeros(6, dtype = np.double)
    delta_eps = np.zeros(6, dtype = np.double)
    delta_value = 1e-4
    for i in range(6):
        eps_voigt[:] = 0
        eps_voigt[i] = delta_value

        #if i > 2:
        #    eps_voigt[i] *= 2

        s_strained = new_s.strain(eps_voigt, voigt = True)
        atm_strain = s_strained.get_ase_atoms()
        atm_strain.set_calculator(calculator)
        delta_eps[i] = (atm_strain.get_potential_energy() - energy) / delta_value

    numerical_stress = delta_eps / new_s.get_volume()

    if verbose:
        print("Numerical stress: ", numerical_stress)


    assert np.allclose(stress, numerical_stress, atol = 1e-7), """
Error in the stress calculation.
Stress from automatic differentiation: {}
Numerical stress: {}
""".format(stress, numerical_stress)


    
    

if __name__ == "__main__":
    test_stress()
    
    

