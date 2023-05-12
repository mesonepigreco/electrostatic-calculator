from unittest import BaseTestSuite
import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize
import pytest

import numpy as np
import matplotlib.pyplot as plt



def test_stress(plot = False, nat = 5):
    s = CC.Structure.Structure(nat)
    s.unit_cell = np.zeros((3,3), dtype = np.double)
    s.unit_cell[0,0] = 1
    s.unit_cell[1,1] = 1
    s.unit_cell[2,2] = 1
    s.has_unit_cell = True

    for i in range(nat):
        s.coords[i, :] = np.random.uniform(size = 3)

    effective_charges = np.zeros((nat,3,3), dtype = np.double)
    dielectric_tensor = np.eye(3)

    for i in range(nat):
        effective_charges[i, :, :] = np.eye(3) 
        

    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 0.1
    calculator.cutoff = 20
    calculator.init(s.copy(), effective_charges, dielectric_tensor, unique_atom_element="H")    
    calculator.kpoints = calculator.kpoints[:1, :]
    print("kpts:", calculator.kpoints)

    s.coords += np.random.normal(size = s.coords.shape, scale = 0.1) # This is the culprit

    n_steps = 20
    dstep = 0.001
    coordinate = 0

    energy = np.zeros(n_steps, dtype = np.double)
    volumes = np.zeros(n_steps, dtype = np.double)
    stress = np.zeros_like(energy)

    original_uc = s.unit_cell.copy()

    for i in range(n_steps):
        # Enlarge the volume
        s.change_unit_cell(original_uc *(1.0 + dstep * i))

        print("Original cell:", original_uc)
        print("Cell:", s.unit_cell)

        atm = s.get_ase_atoms()
        atm.set_calculator(calculator)
        energy[i] = atm.get_total_energy()
        stress[i] = np.trace(atm.get_stress(voigt=False)) / 3
        volumes[i] = s.get_volume()

    if plot:
        plt.figure()
        plt.title("{} particles".format(nat))
        plt.plot(volumes, -np.gradient(energy, volumes), label = "Numerical diff")
        plt.plot(volumes, stress, label = "Forces")
        plt.legend()
        plt.tight_layout()

    # TODO: assert the derivative


if __name__ == "__main__":
    test_stress(plot=True)
    plt.show()