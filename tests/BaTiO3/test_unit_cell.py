from unittest import BaseTestSuite
import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize
import pytest

import numpy as np
import matplotlib.pyplot as plt


def test_1D(plot = False):
    s = CC.Structure.Structure(1)
    s.unit_cell = np.eye(3)
    s.has_unit_cell = True

    effective_charges = np.zeros((1,3,3), dtype = np.double)
    dielectric_tensor = np.eye(3)

    effective_charges[0, :, :] = np.eye(3)


    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 4
    calculator.init(s.copy(), effective_charges, dielectric_tensor)    
    print("kpts:", calculator.kpoints)

    n_steps = 20
    dstep = 0.1
    coordinate = 0

    energy = np.zeros(n_steps, dtype = np.double)
    forces = np.zeros((n_steps, 1, 3), dtype = np.double)
    x_values = np.zeros_like(energy)

    for i in range(n_steps):
        s.coords[0,coordinate] += dstep
        atm = s.get_ase_atoms()
        atm.set_calculator(calculator)
        energy[i] = atm.get_total_energy()
        forces[i, :, :] = atm.get_forces()
        x_values[i] = s.coords[0,coordinate]

    if plot:
        plt.figure()
        plt.title("1 particle")
        plt.plot(x_values, -np.gradient(energy, x_values), label = "Numerical diff")
        plt.plot(x_values, forces[:, 0,coordinate], label = "Forces")
        plt.legend()
        plt.tight_layout()

def test_2D(plot = False, nat = 5):
    s = CC.Structure.Structure(nat)
    s.unit_cell = np.eye(3)
    s.has_unit_cell = True
    s.atoms = ["H"]*nat
    s.atoms[1] = "O"
    s.atoms[2] = "C"
    s.atoms[3] = "N"
    s.atoms[4] = "S"

    for i in range(nat):
        s.coords[i, :] = np.random.uniform(size = 3)

    effective_charges = np.zeros((nat,3,3), dtype = np.double)
    dielectric_tensor = np.eye(3)

    for i in range(nat):
        effective_charges[i, :, :] = np.eye(3) 
        

    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 0.1
    calculator.cutoff = 20
    calculator.init(s.copy(), effective_charges, dielectric_tensor)    
    calculator.kpoints = calculator.kpoints[:1, :]
    print("kpts:", calculator.kpoints)

    s.coords += np.random.normal(size = s.coords.shape, scale = 0.1) # This is the culprit

    n_steps = 20
    dstep = 0.001
    coordinate = 0

    energy = np.zeros(n_steps, dtype = np.double)
    forces = np.zeros((n_steps, nat, 3), dtype = np.double)
    x_values = np.zeros_like(energy)

    for i in range(n_steps):
        print()
        print()
        print("coordinates:")
        print(s.coords)
        s.coords[0,coordinate] += dstep
        atm = s.get_ase_atoms()
        atm.set_calculator(calculator)
        energy[i] = atm.get_total_energy()
        forces[i, :, :] = atm.get_forces()
        x_values[i] = s.coords[0,coordinate]
        print()

    assert_test = -np.gradient(energy, x_values)
    assert_test = assert_test[1:-1]
    assert_test2 = forces[:, 0,coordinate]
    assert_test2 = assert_test2[1:-1]
    tolval = np.max(np.abs(assert_test))/50

    assert np.allclose(assert_test, assert_test2, atol = tolval), "Forces are not correct"

    if plot:
        plt.figure()
        plt.title("{} particles".format(nat))
        plt.plot(x_values, -np.gradient(energy, x_values), label = "Numerical diff")
        plt.plot(x_values, forces[:, 0,coordinate], label = "Forces")
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.title("{} particles".format(nat))
        plt.plot(x_values[1:-1], (-np.gradient(energy, x_values) - forces[:, 0,coordinate])[1:-1], label = "Displacements")


def test_julia_calculator(nat = 5):
    s = CC.Structure.Structure(nat)
    s.unit_cell = np.eye(3)
    s.has_unit_cell = True
    s.atoms[0] = "H"
    s.atoms[1] = "O"
    s.atoms[2] = "C"
    s.atoms[3] = "N"
    s.atoms[4] = "S"

    for i in range(nat):
        s.coords[i, :] = np.random.uniform(size = 3)

    effective_charges = np.zeros((nat,3,3), dtype = np.double)
    dielectric_tensor = np.eye(3)

    for i in range(nat):
        effective_charges[i, :, :] = np.eye(3) 
        

    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 0.2
    calculator.cutoff = 10
    calculator.init(s.copy(), effective_charges, dielectric_tensor)    
    old_s = s.copy()

    s.coords += np.random.normal(size = s.coords.shape, scale = 0.03) # This is the culprit

    atm = s.get_ase_atoms()
    atm.set_calculator(calculator)
    energy = atm.get_total_energy()
    force = atm.get_forces()


    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 0.2
    calculator.cutoff = 10
    calculator.init(old_s.copy(), effective_charges, dielectric_tensor)    
    calculator.julia_speedup = False
    atm.set_calculator(calculator)
    new_energy = atm.get_total_energy()
    new_force = atm.get_forces()

    assert abs(energy - new_energy) < 1e-7, "Julia = {}, Python = {}".format(energy, new_energy)
    assert np.max(abs(force - new_force)) < 1e-7, "Julia = {}, Python = {}".format(force, new_force)

def test_total_translation(plot = False):
    np.random.seed(0)

    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    BaTiO3 = CC.Phonons.Phonons("BaTiO3_")
    BaTiO3.Symmetrize()
    nat = BaTiO3.structure.N_atoms

    #for i in range(BaTiO3.structure.N_atoms):
    #    BaTiO3.effective_charges[i, :, :] = np.eye(3) #* np.trace(BaTiO3.effective_charges[i, :, :]) / 3
    #BaTiO3.dielectric_tensor[:,:] = np.eye(3)

    calculator = calc.ElectrostaticCalculator()
    calculator.eta = 6
    calculator.cutoff = 5
    calculator.init_from_phonons(BaTiO3)
    #calculator.check_asr()

    struct = BaTiO3.structure.copy()
    #struct.coords += np.random.normal(size = struct.coords.shape, scale = 0.05)

    # Move atom
    direction = 0
    delta = 0.001
    N_steps  = 10

    energies = np.zeros(N_steps, dtype = np.double)
    forces = np.zeros(N_steps, dtype = np.double)

    xvalues = np.zeros(N_steps, dtype = np.double)
    asr = np.zeros(N_steps, dtype = np.double)
    asr2 = np.zeros(N_steps, dtype = np.double)
    ss = []

    for i in range(N_steps):    
        struct.coords[0, direction] += delta
        xvalues[i] = struct.coords[0, direction]

        #r_value = np.zeros(3)
        #r_value[direction] = xvalues[i] - xvalues[0]
        
        atm = struct.get_ase_atoms()
        ss.append(atm)
        atm.set_calculator(calculator)

        energies[i] = atm.get_total_energy()
        forces[i] = atm.get_forces()[0, direction] #np.sum(atm.get_forces()[0, direction])

        asr[i] = np.linalg.norm( np.sum(atm.get_forces(), axis = 0))

        # Check the product of the position vector with the effective charges
        asr2[i] = np.linalg.norm( calculator.work_charges.dot(struct.coords.ravel()))

        #electric_field[i,:] = calculator.get_electric_field(r_value, verbose = True)
        #de_dr[i, :] = np.sum(calculator.get_derivative_efield(r_value)[:, direction, :])

        #asr2[i] = np.linalg.norm( np.sum(calculator.u_disps, axis =0))

        #print("CHARGES:")
        #print(calculator.charges)
        #print(calculator.charge_coords)

    if plot:
        #plt.plot(xvalues, energies, label = "Energy")
        plt.figure()
        plt.plot(xvalues, -np.gradient(energies, xvalues), label = "Numerical diff")
        plt.plot(xvalues, forces, label = "Forces")
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.title("BaTiO3")
        plt.plot(xvalues[1:-1], (-np.gradient(energies, xvalues) - forces)[1:-1], label = "Displacements")
        plt.figure()
        plt.plot(xvalues, asr, label = "ASR")
        #plt.plot(xvalues, asr2, label = "ASR - 2")
        plt.legend()
        plt.tight_layout()
        #ase.visualize.view(ss)

    

    
def test_unit_cell(plot = False):
    """
    Simple test to check if forces are derivatives of the energy
    """

    np.random.seed(0)

    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    

    BaTiO3 = CC.Phonons.Phonons("BaTiO3_")
    nat = BaTiO3.structure.N_atoms

    for i in range(BaTiO3.structure.N_atoms):
        BaTiO3.effective_charges[i, :, :] = np.eye(3) * np.trace(BaTiO3.effective_charges[i, :, :]) / 3

    calculator = calc.ElectrostaticCalculator()
    calculator.init_from_phonons(BaTiO3)
    #calculator.dielectric_tensor[:,:] = np.eye(3)

    struct = BaTiO3.structure.copy()
    struct.coords += np.random.normal(size = struct.coords.shape, scale = 0.05)

    # Move atom
    atm_id = 2
    direction = 0
    delta = 0.005
    N_steps  = 100

    energies = np.zeros(N_steps, dtype = np.double)
    forces = np.zeros(N_steps, dtype = np.double)

    xvalues = np.zeros(N_steps, dtype = np.double)
    asr = np.zeros(N_steps, dtype = np.double)
    ss = []

    for i in range(N_steps):    
        struct.coords[atm_id, direction] += delta
        xvalues[i] = struct.coords[atm_id, direction]
        
        atm = struct.get_ase_atoms()
        ss.append(atm)
        atm.set_calculator(calculator)
        energies[i] = atm.get_total_energy()
        forces[i] = atm.get_forces()[atm_id, direction]

        asr[i] = np.linalg.norm( np.sum(atm.get_forces(), axis = 0))


        
    if plot:
        plt.plot(xvalues, energies, label = "Energy")
        plt.figure()
        plt.plot(xvalues, -np.gradient(energies, delta), label = "Numerical diff")
        plt.plot(xvalues, forces, label = "Forces")
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, asr, label = "asr")
        plt.legend()
        plt.tight_layout()
        #ase.visualize.view(ss)


@pytest.mark.skip(reason="deprecated")
def test_one_atom_model(plot = False):
    structure = CC.Structure.Structure(1)
    structure.unit_cell = np.eye(3) * 10
    structure.has_unit_cell = True
    structure.set_masses({"H" : 1.008 / CC.Units.MASS_RY_TO_UMA})

    dyn = CC.Phonons.Phonons(structure)
    dyn.effective_charges = np.zeros( (1, 3, 3), dtype = np.double)
    dyn.effective_charges[0, :,:] = np.eye(3)
    dyn.dielectric_tensor = np.eye(3)
    nat = 1
    
    calculator = calc.ElectrostaticCalculator()
    calculator.init_from_phonons(dyn)
    #calculator.dielectric_tensor[:,:] = np.eye(3)

    struct = dyn.structure.copy()
    #struct.coords += np.random.normal(size = struct.coords.shape, scale = 0.05)

    # Move atom
    atm_id = 0
    direction = 0
    delta = 0.01
    N_steps  = 50

    e_field_point = np.zeros(3, dtype = np.double)
    e_field_point[1] = 1

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

        r_mean = np.mean(calculator.charge_coords, axis = 0)

    #assert np.max(np.abs(forces[1:-1] + np.gradient(energies, delta)[1:-1])) < 1e-6


    if plot:
        plt.plot(xvalues, energies, label = "Energy")
        plt.figure()
        plt.plot(xvalues, -np.gradient(energies, delta), label = "Numerical diff")
        plt.plot(xvalues, forces, label = "Forces")
        plt.legend()
        plt.tight_layout()

if __name__ == "__main__":
    test_julia_calculator()
    #test_1D(True)
    test_2D(True)
    test_total_translation(True)
    #test_unit_cell(plot = True)
    #test_one_atom_model(plot = True)
    plt.show()
