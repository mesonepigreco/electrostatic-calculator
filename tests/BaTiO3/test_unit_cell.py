from unittest import BaseTestSuite
import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize

import numpy as np
import matplotlib.pyplot as plt


def test_total_translation(plot = False):
    

    np.random.seed(0)

    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    

    BaTiO3 = CC.Phonons.Phonons("BaTiO3_")
    nat = BaTiO3.structure.N_atoms

    for i in range(BaTiO3.structure.N_atoms):
        BaTiO3.effective_charges[i, :, :] = np.eye(3) * np.trace(BaTiO3.effective_charges[i, :, :]) / 3

    calculator = calc.LongRangeInteractions()
    calculator.init_from_dyn(BaTiO3)
    calculator.dielectric_tensor[:,:] = np.eye(3)

    struct = BaTiO3.structure.copy()
    struct.coords += np.random.normal(size = struct.coords.shape, scale = 0.05)

    # Move atom
    direction = 0
    delta = 0.01
    N_steps  = 50

    energies = np.zeros(N_steps, dtype = np.double)
    electric_field = np.zeros((N_steps, 3), dtype = np.double)
    de_dr = np.zeros((N_steps, 3), dtype = np.double)
    forces = np.zeros(N_steps, dtype = np.double)

    xvalues = np.zeros(N_steps, dtype = np.double)
    s_charges = np.zeros( (N_steps, 3 * nat, 3), dtype = np.double)
    s_force = np.zeros( (N_steps, 3 * nat, 3), dtype = np.double)
    asr = np.zeros(N_steps, dtype = np.double)
    asr2 = np.zeros(N_steps, dtype = np.double)
    ss = []

    for i in range(N_steps):    
        struct.coords[:, direction] += delta
        xvalues[i] = struct.coords[0, direction]

        r_value = np.zeros(3)
        r_value[direction] = xvalues[i] - xvalues[0]
        
        atm = struct.get_ase_atoms()
        ss.append(atm)
        atm.set_calculator(calculator)

        energies[i] = atm.get_total_energy()
        forces[i] = np.sum(atm.get_forces()[:, direction]) / nat

        asr[i] = np.linalg.norm( np.sum(atm.get_forces(), axis = 0))

        electric_field[i,:] = calculator.get_electric_field(r_value, verbose = True)
        de_dr[i, :] = np.sum(calculator.get_derivative_efield(r_value)[:, direction, :])

        asr2[i] = np.linalg.norm( np.sum(calculator.u_disps, axis =0))

        #print("CHARGES:")
        #print(calculator.charges)
        #print(calculator.charge_coords)

    if plot:
        plt.plot(xvalues, energies, label = "Energy")
        plt.figure()
        plt.plot(xvalues, -np.gradient(energies, delta), label = "Numerical diff")
        plt.plot(xvalues, forces, label = "Forces")
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, np.gradient(electric_field[:,0], delta), label = "Numerical efield diff")
        plt.plot(xvalues, de_dr[:, 0], label = "Anal efield diff") 
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, electric_field[:,0], label = "E field")
        #plt.plot(xvalues, asr2, label = "asr on displacements")
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
    s_charges = np.zeros( (N_steps, 3 * nat, 3), dtype = np.double)
    s_force = np.zeros( (N_steps, 3 * nat, 3), dtype = np.double)
    asr = np.zeros(N_steps, dtype = np.double)
    asr2 = np.zeros(N_steps, dtype = np.double)
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

        electric_field[i,:] = calculator.get_electric_field(np.zeros(3))
        de_dr[i, :] = calculator.get_derivative_efield(np.zeros(3))[atm_id, direction, :]

        asr2[i] = np.linalg.norm( np.sum(calculator.u_disps, axis =0))

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
        plt.plot(xvalues, np.gradient(electric_field[:,0], delta), label = "Numerical efield diff")
        plt.plot(xvalues, de_dr[:, 0], label = "Anal efield diff") 
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, asr, label = "asr")
        plt.plot(xvalues, asr2, label = "asr on displacements")
        plt.legend()
        plt.tight_layout()
        #ase.visualize.view(ss)


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
    
    calculator = calc.LongRangeInteractions()
    calculator.init_from_dyn(dyn)
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
    electric_field = np.zeros((N_steps, 3), dtype = np.double)
    de_dr = np.zeros((N_steps, 3), dtype = np.double)
    forces = np.zeros(N_steps, dtype = np.double)

    xvalues = np.zeros(N_steps, dtype = np.double)
    s_charges = np.zeros( (N_steps, 3 * nat, 3), dtype = np.double)
    s_force = np.zeros( (N_steps, 3 * nat, 3), dtype = np.double)

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
        electric_field[i,:] = calculator.get_electric_field(r_mean)
        de_dr[i, :] = calculator.get_derivative_efield(r_mean)[atm_id, direction, :]

        print("CHARGES:")
        print(calculator.charges)
        print(calculator.charge_coords)

    #assert np.max(np.abs(forces[1:-1] + np.gradient(energies, delta)[1:-1])) < 1e-6


    if plot:
        plt.plot(xvalues, energies, label = "Energy")
        plt.figure()
        plt.plot(xvalues, -np.gradient(energies, delta), label = "Numerical diff")
        plt.plot(xvalues, forces, label = "Forces")
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, np.gradient(electric_field[:,0], delta), label = "Numerical efield diff")
        plt.plot(xvalues, de_dr[:, 0], label = "Anal efield diff") 
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.plot(xvalues, electric_field[:,0], label = "E field")
        plt.legend()
        plt.tight_layout()

if __name__ == "__main__":
    #test_total_translation(True)
    test_unit_cell(plot = True)
    #test_one_atom_model(plot = True)
    plt.show()
