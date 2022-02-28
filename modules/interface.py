import pyelectrostatic
import pyelectrostatic.calculator as calculator

import numpy as np
import cellconstructor as CC
import cellconstructor.calculators

"""
This module is to interface pyelectrostatic with other codes
"""


def get_forceset_from_phonopy_disp(calculator, disp_file = "phonopy_disp.yaml", forceset_file = "FORCE_SETS"):
    """
    Read a phonopy_disp.yalm file, and use the given ASE calculator to write the FORCE_SET file
    This can be used to interface the code with phonopy for phonon calculation.

    Parameters
    ----------

        calculator : ASE or cellconstructor calculator.
            The calculator used to compute forces and energies
        disp_file : string
            Path to the file phonopy_disp.yaml that contains info about the displacements.
        forceset_file : string
            Path to the output file on which to write the FORCE_SET file. 

    """

    read_units = False
    read_supercell = False
    read_lattice = False
    read_atoms = False
    read_displacements = False
    lattice_vectors = np.zeros((3,3), dtype=np.double)

    index = 0

    atoms = []
    coords = []
    displacements = []

    length_unit = None
    fc_unit = None

    with open(disp_file, "r") as fp:
        for line in fp.readlines():
            # Crop the comments
            if "#" in line:
                line = line[: line.find("#")]

            line = line.strip()
            data = line.split()

            if not len(line):
                continue

            if line == "supercell:":
                read_supercell = True
                read_units = False
                continue 

            if "physical_unit" in line:
                read_units = True
                read_supercell = False

            if read_units:
                if "length" in line:
                    data = line.replace('"', ' ').split()
                    length_unit = data[-1]
                elif 'force_constants' in line: 
                    data = line.replace('"', ' ').split()
                    fc_unit = data[-1]


            if not read_supercell:
                continue

            if "lattice:" in line:
                read_lattice = True
                read_atoms = False
                read_displacements = False
                index = 0
                continue
            elif "points:" in line:
                read_atoms = True
                read_lattice = False
                read_displacements = False
                index = -1
            elif "displacements:" in line:
                read_atoms = False
                read_lattice = False
                read_displacements = True 
                index = 0

            
            if read_lattice:
                data = line.replace(",", " ").split()
                if len(data) == 6:
                    print("LINE: ", line)
                    print(data)
                    lattice_vectors[index,:] = [float(x) for x in data[2:5]]
                    index += 1
            
            if read_atoms:
                if "symbol" in line:
                    index += 1
                    atoms.append(data[-1])
                if "coordinates" in line:                
                    data = line.replace(",", " ").split()
                    coords.append(np.array([float(x) for x in data[2:5]]))
        
            if read_displacements:
                data = line.replace(",", " ").replace("[", '[ ').replace(']', ' ]').split()
                if "atom" in line:
                    index = int(data[-1])
                if len(data) == 5:
                    displacements.append({"atm" : index -1, "disp" : np.array([float(x) for x in data[1:4]])})


    # Get the conversion factors
    c_len = 1
    c_force = 1
    if length_unit == "au":
        c_len = CC.Units.BOHR_TO_ANGSTROM

    if fc_unit == "Ry/au^2":
        c_force = CC.Units.RY_TO_EV / CC.Units.BOHR_TO_ANGSTROM**2 * c_len
    elif fc_unit == "hartree/au^2":
        c_force = CC.Units.RY_TO_EV*2 / CC.Units.BOHR_TO_ANGSTROM**2 * c_len
    elif fc_unit == "mRy/au^2":
        c_force = CC.Units.RY_TO_EV*1000 / CC.Units.BOHR_TO_ANGSTROM**2 * c_len
    elif fc_unit == "hartree/Angstrom.au":
        c_force = CC.Units.RY_TO_EV*2 / CC.Units.BOHR_TO_ANGSTROM * c_len
    elif fc_unit == "eV/Angstrom.au":
        c_force = 1 / CC.Units.BOHR_TO_ANGSTROM * c_len


    
    # Create the centroid structure
    centroid = CC.Structure.Structure(len(atoms))
    centroid.atoms = atoms
    centroid.coords[:,:] = CC.Methods.cryst_to_cart(lattice_vectors, np.array(coords)) * c_len
    centroid.has_unit_cell = True
    centroid.unit_cell[:,:] = lattice_vectors * c_len


    # Prepare the displaced structures and compute the forces
    forces = []

    print(displacements)

    for i, disp in enumerate(displacements):
        print("Evaluating displacement:")
        print(disp)
        disp_structure = centroid.copy()
        disp_structure.coords[disp["atm"], :] += disp["disp"] * c_len


        # Compute with the calculator
        energ, force = CC.calculators.get_energy_forces(calculator, disp_structure)

        forces.append(force)

    
    # Now write the output file
    with open(forceset_file, "w") as fp:
        fp.write("{}\n{}\n\n".format(centroid.N_atoms, len(displacements)))

        for i, (disp, force) in enumerate(zip(displacements, forces)):
            fp.write("{}\n".format(disp["atm"]+1))
            fp.write("{:20.16f} {:20.16f} {:20.16f}\n".format(*list(disp["disp"])))
            for j in range(centroid.N_atoms):
                fp.write("{:16.8f} {:16.8f} {:16.8f}\n".format(*list(force[j, :] / c_force)))
            fp.write("\n")




            

