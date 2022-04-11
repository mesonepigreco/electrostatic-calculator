import pyelectrostatic, pyelectrostatic.fourier_calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.io
import numpy as np


def test_remove():
    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    atms = ase.io.read("rhombo-002.pwo")

    forces = atms.get_forces().copy()

    BaTiO3 = CC.Phonons.Phonons("../BaTiO3_")
    calculator = calc.FourierCalculator()
    calculator.init_from_dyn(BaTiO3)

    atms.set_calculator(calculator)
    
    print(atms.get_total_energy())
    print(forces - atms.get_forces())

    assert np.max(np.linalg.norm(calculator.u_disps, axis = 1)) < .5

    # struct = CC.Structure.Structure()
    # struct.generate_from_ase_atoms(atms)
    # new_struct = CC.Structure.Structure(struct.N_atoms + calculator.reference_supercell.N_atoms)
    # new_struct.atoms[: struct.N_atoms] = struct.atoms
    # new_struct.atoms[struct.N_atoms:] = calculator.reference_supercell.atoms
    # new_struct.coords[:struct.N_atoms] = struct.coords
    # new_struct.coords[struct.N_atoms:] = calculator.reference_supercell.coords
    # new_struct.has_unit_cell = True
    # new_struct.unit_cell = struct.unit_cell
    # new_struct.save_scf('comparison.scf')
    # ase.io.write('comparison.cif',
    #              new_struct.get_ase_atoms())

    # ase.io.write('reference.cif',
    #              calculator.reference_supercell.get_ase_atoms())
    # ase.io.write('current.cif', atms)

    # print('ATOMS: ', struct.N_atoms)
    # print('TOT:', new_struct.N_atoms)
    # print('REFERENCE:', calculator.reference_supercell.N_atoms)

if __name__ == "__main__":
    test_remove()
