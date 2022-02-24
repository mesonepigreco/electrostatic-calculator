# py-electrostatic
A python calculator that can calculate the electrostatic long-range forces and stress tensor from Born effective charges and the dielectric tensor. 


## Requirements

1. CellConstructor
2. ASE

The requirements can be installed as

```
pip install -r requirements.txt
```

If you are not using anaconda and need root privileges, add the --user flag at the end of the command.

## Installation

The code can be installed like any python package with

```
python setup.py install
```

also in this case, add the --user option if you need root privileges.


## Usage

The pyelectrostatic can be used as a simple ASE calculator.
It requires to run to have effective charges and the dielectric tensor of a reference structure.

It is interfaced with CellConstructor, and these information can be read from the standard Quantum-Espresso 
dynamical matrix.

Here an example

.. code :: python

    # Load CellConstructor to import the quantum-espresso dynamical matrix
    import cellconstructor as CC, cellconstructor.Phonons    
    import pyelectrostatic, pyelectrostatic.calculator as calc
    import numpy as np

    # The example dynamical matrix is located in tests/BaTiO3/BaTiO3_1
    # It contains the dielectric tensor, centroid structure and effective charges
    dyn = CC.Phonons.Phonons("BaTiO3_")

    # Now setup the calculator and initialize the dielectric tensor and effective charges
    calculator = calc.LongRangeInteractions()
    calculator.init_from_dyn(dyn)

    # Get a reference structure (randomize the atomic position to have a non zero dipole moment)
    struct = dyn.structure.copy()
    struct.coords[:,:] += np.random.normal(size = struct.coords.shape, scale = 0.05)

    # Get the ASE Atoms 
    ase_atoms = struct.get_ase_atoms()

    # Setup the calculator
    ase_atoms.set_calculator(calculator)

    # Now use it as a normal calculator:
    print("Energy = ", ase_atoms.get_total_energy(), " eV")
    print("Total forces [eV/A] is:")
    print(ase_atos.get_forces())



We employed cellconstructor Phonon class to read a quantum espresso dinamical matrix.
You can setup the calculation even without one, by directly passing the centrinds, effective charges
and dielectric tensor:

.. code :: python

    calculator.init(centroids = structure, effective_charges = eff_charges, dielectric_tensor = diel_tensor)

Where structure is a CellConstructor Structure, eff_charges is a numpy array of size (n_atoms, 3, 3)
(atomic index, electric field polarization, cartesian coordinate of the atom)
and diel_tensor is a symmetric 3x3 numpy array.

For more info, look at the tests inside the tests directory and see the calculator at work!


