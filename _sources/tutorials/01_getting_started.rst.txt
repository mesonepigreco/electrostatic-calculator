Tutorial 1: Getting Started
===========================

Learn the basics of py-electrostatic: loading structures, initializing the calculator, and computing energy and forces.

Prerequisites
-------------

* py-electrostatic installed
* BaTiO3 example files (included in repository)

Step 1: Load Structure
----------------------

Load a structure with effective charges from a Quantum ESPRESSO dynamical matrix:

.. code-block:: python

   import cellconstructor as CC
   import cellconstructor.Phonons

   dyn = CC.Phonons.Phonons("BaTiO3_")
   structure = dyn.structure
   
   print(f"Atoms: {structure.N_atoms}")
   print(f"Charges shape: {dyn.effective_charges.shape}")  # (nat, 3, 3)
   print(f"Dielectric:\n{dyn.dielectric_tensor}")

Step 2: Initialize Calculator
-----------------------------

.. code-block:: python

   import pyelectrostatic.calculator as calc

   calculator = calc.ElectrostaticCalculator()
   calculator.eta = 0.5        # Gaussian screening (Å)
   calculator.cutoff = 5.0     # k-point cutoff
   calculator.init_from_phonons(dyn)

Step 3: Compute Energy and Forces
---------------------------------

.. code-block:: python

   atoms = structure.get_ase_atoms()
   atoms.calc = calculator
   
   energy = atoms.get_total_energy()
   forces = atoms.get_forces()
   
   print(f"Energy: {energy:.6f} eV")
   print(f"Max force: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Å")

Complete Example
----------------

.. literalinclude:: ../examples/01_basic_energy_force.py
   :language: python

Running
-------

.. code-block:: bash

   python 01_basic_energy_force.py
