Tutorial 2: Supercell Calculations
==================================

Work with supercells in py-electrostatic.

Step 1: Generate Supercell
--------------------------

.. code-block:: python

   import cellconstructor as CC
   import cellconstructor.Phonons

   dyn = CC.Phonons.Phonons("BaTiO3_")
   primitive = dyn.structure
   
   supercell = primitive.generate_supercell((2, 2, 2))
   print(f"Primitive: {primitive.N_atoms} atoms")
   print(f"Supercell: {supercell.N_atoms} atoms")

Step 2: Initialize for Supercell
--------------------------------

.. code-block:: python

   import pyelectrostatic.calculator as calc

   calculator = calc.ElectrostaticCalculator()
   calculator.eta = 0.5
   calculator.cutoff = 5.0
   calculator.init(
       primitive,
       dyn.effective_charges,
       dyn.dielectric_tensor,
       unique_atom_element="Ba",  # One per primitive cell
       supercell=(2, 2, 2)
   )

Step 3: Verify Energy Consistency
---------------------------------

Energy per atom should be the same for all supercell sizes:

.. code-block:: python

   for size in [(1, 1, 1), (2, 2, 2)]:
       sc = primitive.generate_supercell(size)
       
       calc_sc = calc.ElectrostaticCalculator()
       calc_sc.init(primitive, dyn.effective_charges,
                    dyn.dielectric_tensor, "Ba", supercell=size)
       
       atoms = sc.get_ase_atoms()
       atoms.calc = calc_sc
       energy = atoms.get_total_energy()
       
       print(f"Size {size}: E/atom = {energy/sc.N_atoms:.10f} eV")

Complete Example
----------------

.. literalinclude:: ../examples/02_supercell.py
   :language: python

Running
-------

.. code-block:: bash

   python 02_supercell.py
