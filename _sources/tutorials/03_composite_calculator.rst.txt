Tutorial 3: Composite Calculator
=================================

Combine electrostatics with other calculators.

Basic Usage
-----------

.. code-block:: python

   import pyelectrostatic.calculator as calc

   # Create individual calculators
   short_range = ...  # Your ML potential, DFT, etc.
   electrostatic = calc.ElectrostaticCalculator()
   electrostatic.init_from_phonons(dyn)

   # Combine
   composite = calc.CompositeCalculator([
       short_range,
       electrostatic
   ])

   # Use like any ASE calculator
   atoms.calc = composite
   energy = atoms.get_total_energy()

Example with Fake Calculator
-----------------------------

.. code-block:: python

   from ase.calculators.calculator import Calculator as ASECalculator

   class FakeCalculator(ASECalculator):
       def __init__(self, **kwargs):
           ASECalculator.__init__(self, **kwargs)
           self.implemented_properties = ["energy", "forces"]
       
       def calculate(self, atoms=None, properties=["energy"], 
                     system_changes=None):
           ASECalculator.calculate(self, atoms, properties, system_changes)
           nat = len(atoms)
           self.results = {
               "energy": 0.0,
               "forces": np.zeros((nat, 3))
           }

Complete Example
----------------

.. literalinclude:: ../examples/03_composite.py
   :language: python

Running
-------

.. code-block:: bash

   python 03_composite.py
