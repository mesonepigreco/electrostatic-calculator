py-electrostatic: Long-range Electrostatics for Polar Materials
================================================================

**py-electrostatic** is a Python package for computing long-range electrostatic interactions in polar materials using Born effective charges and dielectric tensors. It integrates seamlessly with ASE (Atomic Simulation Environment) and can be combined with other calculators for machine learning potentials, DFT, and more.

Key Features
------------

* **Born Effective Charges**: Uses proper Born effective charge tensors for accurate polar interactions
* **Ewald Summation**: Efficient k-space summation with automatic convergence handling
* **NUFFT Acceleration**: Optional Non-Uniform FFT for O(N²) scaling (vs O(N³) standard)
* **ASE Integration**: Full compatibility with ASE calculators for MD, optimization, and more
* **SSCHA Integration**: Works with the Stochastic Self-Consistent Harmonic Approximation code
* **Julia Acceleration**: High-performance Julia backend for stress tensors and fast evaluation

Quick Start
-----------

.. code-block:: python

   import pyelectrostatic.calculator as calc
   import cellconstructor as CC

   # Load structure with effective charges
   dyn = CC.Phonons.Phonons("BaTiO3_")

   # Initialize calculator
   calculator = calc.ElectrostaticCalculator()
   calculator.eta = 0.5
   calculator.cutoff = 5.0
   calculator.init_from_phonons(dyn)

   # Compute energy and forces
   atm = dyn.structure.get_ase_atoms()
   atm.calc = calculator
   energy = atm.get_total_energy()
   forces = atm.get_forces()

Documentation Contents
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   user_guide/installation
   user_guide/theory_minimal
   user_guide/parameters

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/01_getting_started
   tutorials/02_supercell_calculations
   tutorials/03_composite_calculator
   tutorials/04_sscha_electrostatic

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference/index
   citations

Citation
--------

If you use py-electrostatic in your research, please cite:

L. Monacelli and N. Marzari, *Electrostatic interactions in atomistic and machine-learned potentials for polar materials*, Phys. Rev. B 113, 094101 (2026).

See the :doc:`citations` page for full citation information and BibTeX.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
