Tutorials
=========

These tutorials will guide you through using py-electrostatic, from basic calculations to advanced applications with machine learning potentials.

Prerequisites
-------------

Before starting the tutorials, ensure you have:

1. Installed py-electrostatic (see :doc:`../user_guide/installation`)
2. Basic knowledge of Python and ASE
3. A structure file with effective charges (we use BaTiO3 as example)

Tutorial Overview
----------------

.. toctree::
   :maxdepth: 1

   01_getting_started
   02_supercell_calculations
   03_composite_calculator
   04_sscha_electrostatic

Tutorial 1: Getting Started
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn the basics:

* Loading structures with effective charges
* Initializing the calculator
* Computing energy and forces
* Understanding the output

:doc:`Start Tutorial 1 <01_getting_started>`

Tutorial 2: Supercell Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Work with larger systems:

* Primitive cell vs supercell concepts
* Proper supercell initialization
* Energy consistency checks
* Atom ordering in supercells

:doc:`Start Tutorial 2 <02_supercell_calculations>`

Tutorial 3: Composite Calculator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine electrostatics with other potentials:

* Why combine short-range and long-range?
* Using CompositeCalculator
* Example: Harmonic + Electrostatic
* Example: ML potential + Electrostatic

:doc:`Start Tutorial 3 <03_composite_calculator>`

Tutorial 4: SSCHA with Electrostatics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full anharmonic calculation:

* Overview of SSCHA method
* Using GAP potential with electrostatics
* Running finite-temperature structural relaxation
* Analyzing results

:doc:`Start Tutorial 4 <04_sscha_electrostatic>`

Example Files
-------------

All tutorial examples are available in the ``docs/examples/`` directory:

* ``01_basic_energy_force.py`` - Basic usage example
* ``02_supercell.py`` - Supercell calculations
* ``03_composite.py`` - Combining potentials
* ``04_sscha_electrostatic.py`` - Full SSCHA calculation

Getting Help
------------

If you encounter issues:

1. Check the :doc:`../user_guide/troubleshooting` guide
2. Review the :doc:`../user_guide/parameters` guide
3. Open an issue on GitHub: https://github.com/mesonepigreco/py-electrostatic/issues
