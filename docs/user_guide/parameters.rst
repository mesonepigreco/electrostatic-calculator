Parameter Guide
===============

This guide explains all the key parameters in py-electrostatic and how to choose them.

ElectrostaticCalculator Parameters
----------------------------------

eta
~~~

**Type**: float (Angstroms)
**Default**: 6.0

The Gaussian screening parameter for Ewald summation. Controls the width of the Gaussian charge distribution used to screen the Coulomb interaction.

**How to choose:**

* Start with 0.5-1.0 Å for most materials
* Smaller values need more k-points but give better accuracy
* Larger values are faster but less accurate
* Check convergence by varying η and monitoring energy/forces

**Example:**

.. code-block:: python

   calculator = calc.ElectrostaticCalculator()
   calculator.eta = 0.5  # Good starting value

cutoff
~~~~~~

**Type**: float (dimensionless)
**Default**: 5.0

Determines the k-point cutoff for the reciprocal space sum. Include all k-points with :math:`|k| < \text{cutoff}/\eta`.

**How to choose:**

* Start with 3.0-5.0
* Higher values = more accurate but slower
* Check convergence by increasing until energy stabilizes
* For small cells, may need larger values

**Example:**

.. code-block:: python

   calculator.cutoff = 5.0  # Include k-points up to |k| < 5/η

use_nufft
~~~~~~~~~

**Type**: bool
**Default**: True

Enable Non-Uniform FFT for O(N²) scaling instead of O(N³).

**When to use:**

* **True**: Large systems (>50 atoms), faster for big supercells
* **False**: Small systems, debugging, or when Julia is not available

**Example:**

.. code-block:: python

   calculator.init(structure, charges, dielectric, 
                   supercell=(2,2,2), use_nufft=True)

compute_stress
~~~~~~~~~~~~~~

**Type**: bool
**Default**: False (auto-detected based on Julia availability)

Enable stress tensor calculation. Requires Julia.

**Example:**

.. code-block:: python

   calculator.compute_stress = True  # Enable stress calculation

julia_speedup
~~~~~~~~~~~~~

**Type**: bool
**Default**: True

Use Julia backend for accelerated calculations. Automatically disabled if Julia is not available.

supercell
~~~~~~~~~

**Type**: tuple(int, int, int)
**Default**: (1, 1, 1)

The supercell dimensions when working with supercells. The calculator will automatically replicate the primitive cell effective charges.

**Example:**

.. code-block:: python

   calculator.init(primitive_structure, charges, dielectric,
                   supercell=(2, 2, 2))  # 2x2x2 supercell

unique_atom_element
~~~~~~~~~~~~~~~~~~~

**Type**: str
**Default**: None (uses first atom)

The atomic species used to identify the origin of the structure. There should be exactly one per primitive cell. Used for proper supercell handling.

**Example:**

.. code-block:: python

   calculator.init(structure, charges, dielectric,
                   unique_atom_element="Ba")  # Use Ba as reference

Convergence Checking
--------------------

Use the utility functions to check parameter convergence:

.. code-block:: python

   from pyelectrostatic.utils import study_cutoff_convergence

   # Check how energy converges with cutoff
   energies, cutoffs = study_cutoff_convergence(
       structure, charges, dielectric,
       eta=0.5, cutoffs=[3, 4, 5, 6, 7]
   )

Recommended Starting Values
---------------------------

For most polar materials:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Starting Value
     - Notes
   * - eta
     - 0.5-1.0 Å
     - Smaller for accuracy, larger for speed
   * - cutoff
     - 5.0
     - Increase if not converged
   * - use_nufft
     - True
     - Use False only for small systems
   * - compute_stress
     - True (if Julia available)
     - Needed for variable cell calculations

Common Issues
-------------

"No k-points for the sum"
~~~~~~~~~~~~~~~~~~~~~~~~~~

The cell is too small for the given eta. Solutions:

* Decrease eta (try 0.3-0.5)
* Increase cutoff
* Use a supercell

Energy not converged
~~~~~~~~~~~~~~~~~~~~

Increase cutoff gradually until energy changes by < 1e-6 eV/atom.

Slow calculations
~~~~~~~~~~~~~~~~~

* Enable use_nufft=True (requires Julia)
* Increase eta slightly
* Decrease cutoff (if accuracy allows)
* Use Julia acceleration
