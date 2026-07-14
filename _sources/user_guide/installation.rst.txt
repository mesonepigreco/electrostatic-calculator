Installation
============

py-electrostatic can be installed via pip. The package requires Python 3.8 or later.

Requirements
------------

* Python >= 3.8
* NumPy
* SciPy
* ASE (Atomic Simulation Environment)
* CellConstructor
* Julia (optional but recommended for performance)

Basic Installation
------------------

Install from PyPI (when available):

.. code-block:: bash

   pip install py-electrostatic

Or install from source:

.. code-block:: bash

   git clone https://github.com/mesonepigreco/py-electrostatic.git
   cd py-electrostatic
   pip install -e .

Julia Installation (Recommended)
--------------------------------

For optimal performance, especially for stress tensor calculations, install Julia:

1. **Install Julia** (version 1.8 or later):

   * Download from https://julialang.org/downloads/
   * Or use conda: ``conda install -c conda-forge julia``

2. **Install PyJulia**:

   .. code-block:: bash

      pip install julia

3. **Configure PyJulia**:

   .. code-block:: python

      import julia
      julia.install()

4. **Install required Julia packages**:

   .. code-block:: julia

      using Pkg
      Pkg.add("FFTW")
      Pkg.add("DiffResults")
      Pkg.add("ForwardDiff")

Verification
------------

Test your installation:

.. code-block:: python

   import pyelectrostatic
   import pyelectrostatic.calculator as calc

   # Check if Julia is available
   print(calc.is_julia_available())  # Should print True if Julia is configured

Troubleshooting
---------------

Julia not found
~~~~~~~~~~~~~~~

If you get errors about Julia not being found:

1. Make sure Julia is in your system PATH
2. Try setting the ``JULIA_HOME`` environment variable
3. Re-run ``julia.install()``

Import errors
~~~~~~~~~~~~~

If you get import errors for CellConstructor:

.. code-block:: bash

   pip install cellconstructor

For compilation issues with CellConstructor, you may need a Fortran compiler:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install gfortran libblas-dev liblapack-dev

   # macOS
   brew install gcc

   # Then reinstall cellconstructor
   pip install --no-build-isolation --force-reinstall cellconstructor
