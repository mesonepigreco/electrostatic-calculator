Tutorial 4: SSCHA with Electrostatics
======================================

Run SSCHA calculations combining machine learning potentials with electrostatics.

Setup
-----

.. code-block:: python

   import cellconstructor as CC
   import pyelectrostatic.calculator as calc
   import sscha, sscha.Ensemble, sscha.SchaMinimizer, sscha.Relax

   # Load structure
   dyn = CC.Phonons.Phonons("BaTiO3_")
   dyn.Symmetrize()

   # Your ML potential (example uses fake calculator)
   short_range = YourGAPPotential()

   # Electrostatics
   electrostatic = calc.ElectrostaticCalculator()
   electrostatic.eta = 0.5
   electrostatic.cutoff = 5.0
   electrostatic.init_from_phonons(dyn)
   electrostatic.compute_stress = True

   # Combine
   composite = calc.CompositeCalculator([short_range, electrostatic])

SSCHA Calculation
-----------------

.. code-block:: python

   # Generate ensemble
   ensemble = sscha.Ensemble.Ensemble(dyn, T0=300.0, 
                                       supercell=dyn.GetSupercell())
   ensemble.generate(N=1000)
   
   # Compute with composite calculator
   ensemble.compute_ensemble(composite, compute_stress=True)
   
   # Minimize
   minimizer = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
   minimizer.min_step_dyn = 0.1
   
   relax = sscha.Relax.SSCHA(minimizer, ase_calculator=composite,
                             N_configs=1000, max_pop=50)
   relax.vc_relax(ensemble_loc='ensemble_pop')
   
   # Save results
   relax.minim.dyn.save_qe('final_dyn')

Complete Example
----------------

.. literalinclude:: ../examples/04_sscha_electrostatic.py
   :language: python

Running
-------

.. code-block:: bash

   python 04_sscha_electrostatic.py

Note: Replace ``FakeShortRangeCalculator`` with your GAP potential:

.. code-block:: python

   from quippy.potential import Potential
   short_range = Potential('IP GAP', param_filename='BaTiO3_GAP.xml')
