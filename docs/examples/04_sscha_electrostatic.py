"""
Tutorial 4: SSCHA with Electrostatics

This example demonstrates setting up a SSCHA calculation with:
1. A fake short-range calculator (returns 0)
2. Long-range electrostatics
3. Combined using CompositeCalculator

Note: In practice, replace the fake calculator with your trained GAP potential.
"""

import numpy as np
import cellconstructor as CC
import cellconstructor.Phonons
from ase.calculators.calculator import Calculator as ASECalculator

try:
    import sscha
    import sscha.Ensemble
    import sscha.SchaMinimizer
    import sscha.Relax
    SSCHA_AVAILABLE = True
except ImportError:
    print("Warning: SSCHA not installed. Setup demonstration only.")
    SSCHA_AVAILABLE = False

import pyelectrostatic.calculator as calc
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class FakeShortRangeCalculator(ASECalculator):
    """Fake short-range calculator that returns zero."""
    
    def __init__(self, **kwargs):
        ASECalculator.__init__(self, **kwargs)
        self.implemented_properties = ["energy", "forces", "stress"]
    
    def calculate(self, atoms=None, properties=["energy"], system_changes=None):
        ASECalculator.calculate(self, atoms, properties, system_changes)
        nat = len(atoms)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((nat, 3)),
            "stress": np.zeros(6)
        }


print("=" * 60)
print("Tutorial 4: SSCHA with Electrostatics")
print("=" * 60)

print("\n1. Loading BaTiO3...")
dyn = CC.Phonons.Phonons("BaTiO3_")
dyn.Symmetrize()
dyn.ForcePositiveDefinite()
print(f"   Atoms: {dyn.structure.N_atoms}")

print("\n2. Creating fake short-range calculator...")
short_range = FakeShortRangeCalculator()
print("   ✓ Returns 0 (replace with GAP in practice)")

print("\n3. Creating electrostatic calculator...")
electrostatic = calc.ElectrostaticCalculator()
electrostatic.eta = 0.5
electrostatic.cutoff = 5.0
electrostatic.init_from_phonons(dyn)
if calc.is_julia_available():
    electrostatic.compute_stress = True
print(f"   eta={electrostatic.eta}Å, stress={electrostatic.compute_stress}")

print("\n4. Creating composite calculator...")
composite = calc.CompositeCalculator([short_range, electrostatic])
print("   ✓ Combined")

if SSCHA_AVAILABLE:
    print("\n5. Setting up SSCHA at T=300K...")
    try:
        ensemble = sscha.Ensemble.Ensemble(dyn, T0=300.0, supercell=dyn.GetSupercell())
        ensemble.generate(N=100)
        print(f"   Generated {len(ensemble.structures)} configurations")
        
        print("\n6. Computing ensemble...")
        ensemble.compute_ensemble(composite, compute_stress=True, verbose=True)
        print("   ✓ Ensemble computed")
        
        print("\n7. Setting up minimizer...")
        minimizer = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
        minimizer.min_step_dyn = 0.1
        minimizer.kong_liu_ratio = 0.5
        print("   ✓ Ready (run relax.vc_relax() to execute)")
    except Exception as e:
        print(f"   ! SSCHA setup error: {e}")
        print("   (This can happen with unstable dynamical matrices)")
else:
    print("\n5-7. SSCHA steps skipped (not installed)")

print("\n" + "=" * 60)
print("Tutorial 4 complete!")
print("=" * 60)
