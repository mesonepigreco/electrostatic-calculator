"""
Tutorial 3: Composite Calculator

This example demonstrates combining multiple calculators:
1. Create a fake short-range calculator (returns 0)
2. Create electrostatic long-range calculator
3. Combine using CompositeCalculator
"""

import cellconstructor as CC
import cellconstructor.Phonons
import pyelectrostatic.calculator as calc
import numpy as np
from ase.calculators.calculator import Calculator as ASECalculator
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class FakeShortRangeCalculator(ASECalculator):
    """Fake short-range calculator that returns zero energy and forces."""
    
    def __init__(self, **kwargs):
        ASECalculator.__init__(self, **kwargs)
        self.implemented_properties = ["energy", "forces"]
    
    def calculate(self, atoms=None, properties=["energy"], system_changes=None):
        ASECalculator.calculate(self, atoms, properties, system_changes)
        nat = len(atoms)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((nat, 3))
        }


print("=" * 60)
print("Tutorial 3: Composite Calculator")
print("=" * 60)

print("\n1. Loading BaTiO3...")
dyn = CC.Phonons.Phonons("BaTiO3_")
print(f"   Atoms: {dyn.structure.N_atoms}")

print("\n2. Creating fake short-range calculator...")
short_range = FakeShortRangeCalculator()
print("   ✓ Returns 0 energy and forces")

print("\n3. Creating electrostatic calculator...")
electrostatic = calc.ElectrostaticCalculator()
electrostatic.eta = 0.5
electrostatic.cutoff = 5.0
electrostatic.init_from_phonons(dyn)
print(f"   eta={electrostatic.eta}Å, {len(electrostatic.kpoints)} k-points")

print("\n4. Combining with CompositeCalculator...")
composite = calc.CompositeCalculator([short_range, electrostatic])
print("   ✓ Combined")

print("\n5. Testing on displaced structure...")
displaced = dyn.structure.copy()
displaced.coords[0, 0] += 0.1

atoms = displaced.get_ase_atoms()
atoms.calc = composite

energy = atoms.get_total_energy()
forces = atoms.get_forces()

print(f"   Total energy: {energy:.6f} eV")
print(f"   Max force: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Å")

print("\n6. Verifying electrostatic contribution...")
atoms.calc = electrostatic
energy_elec = atoms.get_total_energy()
print(f"   Electrostatic only: {energy_elec:.6f} eV")
print(f"   Match: {np.isclose(energy, energy_elec)}")

print("\n" + "=" * 60)
print("Tutorial 3 complete!")
print("=" * 60)
