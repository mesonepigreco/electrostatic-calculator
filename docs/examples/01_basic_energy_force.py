"""
Tutorial 1: Getting Started with py-electrostatic

This example shows the basic usage of py-electrostatic:
1. Load a structure with effective charges
2. Initialize the calculator
3. Compute energy and forces
4. Displace an atom and recompute
"""

import cellconstructor as CC
import cellconstructor.Phonons
import pyelectrostatic.calculator as calc
import numpy as np
import os

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Tutorial 1: Getting Started")
print("=" * 60)

# Step 1: Load the structure with effective charges
print("\n1. Loading structure from dynamical matrix...")
dyn = CC.Phonons.Phonons("BaTiO3_")

structure = dyn.structure
print(f"   Number of atoms: {structure.N_atoms}")
print(f"   Atomic species: {structure.atoms}")
print(f"   Unit cell (Å):\n{structure.unit_cell}")

# Check effective charges and dielectric tensor
print(f"\n   Effective charges shape: {dyn.effective_charges.shape}")
print(f"   Dielectric tensor:\n{dyn.dielectric_tensor}")

# Step 2: Initialize the calculator
print("\n2. Initializing calculator...")
calculator = calc.ElectrostaticCalculator()
calculator.eta = 0.5        # Gaussian screening parameter (Å)
calculator.cutoff = 5.0     # k-point cutoff

# Initialize from phonons (loads structure, charges, and dielectric)
calculator.init_from_phonons(dyn)

print(f"   eta = {calculator.eta} Å")
print(f"   cutoff = {calculator.cutoff}")
print(f"   Number of k-points: {len(calculator.kpoints)}")
print(f"   Julia acceleration: {calc.is_julia_available()}")

# Step 3: Compute energy and forces on reference structure
print("\n3. Computing energy and forces on reference structure...")
atoms = structure.get_ase_atoms()
atoms.calc = calculator

energy = atoms.get_total_energy()
forces = atoms.get_forces()

print(f"   Energy: {energy:.6f} eV")
print(f"   Max force magnitude: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Å")

# Step 4: Displace an atom and recompute
print("\n4. Displacing first atom by 0.1 Å in x direction...")
displaced_structure = structure.copy()
displaced_structure.coords[0, 0] += 0.1  # Displace 0.1 Å in x

displaced_atoms = displaced_structure.get_ase_atoms()
displaced_atoms.calc = calculator

new_energy = displaced_atoms.get_total_energy()
new_forces = displaced_atoms.get_forces()

print(f"   New energy: {new_energy:.6f} eV")
print(f"   Energy change: {new_energy - energy:.6f} eV")
print(f"   Force on displaced atom: [{new_forces[0, 0]:.3f}, {new_forces[0, 1]:.3f}, {new_forces[0, 2]:.3f}] eV/Å")
print(f"   Max force magnitude: {np.max(np.linalg.norm(new_forces, axis=1)):.6f} eV/Å")

# Step 5: Check acoustic sum rule
print("\n5. Checking acoustic sum rule (ASR)...")
try:
    calculator.check_asr(threshold=1e-6)
    print("   ✓ Acoustic sum rule satisfied!")
except ValueError as e:
    print(f"   ✗ Acoustic sum rule violated: {e}")

print("\n" + "=" * 60)
print("Tutorial 1 complete!")
print("=" * 60)
