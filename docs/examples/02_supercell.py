"""
Tutorial 2: Supercell Calculations

This example demonstrates:
1. Generating supercells from primitive cells
2. Initializing calculator for supercells
3. Verifying energy extensivity
4. Working with displaced supercells
"""

import cellconstructor as CC
import cellconstructor.Phonons
import pyelectrostatic.calculator as calc
import numpy as np
import os

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Tutorial 2: Supercell Calculations")
print("=" * 60)

# Load primitive cell
print("\n1. Loading primitive cell...")
dyn = CC.Phonons.Phonons("BaTiO3_")
primitive = dyn.structure

print(f"   Primitive cell: {primitive.N_atoms} atoms")
print(f"   Atomic species: {primitive.atoms}")

# Generate supercells
print("\n2. Generating supercells...")
supercell_1x1x1 = primitive.generate_supercell((1, 1, 1))
supercell_2x2x2 = primitive.generate_supercell((2, 2, 2))

print(f"   1x1x1: {supercell_1x1x1.N_atoms} atoms")
print(f"   2x2x2: {supercell_2x2x2.N_atoms} atoms")

# Step 3: Test energy consistency
print("\n3. Testing energy consistency (extensivity)...")
print("   Energy per atom should be the same for all supercell sizes")

sizes = [(1, 1, 1), (2, 2, 2)]
energies_per_atom = []

for size in sizes:
    # Create supercell
    sc = primitive.generate_supercell(size)
    
    # Initialize calculator
    calc_sc = calc.ElectrostaticCalculator()
    calc_sc.eta = 0.5
    calc_sc.cutoff = 5.0
    calc_sc.init(
        primitive,
        dyn.effective_charges,
        dyn.dielectric_tensor,
        unique_atom_element="Ba",
        supercell=size
    )
    
    # Compute energy
    atoms_sc = sc.get_ase_atoms()
    atoms_sc.calc = calc_sc
    energy = atoms_sc.get_total_energy()
    e_per_atom = energy / sc.N_atoms
    energies_per_atom.append(e_per_atom)
    
    print(f"   Size {size}: {sc.N_atoms:3d} atoms, E/atom = {e_per_atom:.10f} eV")

# Check consistency
diff = abs(energies_per_atom[1] - energies_per_atom[0])
print(f"\n   Difference: {diff:.2e} eV/atom")
if diff < 1e-10:
    print("   ✓ Energy is extensive (consistent)!")
else:
    print("   ✗ Warning: Energy not consistent!")

# Step 4: Displace atoms in supercell
print("\n4. Working with displaced supercell...")

# Create displaced 2x2x2 supercell
displaced_sc = supercell_2x2x2.copy()
displaced_sc.coords[0, 0] += 0.05  # Displace first atom 0.05 Å in x
displaced_sc.coords[5, 1] += 0.03  # Displace another atom 0.03 Å in y

# Use the same calculator (initialized for 2x2x2)
calc_2x2x2 = calc.ElectrostaticCalculator()
calc_2x2x2.eta = 0.5
calc_2x2x2.cutoff = 5.0
calc_2x2x2.init(
    primitive,
    dyn.effective_charges,
    dyn.dielectric_tensor,
    unique_atom_element="Ba",
    supercell=(2, 2, 2)
)

# Compute energy and forces
displaced_atoms = displaced_sc.get_ase_atoms()
displaced_atoms.calc = calc_2x2x2

energy_sc = displaced_atoms.get_total_energy()
forces_sc = displaced_atoms.get_forces()

print(f"   Supercell energy: {energy_sc:.6f} eV")
print(f"   Energy per atom: {energy_sc / displaced_sc.N_atoms:.6f} eV")
print(f"   Max force: {np.max(np.linalg.norm(forces_sc, axis=1)):.6f} eV/Å")

# Show forces on displaced atoms
print(f"\n   Forces on displaced atoms:")
print(f"   Atom 0: [{forces_sc[0, 0]:.4f}, {forces_sc[0, 1]:.4f}, {forces_sc[0, 2]:.4f}] eV/Å")
print(f"   Atom 5: [{forces_sc[5, 0]:.4f}, {forces_sc[5, 1]:.4f}, {forces_sc[5, 2]:.4f}] eV/Å")

# Step 5: Compare with primitive cell calculation
print("\n5. Comparing with equivalent primitive cell displacement...")

# Displace primitive cell equivalent atom
# In a 2x2x2 supercell, atom 0 corresponds to atom 0 in primitive
# Atom 5 corresponds to atom 0 in the second cell
displaced_primitive = primitive.copy()
displaced_primitive.coords[0, 0] += 0.05

calc_prim = calc.ElectrostaticCalculator()
calc_prim.eta = 0.5
calc_prim.cutoff = 5.0
calc_prim.init_from_phonons(dyn)

atoms_prim = displaced_primitive.get_ase_atoms()
atoms_prim.calc = calc_prim

energy_prim = atoms_prim.get_total_energy()
forces_prim = atoms_prim.get_forces()

print(f"   Primitive energy: {energy_prim:.6f} eV")
print(f"   Supercell energy per cell: {energy_sc / 8:.6f} eV")  # 8 cells in 2x2x2
print(f"   Difference: {abs(energy_prim - energy_sc/8):.2e} eV")

print("\n" + "=" * 60)
print("Tutorial 2 complete!")
print("=" * 60)
