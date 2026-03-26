"""
Test for NUFFT implementation.

This test verifies that the NUFFT-based calculation produces forces that match
the standard k-point summation method in a 2x2x2 supercell.
"""

import pyelectrostatic.calculator as calc
import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np
import sys
import os


def test_nufft_forces_match_standard():
    """
    Test that NUFFT forces match standard k-point summation in a 2x2x2 supercell.
    
    This test compares the forces computed by the NUFFT-based method
    (O(N^2) complexity) against the standard k-point summation method
    (O(N^3) complexity) to verify correctness of the NUFFT implementation.
    """
    # Change to test directory to load BaTiO3 data
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # Load BaTiO3 dynamical matrix
    ba_tio3 = CC.Phonons.Phonons("BaTiO3_")
    ba_tio3.Symmetrize()
    
    # Use 2x2x2 supercell for testing
    supercell_size = (2, 2, 2)
    
    # Generate supercell structure
    s_sc = ba_tio3.structure.generate_supercell(supercell_size)
    
    # Displace one atom to create non-zero forces
    s_sc.coords[0, :] += np.array([0.05, 0.03, 0.02])
    
    # Setup standard calculator (without NUFFT)
    calc_standard = calc.ElectrostaticCalculator()
    calc_standard.eta = 0.5
    calc_standard.cutoff = 5.0
    calc_standard.compute_stress = False
    calc_standard.init(
        ba_tio3.structure.copy(),
        ba_tio3.effective_charges.copy(),
        ba_tio3.dielectric_tensor.copy(),
        unique_atom_element=ba_tio3.structure.atoms[0],
        supercell=supercell_size,
        use_nufft=False
    )
    
    # Setup NUFFT calculator
    calc_nufft = calc.ElectrostaticCalculator()
    calc_nufft.eta = 0.5
    calc_nufft.cutoff = 5.0
    calc_nufft.compute_stress = False
    calc_nufft.init(
        ba_tio3.structure.copy(),
        ba_tio3.effective_charges.copy(),
        ba_tio3.dielectric_tensor.copy(),
        unique_atom_element=ba_tio3.structure.atoms[0],
        supercell=supercell_size,
        use_nufft=True
    )
    
    # Get ASE atoms object
    atm = s_sc.get_ase_atoms()
    
    # Compute energy and forces with standard method
    atm.calc = calc_standard
    energy_standard = atm.get_total_energy()
    forces_standard = atm.get_forces()
    
    # Compute energy and forces with NUFFT method
    atm.calc = calc_nufft
    energy_nufft = atm.get_total_energy()
    forces_nufft = atm.get_forces()
    
    # Check energy match
    energy_diff = abs(energy_standard - energy_nufft)
    energy_tol = 1e-6  # eV
    
    # Check force match
    force_diff = np.max(np.abs(forces_standard - forces_nufft))
    force_tol = 1e-5  # eV/Angstrom
    
    print(f"\nNUFFT Test Results (2x2x2 supercell, {s_sc.N_atoms} atoms):")
    print(f"  Energy (standard):  {energy_standard:.10f} eV")
    print(f"  Energy (NUFFT):     {energy_nufft:.10f} eV")
    print(f"  Energy difference:  {energy_diff:.2e} eV")
    print(f"  Max force diff:     {force_diff:.2e} eV/Ang")
    
    # Assertions
    assert energy_diff < energy_tol, (
        f"Energy mismatch between standard and NUFFT methods: {energy_diff:.2e} eV "
        f"(tolerance: {energy_tol:.0e} eV)"
    )
    
    assert force_diff < force_tol, (
        f"Force mismatch between standard and NUFFT methods: {force_diff:.2e} eV/Ang "
        f"(tolerance: {force_tol:.0e} eV/Ang)"
    )


if __name__ == "__main__":
    test_nufft_forces_match_standard()
    print("\nTest passed!")
