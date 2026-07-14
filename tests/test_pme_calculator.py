"""
Test PME (Particle-Mesh Ewald) calculator against NUFFT implementation.
Verifies energy/force accuracy, force-energy consistency, spline convergence, and stress.
"""
import pytest
import numpy as np
import sys
import os

import cellconstructor as CC
import cellconstructor.Phonons
import pyelectrostatic.calculator as calc


np.random.seed(42)


def load_batio3():
    """Load BaTiO3 dynamical matrix from test data."""
    total_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BaTiO3")
    orig_dir = os.getcwd()
    os.chdir(total_path)
    ba_tio3 = CC.Phonons.Phonons("BaTiO3_")
    ba_tio3.Symmetrize()
    os.chdir(orig_dir)
    return ba_tio3


def test_pme_matches_nufft_batio3():
    """
    Test PME vs NUFFT on BaTiO3 2x2x2 supercell with displaced atom.
    PME is approximate due to B-splines, so tolerance is ~1e-4 eV.
    """
    ba_tio3 = load_batio3()
    supercell_size = (2, 2, 2)

    s_sc = ba_tio3.structure.generate_supercell(supercell_size)
    s_sc.coords[0, :] += np.array([0.05, 0.03, 0.02])

    # NUFFT calculator (reference)
    calc_nufft = calc.ElectrostaticCalculator()
    calc_nufft.eta = 0.5
    calc_nufft.cutoff = 5.0
    calc_nufft.compute_stress = False
    calc_nufft.init(ba_tio3.structure.copy(),
                    ba_tio3.effective_charges.copy(),
                    ba_tio3.dielectric_tensor.copy(),
                    unique_atom_element=ba_tio3.structure.atoms[0],
                    supercell=supercell_size,
                    use_nufft=True)

    # PME calculator
    calc_pme = calc.ElectrostaticCalculator()
    calc_pme.eta = 0.5
    calc_pme.cutoff = 5.0
    calc_pme.compute_stress = False
    calc_pme.spline_order = 8
    calc_pme.init(ba_tio3.structure.copy(),
                  ba_tio3.effective_charges.copy(),
                  ba_tio3.dielectric_tensor.copy(),
                  unique_atom_element=ba_tio3.structure.atoms[0],
                  supercell=supercell_size,
                  use_pme=True)

    atm = s_sc.get_ase_atoms()

    # NUFFT reference
    atm.calc = calc_nufft
    energy_nufft = atm.get_total_energy()
    forces_nufft = atm.get_forces()

    # PME
    atm.calc = calc_pme
    energy_pme = atm.get_total_energy()
    forces_pme = atm.get_forces()

    # Energy should be non-zero
    assert abs(energy_nufft) > 1e-6, f"NUFFT energy is zero: {energy_nufft}"
    assert abs(energy_pme) > 1e-6, f"PME energy is zero: {energy_pme}"

    energy_diff = abs(energy_nufft - energy_pme)
    force_diff = np.max(np.abs(forces_nufft - forces_pme))

    print(f"\nPME vs NUFFT (BaTiO3 2x2x2):")
    print(f"  Energy NUFFT: {energy_nufft:.10f} eV")
    print(f"  Energy PME:   {energy_pme:.10f} eV")
    print(f"  Energy diff:  {energy_diff:.2e} eV")
    print(f"  Max force diff: {force_diff:.2e} eV/Ang")

    assert energy_diff < 1e-4, \
        f"PME energy differs from NUFFT by {energy_diff:.2e} eV"
    assert force_diff < 1e-3, \
        f"PME forces differ from NUFFT by {force_diff:.2e} eV/Ang"


def test_pme_forces_are_energy_derivatives():
    """
    Verify F = -dE/dr by numerical differentiation.
    This tests internal consistency of the PME implementation.
    """
    ba_tio3 = load_batio3()
    supercell_size = (2, 2, 2)

    s_sc = ba_tio3.structure.generate_supercell(supercell_size)
    s_sc.coords[0, :] += np.array([0.05, 0.03, 0.02])

    calc_pme = calc.ElectrostaticCalculator()
    calc_pme.eta = 0.5
    calc_pme.cutoff = 5.0
    calc_pme.compute_stress = False
    calc_pme.spline_order = 8
    calc_pme.init(ba_tio3.structure.copy(),
                  ba_tio3.effective_charges.copy(),
                  ba_tio3.dielectric_tensor.copy(),
                  unique_atom_element=ba_tio3.structure.atoms[0],
                  supercell=supercell_size,
                  use_pme=True)

    atm = s_sc.get_ase_atoms()
    atm.calc = calc_pme
    forces = atm.get_forces()

    # Numerical differentiation
    delta = 0.001  # Angstrom
    forces_num = np.zeros_like(forces)

    for i in range(s_sc.N_atoms):
        for j in range(3):
            s_plus = s_sc.copy()
            s_plus.coords[i, j] += delta
            atm_p = s_plus.get_ase_atoms()
            atm_p.calc = calc_pme
            e_plus = atm_p.get_total_energy()

            s_minus = s_sc.copy()
            s_minus.coords[i, j] -= delta
            atm_m = s_minus.get_ase_atoms()
            atm_m.calc = calc_pme
            e_minus = atm_m.get_total_energy()

            forces_num[i, j] = -(e_plus - e_minus) / (2 * delta)

    max_diff = np.max(np.abs(forces - forces_num))
    print(f"\nPME force-energy consistency:")
    print(f"  Max |F_analytical - F_numerical|: {max_diff:.2e} eV/Ang")

    assert max_diff < 5e-4, \
        f"PME forces are not energy derivatives: max diff = {max_diff:.2e} eV/Ang"


def test_pme_spline_order_convergence():
    """
    Show that increasing spline order improves agreement with NUFFT.
    """
    ba_tio3 = load_batio3()
    supercell_size = (2, 2, 2)

    s_sc = ba_tio3.structure.generate_supercell(supercell_size)
    s_sc.coords[0, :] += np.array([0.05, 0.03, 0.02])

    # NUFFT reference
    calc_nufft = calc.ElectrostaticCalculator()
    calc_nufft.eta = 0.5
    calc_nufft.cutoff = 5.0
    calc_nufft.compute_stress = False
    calc_nufft.init(ba_tio3.structure.copy(),
                    ba_tio3.effective_charges.copy(),
                    ba_tio3.dielectric_tensor.copy(),
                    unique_atom_element=ba_tio3.structure.atoms[0],
                    supercell=supercell_size,
                    use_nufft=True)

    atm = s_sc.get_ase_atoms()
    atm.calc = calc_nufft
    energy_ref = atm.get_total_energy()
    forces_ref = atm.get_forces()

    errors = []
    spline_orders = [4, 6, 8]

    for p in spline_orders:
        calc_pme = calc.ElectrostaticCalculator()
        calc_pme.eta = 0.5
        calc_pme.cutoff = 5.0
        calc_pme.compute_stress = False
        calc_pme.spline_order = p
        calc_pme.init(ba_tio3.structure.copy(),
                      ba_tio3.effective_charges.copy(),
                      ba_tio3.dielectric_tensor.copy(),
                      unique_atom_element=ba_tio3.structure.atoms[0],
                      supercell=supercell_size,
                      use_pme=True)

        atm.calc = calc_pme
        energy_pme = atm.get_total_energy()
        forces_pme = atm.get_forces()

        energy_err = abs(energy_pme - energy_ref)
        force_err = np.max(np.abs(forces_pme - forces_ref))
        errors.append((energy_err, force_err))

        print(f"  Spline order {p}: energy err = {energy_err:.2e}, force err = {force_err:.2e}")

    # Verify convergence: error should decrease with increasing spline order
    assert errors[-1][0] < errors[0][0], \
        f"Energy error did not decrease: p={spline_orders[0]} err={errors[0][0]:.2e}, p={spline_orders[-1]} err={errors[-1][0]:.2e}"
    assert errors[-1][1] < errors[0][1], \
        f"Force error did not decrease: p={spline_orders[0]} err={errors[0][1]:.2e}, p={spline_orders[-1]} err={errors[-1][1]:.2e}"


def test_pme_stress():
    """
    Compare PME stress tensor to numerical strain derivatives.
    Same approach as tests/BaTiO3/test_stress.py.
    """
    ba_tio3 = load_batio3()
    supercell_size = (2, 2, 2)

    s_sc = ba_tio3.structure.generate_supercell(supercell_size)
    s_sc.coords[0, :] += np.array([0.05, 0.03, 0.02])

    calc_pme = calc.ElectrostaticCalculator()
    calc_pme.eta = 0.5
    calc_pme.cutoff = 5.0
    calc_pme.compute_stress = True
    calc_pme.spline_order = 8
    calc_pme.init(ba_tio3.structure.copy(),
                  ba_tio3.effective_charges.copy(),
                  ba_tio3.dielectric_tensor.copy(),
                  unique_atom_element=ba_tio3.structure.atoms[0],
                  supercell=supercell_size,
                  use_pme=True)

    atm = s_sc.get_ase_atoms()
    atm.calc = calc_pme

    energy = atm.get_potential_energy()
    stress = atm.get_stress()  # Voigt notation, 6 components

    # Numerical stress via strain derivatives
    delta_value = 1e-4
    delta_eps = np.zeros(6)

    for i in range(6):
        eps_voigt = np.zeros(6)
        eps_voigt[i] = delta_value

        s_strained = s_sc.strain(eps_voigt, voigt=True)
        atm_strain = s_strained.get_ase_atoms()
        atm_strain.calc = calc_pme
        delta_eps[i] = (atm_strain.get_potential_energy() - energy) / delta_value

    numerical_stress = delta_eps / s_sc.get_volume()

    print(f"\nPME Stress test:")
    print(f"  Analytical stress: {stress}")
    print(f"  Numerical stress:  {numerical_stress}")
    print(f"  Max difference:    {np.max(np.abs(stress - numerical_stress)):.2e}")

    assert np.allclose(stress, numerical_stress, atol=1e-5), \
        f"PME stress mismatch:\n  Analytical: {stress}\n  Numerical: {numerical_stress}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
