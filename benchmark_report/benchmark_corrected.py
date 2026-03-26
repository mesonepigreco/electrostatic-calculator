"""
Corrected benchmark with proper JIT warm-up
This separates compilation time from execution time
"""

import numpy as np
import cellconstructor as CC
import cellconstructor.Phonons
import pyelectrostatic.calculator as calc
import time
import sys
import os

np.random.seed(42)

def benchmark_with_warmup(supercell_size, ba_tio3):
    """Benchmark with proper warm-up to exclude JIT compilation time."""
    
    print(f"\n{'='*60}")
    print(f"Supercell: {supercell_size}")
    print(f"{'='*60}")
    
    # Generate supercell
    s_sc = ba_tio3.structure.generate_supercell(supercell_size)
    n_atoms = s_sc.N_atoms
    print(f"Atoms: {n_atoms}")
    
    # Displace one atom
    s_sc.coords[0, :] += np.array([0.05, 0.03, 0.02])
    
    # Setup original calculator
    t0 = time.time()
    calc_orig = calc.ElectrostaticCalculator()
    calc_orig.eta = 0.5
    calc_orig.cutoff = 5.0
    calc_orig.compute_stress = False
    calc_orig.init(ba_tio3.structure.copy(), 
                   ba_tio3.effective_charges.copy(),
                   ba_tio3.dielectric_tensor.copy(),
                   unique_atom_element=ba_tio3.structure.atoms[0],
                   supercell=supercell_size,
                   use_nufft=False)
    t1 = time.time()

    print(f"  Original calculator setup time: {t1-t0:.4f}s")
    
    # Setup NUFFT calculator
    t0 = time.time()
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
    t1 = time.time()
    print(f"  NUFFT calculator setup time: {t1-t0:.4f}s")
    
    atm = s_sc.get_ase_atoms()
    
    # ORIGINAL: Warm-up then time
    print("  Original implementation:")
    atm.calc = calc_orig
    
    # Warm-up run (compilation)
    t0 = time.time()
    energy_orig = atm.get_total_energy()
    forces_orig = atm.get_forces()
    t1 = time.time()
    time_orig = t1- t0
    # Timed runs (after compilation)
    print(f"    Execution time: {time_orig:.4f}s")
    
    # NUFFT: Warm-up then time
    print("  NUFFT implementation:")
    atm.calc = calc_nufft
    
    # Warm-up run (compilation)
    t0 = time.time()
    energy_nufft = atm.get_total_energy()
    forces_nufft = atm.get_forces()
    t1 = time.time()
    time_nufft = t1 - t0
    print(f"    Execution time: {t1-t0:.4f}s")

    # Accuracy check
    energy_diff = abs(energy_orig - energy_nufft) / abs(energy_orig)
    force_diff = np.max(np.abs(forces_orig - forces_nufft)) / np.max(np.abs(forces_orig))
    
    print(f"\n  Results:")
    print(f"    Speedup: {time_orig/time_nufft:.1f}x")
    print(f"    Energy match: {energy_diff:.2e}")
    print(f"    Force match: {force_diff:.2e}")
    
    return {
        'supercell': supercell_size,
        'n_atoms': n_atoms,
        'n_cells': np.prod(supercell_size),
        'time_orig': time_orig,
        'time_nufft': time_nufft,
        'energy_diff': energy_diff,
        'force_diff': force_diff
    }


def main():
    # Load BaTiO3
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'BaTiO3')
    original_dir = os.getcwd()
    os.chdir(tests_dir)
    ba_tio3 = CC.Phonons.Phonons("BaTiO3_")
    ba_tio3.Symmetrize()
    os.chdir(original_dir)
    
    print("="*60)
    print("CORRECTED BENCHMARK: Proper JIT Warm-up")
    print("="*60)
    print(f"BaTiO3 unit cell: {ba_tio3.structure.N_atoms} atoms")
    print()
    print("Note: Each supercell benchmark includes its own warm-up")
    print("      run to exclude Julia JIT compilation time from")
    print("      the execution time measurement.")
    
    # Test all supercells
    supercell_sizes = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]  # Testing with 2 sizes for speed
    
    results = []
    for size in supercell_sizes:
        try:
            result = benchmark_with_warmup(size, ba_tio3)
            results.append(result)
        except Exception as e:
            print(f"Error with {size}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Save results
    print("\n" + "="*60)
    print("SUMMARY (with proper JIT handling)")
    print("="*60)
    
    if results:
        np.savez('benchmark_data_corrected.npz',
                 supercells=[str(r['supercell']) for r in results],
                 n_atoms=[r['n_atoms'] for r in results],
                 n_cells=[r['n_cells'] for r in results],
                 time_orig=[r['time_orig'] for r in results],
                 time_nufft=[r['time_nufft'] for r in results],
                 energy_diff=[r['energy_diff'] for r in results],
                 force_diff=[r['force_diff'] for r in results])
        
        print(f"\n{'Supercell':<12} {'Atoms':<8} {'Orig(s)':<10} {'NUFFT(s)':<10} {'Speedup':<10}")
        print("-" * 60)
        for r in results:
            print(f"{str(r['supercell']):<12} {r['n_atoms']:<8} {r['time_orig']:<10.4f} {r['time_nufft']:<10.4f} {r['time_orig']/r['time_nufft']:<10.1f}")
        
        # Check scaling
        if len(results) >= 2:
            n1, n2 = results[0]['n_atoms'], results[1]['n_atoms']
            t1_orig, t2_orig = results[0]['time_orig'], results[1]['time_orig']
            t1_nufft, t2_nufft = results[0]['time_nufft'], results[1]['time_nufft']
            
            ratio_n = n2 / n1
            ratio_orig = t2_orig / t1_orig
            ratio_nufft = t2_nufft / t1_nufft
            
            print(f"\nScaling check (from {n1} to {n2} atoms, ratio = {ratio_n:.2f}x):")
            print(f"  Original: {t1_orig:.4f}s -> {t2_orig:.4f}s (ratio = {ratio_orig:.2f})")
            print(f"    Expected O(N^3): {ratio_n**3:.1f}x")
            print(f"  NUFFT: {t1_nufft:.4f}s -> {t2_nufft:.4f}s (ratio = {ratio_nufft:.2f})")
            print(f"    Expected O(N^2): {ratio_n**2:.1f}x")
        
        print("\nData saved to benchmark_data_corrected.npz")


if __name__ == "__main__":
    main()
