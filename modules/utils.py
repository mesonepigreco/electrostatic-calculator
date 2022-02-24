import pyelectrostatic
import pyelectrostatic.calculator as calculator

import cellconstructor as CC, cellconstructor.Structure

import numpy as np

import sys, os

import matplotlib, matplotlib.pyplot as plt

def study_cutoff_convergence(calculator, structure, cutoffs = np.linspace(0, 30, 10), progress = True):
    """
    STUDY CUTOFF CONVERGENCE
    ========================

    This utility function helps to choose a good cutoff for the convergence.
    You need to pass the calculator, a target structure and a list of cutoffs,
    we return the total energies for each cutoff and the forces on the atoms. 

    Parameters
    ----------
        calculator : pyelectrostatic.calculator.LongRangeInteractions
            The long range interaction calculator
        structure : cellconstructor.Structure.Structure or ase.Atoms
            Either a cellconstructor or an ASE crystal structure on which the
            test is carried.
        cutoffs : list (or ndarray), optional
            The list of the cutoffs (in Angstrom). 
            The default value is 10 steps between 0 and 30 A
        progress : bool
            If true, print in stdout the progress of the calculation

    Results
    -------
        energies : ndarray(size = len(cutoffs))
            The value of the total energy per unit cell for each cutoff value
        forces : ndarray(size = (len(cutoffs), nat, 3))
            The force that acts on each atom for each value of the cutoff
    """

    # Convert to ase atoms
    ase_atm = structure 
    if isinstance(structure, CC.Structure.Structure):
        ase_atm = structure.get_ase_atoms()

    old_cutoff = calculator.cutoff

    nat = len(ase_atm)

    energies = np.zeros( len(cutoffs), dtype = np.double)
    forces = np.zeros( (len(cutoffs), nat, 3), dtype = np.double)

    for i, cut in enumerate(cutoffs):
        # Reset the cutoff
        calculator.reset_cutoff(cut)
        ase_atm.set_calculator(calculator)

        energies[i] = ase_atm.get_total_energy()
        forces[i, :, :] = ase_atm.get_forces()

        if progress:
            sys.stdout.write("\rCutoff convergence test {:3d} \%".format((i+1) * 100 // len(cutoffs)))
            sys.stdout.flush()
    
    if progress:
        sys.stdout.write("\n")
    return energies, forces


def plot_cutoff_convergence(cutoffs, energies, forces, show = True, save_data = None, last_reference = True):
    """
    PLOT THE CONVERGENCE OF THE CUTOFF
    ==================================

    This is a simple utility that plots in a fancy way the calculation
    performed by study_cutoff_convergence.

    Parameters
    ----------
        cutoffs : ndarray
            The list of the cutoffs values
        energies : ndarray
            the list of total energy for each cutoff
        forces : ndarray
            The total forces on the structure for each cutoff
        show : bool
            If true, calls the show function of matplotlib at the end
        save_data : string
            Optional: path to a file on which the data are saved for further analysis
        last_reference : bool
            If true, subtract the last cutoff as a reference

    Example 
    -------

        >>>   # [initialize the calculator and structure]
        >>>   cutoffs = np.linspace(0, 30, 10)
        >>>   energies, forces = pyelectrostatic.utils.study_cutoff_convergence(calculator, structure, cutoffs)
        >>>   pyelectrostatic.utils.plot_cutoff_convergence(cutoffs, energies, forces)

    """


    new_e = energies
    if last_reference:
        new_e -= energies[-1]
    new_e *= 1000
    ncut, nat, _ = forces.shape
    ff = forces
    if last_reference:
        ff -= forces[-1,:,:]

    new_f = np.linalg.norm(np.reshape(ff, (ncut, 3*nat)), axis = 1)

    fig, axarr = plt.subplots(nrows = 2, ncols = 1, dpi = 150, sharex = True, figsize = (10,5))

    lbl_info = {"fontsize" : 13, "fontname" : "Liberation Serif"}
    title_info = {"fontsize" : 16, "fontname" : "Liberation Serif"}

    axarr[0].plot(cutoffs, new_e)
    axarr[0].set_title("Energy", **title_info)
    #axarr[0].set_xlabel("Cutoff [$\AA$]", **lbl_info)
    axarr[0].set_ylabel("Energy [meV] (reference is the last)", **lbl_info)
    axarr[1].set_title("Forces", **title_info)
    axarr[1].set_xlabel("Cutoff [$\AA$]", **lbl_info)
    axarr[1].set_ylabel("Force convergence [eV/A]")

    axarr[0].plot(cutoffs, new_e)
    axarr[1].plot(cutoffs, new_f)

    fig.tight_layout()

    if save_data is not None:
        np.savetxt(save_data, np.transpose([cutoffs, new_e, new_f]), header = "Cutoffs [A]; Energy - reference [meV]; Force - reference [eV/A]")

    if show:
        plt.show()




