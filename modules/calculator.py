import ase, ase.calculators
from ase.calculators.calculator import Calculator

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Methods

import numpy as np
import sys, os

import scipy, scipy.special
from typing import List, Union

def convert_to_cc_structure(item):
    """
    Convert any structure (either ASE or Celconstructor) into a 
    Cellconstructor Structure.
    """
    if isinstance(item, ase.Atoms):
        ret = CC.Structure.Structure()
        ret.generate_from_ase_atoms(item)
        return ret
    else:
        return item

class ElectrostaticCalculator(Calculator):
    def __init__(self, *args, **kwargs):
        Calculator.__init__(self, *args, **kwargs)

        self.eta = 6  # Default value is 6 Angstrom
        self.reference_structure = None 
        self.effective_charges = None
        self.work_charges = None  # The actually initialized effective charges
        self.dielectric_tensor = None
        self.reciprocal_vectors = None
        self.cutoff = 5 # Stop the sum when k > 5/ eta
        self.kpoints = None

    def init(self, reference_structure : CC.Structure.Structure, effective_charges : np.ndarray, dielectric_tensor : np.ndarray, supercell : tuple[int, int, int] = (1,1,1)) -> None:
        """
        INITIALIZE THE CALCULATOR
        =========================

        Setup the calculator to evaluate energy and forces. 
        The calculator need a reference structure and effective charges of all atoms in the system.
        These quantities needs to be setup only at the beginnig.
        The reference structure should match the correct order of atoms that are provided when the calculator is executed. 

        A supercell can be specified, if the calculator is actually going to be used in a supercell structure from the one provided.
        In that case, the default order is the same in the provided generate_supercell method from cellconstructor.Structure.Structure.
        This is the default behaviour of the set_tau subroutine from quantum espresso.

        Structure, effective charges and dielectric tensors are copied.

        Parameters
        ----------
            reference_structure : CC.Structure.Structure
                The average position of each atom.
            effective_charges : ndarray
                The effective charges, it must be a 3-rank tensor where the indices are
                Z[i, j, k]   -> i is the atom index, j is the polarization of the electric field, k is the atomic-cartesian coordinate.
            dielectric_tensor : ndarray
                The 3x3 matrix containing the high-frequency static dielectric tensor.
            supercell : tuple
                Optional, if provided, generates automatically the data for the calculation on a supercell.
        """

        self.reference_structure = reference_structure.generate_supercell(supercell)
        n_atoms = self.reference_structure.N_atoms
        self.effective_charges = np.zeros( (n_atoms, 3, 3), dtype = np.double)
        self.dielectric_tensor = dielectric_tensor.copy()

        for i in range(np.prod(supercell)):
            self.effective_charges[i * n_atoms : (i+1) * n_atoms, :, :] = effective_charges

        self.work_charges = np.zeros( (3, 3*n_atoms), dtype = np.double)
        for i in range(3):
            self.work_charges[i, :] = self.effective_charges[:, i, :].ravel()

        self.reciprocal_vectors = CC.Methods.get_reciprocal_vectors(self.reference_structure.unit_cell)


    def init_kpoints(self):
        r"""
        INITIALIZE THE K POINTS
        =======================

        Define the k points within the cutoff so that

        .. math::

            \left| \vec k \right| < \frac{C}{\eta}

        where :math:`C` is the cutoff and :math:`\eta` is the size of the
        charges.
        """

        # Initialize the sum over k
        max_values = [1 + int(x) for x in np.floor(np.linalg.norm(self.reciprocal_vectors, axis = 1) * self.eta / self.cutoff)]
        
        self.kpoints = []

        for l in range(-max_values, max_values + 1):
            for m in range(-max_values, max_values+1):
                for n in range(-max_values, max_values+1):
                    kvector = l * self.reciprocal_vectors[0, :] 
                    kvector += m * self.reciprocal_vectors[1, :] 
                    kvector += n * self.reciprocal_vectors[2, :] 

                    if np.linalg.norm(kvector) < self.cutoff / self.eta:
                         self.kpoints.append(kvector)
        
        self.kpoints = np.array(self.kpoints)
        self.energy = None
        self.force = None
        self.results = {}



    def init_from_phonons(self, dynamical_matrix : CC.Phonons.Phonons) -> None : 
        """
        INITIALIZE THE CALCULATOR
        =========================

        Uses the reference of the dynamical matrix to initialize the phonons. 
        Everything is read from the dynamical matrix, included the supercell.

        see init documentation for more details
        """

        assert dynamical_matrix.effective_charges is not None, "Error, the provided dynamical matrix has no effective charges"
        assert dynamical_matrix.dielectric_tensor is not None, "Error, the provided dynamical matrix has no dielectric tensor"

        self.init(dynamical_matrix.structure, dynamical_matrix.effective_charges, dynamical_matrix.dielectric_tensor, dynamical_matrix.GetSupercell())


    def _get_energy_force(self, struct : CC.Structure.Structure):
        """
        helper function to evaluate the energy
        """

        n_atoms = self.reference_structure.N_atoms

        delta_r = struct.coords - self.reference_structure.coords  

        n_kpoints = self.kpoints.shape[0]
        volume = struct.get_volume() * CC.Units.A_TO_BOHR

        energy = 0
        force = np.zeros_like(struct.coords)


        for kindex in range(n_kpoints):
            kvect = self.kpoints[kindex, :]
            k2 = kvect.dot(kvect)
            
            k_eps_k = kvect.dot(self.dielectric_tensor .dot(kvect))
            kk_matrix = np.outer(kvect, kvect) * np.exp(-self.eta**2 * k2 / 2) / k_eps_k

            #ZkkZ = np.einsum("ai, bj, ab -> ij", self.work_charges, self.work_charges, kk_matrix)
            ZkkZ = self.work_charges.T.dot(kk_matrix.dot(self.work_charges))

            for i in range(n_atoms):
                for j in range(n_atoms):
                    exp_factor = np.exp(-1j* kvect.dot(struct.coords[j, :] - struct.coords[i, :]))
                    cos_factor = exp_factor + np.conj(exp_factor)
                    sin_factor = 1j * (exp_factor - np.conj(exp_factor))
                    energy -= delta_r[i, :].dot(ZkkZ.dot(delta_r[j, :])) * exp_factor 

                    force[i, :] +=  ZkkZ.dot(delta_r[j, :]) * cos_factor
                    force[i, :] -= delta_r[i, :] * ZkkZ.dot(delta_r[j, :])  * kvect *  sin_factor


        self.energy = energy * 4 * np.pi / volume * CC.Units.HA_TO_EV
        self.force = force *  4 * np.pi / volume * CC.Units.HA_TO_EV / CC.Units.BOHR_TO_ANGSTROM


    def calculate(self, atoms=None, *args, **kwargs):
        super().calculate(atoms, *args, **kwargs)
        self.atoms = atoms

        if self.kpoints is None:
            raise ValueError("Error, calculator not initialized.")

        cc_struct = convert_to_cc_structure(atoms)

        # perform the actual calculation
        self._get_energy_force(cc_struct)


        self.results["energy"] = self.energy
        self.results["forces"] = self.force
        #self.results["dipole"] = self.get_dipole() 




        
        