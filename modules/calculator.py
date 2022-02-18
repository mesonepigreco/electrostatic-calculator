import ase, ase.calculators
from ase.calculators.calculator import Calculator

import cellconstructor as CC
import cellconstructor.Structure

import numpy as np
import sys, os

# Conversion factor 1/(4pi eps0) * e^2 / 10^-10 -> my_units to  J 
#                   1/(4pi eps0) * e / 10^-10   -> my_units to  eV
__MYUNITS_TO_EV__ = 14.399645344793663

        
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

class LongRangeInteractions(Calculator):

    def __init__(self, *args, **kwargs):
        Calculator.__init__(self, *args, **kwargs)

        # Info about the structure
        self.centroids = CC.Structure.Structure()
        self.effective_charges = None
        self.dielectric_tensor = None

        self.charge_coords = None
        self.charge_values = None

        # Fixed supercell is used if you know that all the supercell of a calculations are the same
        self.fixed_supercell = None
        self.fixed_zeff = None

        # Integration details 
        self.cutoff = 60 # Angstrom

        self.implemented_properties = ["energy", "forces"]#, "stress"]

    def init_from_dyn(self, dyn, **kwargs):
        """
        It is possible to initialze the model directly from a quantum espresso dynamical matrix
        containing the dielectric tensor and Born effective charges.

        Parameters
        ----------
            dyn : CC.Phonons.Phonons()
                The dynamical matrix. Must contain the dielectric tensor and the effective charges
        """

        assert dyn.dielectric_tensor is not None, "Error, the dynamical matrix has no dielectric tensor"
        assert dyn.effective_charges is not None, "Error, the dynamical matrix has no effective charges"

        self.init(dyn.structure, dyn.effective_charges, dyn.dielectric_tensor, **kwargs)

    def init(self, centroids, effective_charges, dieletric_tensor, **kwargs):
        """
        Initialize the calculator by setting the centrids and effective charges.

        Both the structure and the effective charges are copied, 
        further modifications to the passed arguments do not affect the calculator.
        
        Parameters
        ----------
            centroids : CC.Structure.Structure()
                CellConstructor structure of the average centroid position.
                It can also be an ase atom object
            effective_charges : ndarray(sizeof=(N_atoms, 3, 3))
                The effective charges of the system, in the same format of CellConstructor.
                    - First index  -> atom
                    - Second index -> Electric field polarization
                    - Third index  -> Cartesian coordinate of the atomic displacement.
            dielectric_tensor : ndarray(size = (3,3))
                The dielectric tensor
            **kwargs : dict
                Any other keyword, as the cutoff for the coulomb interaction.

        """

        if isinstance(centroids, ase.Atoms):
            self.centroids.generate_from_ase_atoms(centroids)
        else:
            assert isinstance(centroids, CC.Structure.Structure), "Error, centroid parameter must be either a cellconstructor Structure or a ase Atoms"
            self.centroids = centroids.copy()
        
        assert effective_charges.shape == (self.centroids.N_atoms, 3, 3), "Error, wrong shape of the effective_charges tensor, expected {}, found {}".format((self.centroids.N_atoms, 3, 3), effective_charges.shape)

        self.effective_charges = effective_charges.copy()
        self.dielectric_tensor = dieletric_tensor.copy()
        
        # Change indices in a more convenient order
        self.zeff = np.einsum("abc -> bac", self.effective_charges).reshape((3, 3* self.centroids.N_atoms))

        for key in kwargs:
            self.__setattr__(key, kwargs[key])


    def is_initialized(self):
        if self.effective_charges is None or self.centroids.N_atoms < 1 or self.dielectric_tensor is None:
            return False
        return True

    def get_commensurate_supercell(self, supercell_structure, max_cell_value = 100):
        """
        Given a supercell structure, it computes the corresponding value of the supercell,
        and returns a new centroid structure, with the same unit cell as the one of the corresponding supercell.

        Note the supercell must be a simple rescaling of the primitive cell vectors, linear combinations are not yet
        supported.

        Parameters
        ----------
            supercell_structure : CC.Structure.Structure() 
                A generic supercell of the centroid structure used to initialize the calculator
            max_cell_value : int
                The maximum dimension of the supercell along each axis on which the most similar supercell
                is searched for.
        """

        assert self.is_initialized(), "Error, the effective charges must be initialized"

        total_supercell_dim = supercell_structure.N_atoms // self.centroids.N_atoms
        assert self.centroids.N_atoms * total_supercell_dim == supercell_structure.N_atoms, "Error, the number of atoms is not commensurate with the supercell"

        new_supercell = []
        for i in range(3):
            guess = []
            for k in range(1, max_cell_value + 1):
                v1 = self.centroids.unit_cell[i, :] * k
                guess.append(np.linalg.norm(supercell_structure.unit_cell[i, :] - v1))
            new_supercell.append(range(1, max_cell_value + 1)[np.argmin(guess)])
            
        
        # Generate the supercell and apply the strain to match the given cell
        new_cell = self.centroids.generate_supercell(new_supercell)
        first_itau = new_cell.get_itau(self.centroids) - 1
        new_cell.change_unit_cell(supercell_structure.unit_cell)
        
        # Suffle the atomic order in the correct way
        shuffle_itau = supercell_structure.get_itau(new_cell) - 1

        # Generate the effective charges in the supercell
        zeff = np.zeros((3, 3 *new_cell.N_atoms), dtype = np.double)
        
        # First shuffle
        for i in range(new_cell.N_atoms):
            zeff[:, 3*i : 3*i + 3] = self.zeff[:, 3*first_itau[i] : 3*first_itau[i] + 3]

        # Second shuffle
        final_zeff = np.zeros_like(zeff)
        for i in range(new_cell.N_atoms):
            final_zeff[:, 3*i : 3*i + 3] = zeff[:, 3*shuffle_itau[i] : 3*shuffle_itau[i] + 3]
        
        new_cell.coords = new_cell.coords[shuffle_itau, :]
        new_cell.atoms = [new_cell.atoms[x] for x in shuffle_itau]


        return new_cell, zeff
    
    def fix_supercell(self, supercell_structure):
        """
        Fix permanently the supercell structure. Can be undone with unfix_supercell.
        In this way the optimal strained supercell is not recomputed for each calculation.

        Parameters
        ----------
            supercell_structure: CC.Structure.Structure
                A supercell structure on which to compute calculations.
                could also be an ase atoms class
        """

        if isinstance(supercell_structure, ase.Atoms):
            new_structure = CC.Structure.Structure()
            new_structure.generate_from_ase_atoms(supercell_structure)
        else:
            new_structure = supercell_structure

        self.fixed_supercell, self.fixed_zeff = self.get_commensurate_supercell(new_structure)
    
    def unfix_supercell(self):
        """
        Revert fix_supercell.
        Now the optimal supercell will be recomputed for each atomic structure
        """
        self.fixed_supercell = None
        self.fixed_zeff = None

    def setup_charges(self, structure):
        """
        Setup the system of charges from the distorted structure.

        Parameters
        ----------
            structure : CC.Structure.Structure()
                The structure with distorted atoms (also accepted ase Atoms)
        """
        # Convert to cellconstructor
        structure = convert_to_cc_structure(structure)

        assert self.is_initialized(), "Error, you must initialize the charge system before computing anything"

        supercell = self.fixed_supercell
        zeff = self.fixed_zeff
        if self.fixed_supercell is None or self.fixed_zeff is None:
            supercell, zeff = self.get_commensurate_supercell(structure)
            
        
        # TODO add a flag to not assume the coordinate fixed in the cell position.
        u_disps = supercell.coords - structure.coords

        # If u_disp respect the cell, then also av_pos
        av_pos = .5 * u_disps + structure.coords
        new_zeff = np.reshape(zeff, (3, supercell.N_atoms, 3)) 
        dipole = np.einsum("abc, bc-> ab", new_zeff, u_disps).T  # Size (N_atoms, 3)

        # Get the charges and their coordinates
        self.charges = np.zeros(structure.N_atoms * 2, dtype = np.double)
        self.charges[: structure.N_atoms] = np.einsum("aia -> i", new_zeff) / 3
        self.charges[structure.N_atoms:] = - self.charges[: structure.N_atoms]
        
        _q_ = np.tile(self.charges, (3,1)).T
        self.charge_coords = np.zeros( (structure.N_atoms*2, 3), dtype = np.double)
        self.charge_coords[:, :] = np.tile(av_pos, (2,1)) + np.tile(dipole, (2,1)) / _q_

        self.u_disps = u_disps

        
    def evaluate_energy_forces(self):
        """
        Compute the forces of the charge system.

        Parameters
        ----------
            structure : optional
                If the structure on the supercell has not been fixed, it must be provided.

        Returns
        -------
            energy : float
                The value of the electrostatic energy
            forces : ndarray(size = (N_atoms, 3))
                Return the long range forces
            
        """
        assert self.is_initialized(), "Error, initialize the structure"

        total_energy = 0
        forces = np.zeros_like(self.u_disps)

        for i in range(self.fixed_supercell.N_atoms):
            Efield = self.get_electric_field(self.fixed_supercell.coords[i, :] + .5 * self.u_disps[i, :], discard = i)

            forces[i, :] =  Efield.dot( self.zeff[:, 3*i : 3*i+3])
            total_energy += forces[i, :].dot(self.u_disps[i, :])
        
        total_energy *= __MYUNITS_TO_EV__
        forces *= __MYUNITS_TO_EV__

        return total_energy, forces


    def get_electric_field(self, r, discard = None):
        """
        Get the electric field of a given position

        Parameters
        ----------
            r : ndarray(size = 3)
                The position in which you want the electric field
            dicard : int or none
                If int, the atom is discarded (but not its replica)
        
        Results
        -------
            E(r) : ndarray(size = 3)
                The electric field in that position.
        """
        assert self.charges is not None, "Error, setup the charges before computing the electric field."
        n = len(self.charges)
        
        # TODO: sum over the periodic replica ()

        
        new_mask = np.ones(n, dtype = bool)
        if discard is not None: # Do this only in the first unit cell
            # Discard the full dipole
            new_mask[discard] = False
            new_mask[discard + n//2] = False

        new_n = np.sum(new_mask.astype(int))

        disp = self.charge_coords[new_mask, :] - np.tile(r, (new_n, 1))
        dist = np.sqrt(np.sum(disp**2, axis = 1))

        q_over_dist = self.charges[new_mask] / dist**3

        Efield = np.sum( disp.dot(self.dielectric_tensor) * np.tile(q_over_dist, (3, 1)).T, axis = 0)

        return Efield

    def calculate(self, atoms=None, *args, **kwargs):
        super().calculate(atoms, *args, **kwargs)

        cc_struct = convert_to_cc_structure(atoms)
        # Check if the unit cell differ from the fixed one
        if self.fixed_supercell is None:
            self.fix_supercell(cc_struct)
        else:
            # Check if the supercell is different
            if np.max(np.abs(cc_struct.unit_cell - self.fixed_supercell.unit_cell)) > 1e-6:
                self.fix_supercell(cc_struct)

        # Setup the charges
        self.setup_charges(cc_struct)

        # Compute the energy and forces
        energy, forces = self.evaluate_energy_forces()

        self.results["energy"] = energy
        self.results["forces"] = forces 
        



    
