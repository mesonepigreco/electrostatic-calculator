import ase, ase.calculators
from ase.calculators.calculator import Calculator

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Methods

import numpy as np
import sys, os

import scipy, scipy.special

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
        self.charges = None

        # Fixed supercell is used if you know that all the supercell of a calculations are the same
        self.fixed_supercell = None
        self.fixed_zeff = None

        # Integration details 
        self.eta = 6 # Angstrom
        self.cutoff = 60 # Angstrom (cutoff for the PBC)
        self.use_pbc = True

        self.u_disps = None
        self.dipole_positions = None

        # Sum of periodic boundary conditions
        self.lattice_vectors = np.zeros((1, 3), dtype = np.int)

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

        if self.use_pbc:
            self.setup_pbc(self.cutoff)
    
    def unfix_supercell(self):
        """
        Revert fix_supercell.
        Now the optimal supercell will be recomputed for each atomic structure
        """
        self.fixed_supercell = None
        self.fixed_zeff = None

    def setup_pbc(self, cutoff = 60, r_x = 1.5, r_y = 1.5, r_z = 1.5):
        """
        Prepare the lattice vectors for the boundary condition.
        The cutoff is set so that in the charge summation only
        lattice vectors smaller than the cutoff are included.

        NOTE: The structure must be initialized and the supercell fixed.

        Parameters
        ----------
            cutoff : double
                The maximum value of the lattice vectors which is included in
                the periodic boundary images.
            r_x, r_y, r_z : float
                To search for the unit cell, look in a supercell commensurate with the one we are simulating of
                dimension (r_x, r_y, r_z) times the cutoff (divided by the lattice vector) 
                Put a 0 if you do not want periodic boundary condition on that direction.
                the x, y, z are aligned with the vectors of the unit cell, not the cartesian ones

        """

        assert self.is_initialized(), "Error, initialize the calculator before setting periodic boundary conditions"

        cell = self.fixed_supercell.unit_cell.copy()

        lattice_length = np.linalg.norm(cell, axis = 1)
        
        # Decide the maximum supercell on which
        n_len = (lattice_length / cutoff + 1).astype(int)
        n_len[0] = int(n_len[0] * r_x + .5)
        n_len[1] = int(n_len[1] * r_y + .5)
        n_len[2] = int(n_len[2] * r_z + .5)

        all_vectors = []

        _x_ = np.arange(-n_len[0], n_len[0] + 1)
        _y_ = np.arange(-n_len[1], n_len[1] + 1)
        _z_ = np.arange(-n_len[2], n_len[2] + 1)

        # Combine all the vectors in one big array with all combinations
        all_vectors = np.stack(np.meshgrid(_x_, _y_, _z_), -1).reshape(-1,3)

        # Extract the cartesian components of the lattice vectors
        cartesian_vectors = CC.Methods.cryst_to_cart(cell, all_vectors)

        # Measure the length of the lattice vectors
        vect_lengths = np.linalg.norm(cartesian_vectors, axis = 1)

        # Store only the vectors smaller than the given cutoff
        self.lattice_vectors = all_vectors[vect_lengths < cutoff, :]



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
            
        
        u_disps = - supercell.coords + structure.coords

        # Remove a general displacement to consider the recentering of the supercell structure
        nat = supercell.N_atoms
        u_disps -= np.tile(np.sum(u_disps, axis = 0), (nat,1)) / nat
        #print("DISPLACEMENTS:", u_disps)

        # TODO: apply the ASR on u_disps by removing a general shift of all the atoms

        # If u_disp respect the cell, then also av_pos
        av_pos = -.5 * u_disps + structure.coords
        #print("AV position:", av_pos)
        new_zeff = np.reshape(zeff, (3, supercell.N_atoms, 3)) 
        dipole = np.einsum("abc, bc-> ab", new_zeff, u_disps).T  # Size (N_atoms, 3)

        #print("Dipole: ", dipole)

        # Get the charges and their coordinates
        self.charges = np.zeros(structure.N_atoms * 2, dtype = np.double)
        self.charges[: structure.N_atoms] = np.einsum("aia -> i", new_zeff) / 3
        self.charges[structure.N_atoms:] = - self.charges[: structure.N_atoms]
        
        _q_ = np.tile(self.charges, (3,1)).T
        self.charge_coords = np.zeros( (structure.N_atoms*2, 3), dtype = np.double)
        self.charge_coords[:, :] = np.tile(av_pos, (2,1)) + np.tile(dipole, (2,1)) / (2 * _q_) 
        self.u_disps = u_disps
        self.dipole_positions = av_pos.copy()

        
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
        nat = forces.shape[0]

        for i in range(self.fixed_supercell.N_atoms):
            r_middle = self.dipole_positions[i, :]
            Efield = self.get_electric_field(r_middle)

            # TODO: check the correct sign (seems a -) and the ASR, whcih is this term that does not satisfy it.
            Efield_diff = self.get_derivative_efield(r_middle, atom_deriv = i)    # Why the - sign here?????

            #print("CHARGES:", self.charge_coords)
            #print("R middle: {}\nE field: {}\nE field diff: {}".format(r_middle, Efield, Efield_diff))


            E_dot_z = Efield.dot( self.zeff[:, 3*i : 3*i+3])
            forces[i, :] +=  E_dot_z

            # Remove the acustic sum rule contribution
            forces[:,:] -= np.einsum("i, b->ib", np.ones(nat, dtype=np.double) / nat, E_dot_z)

            # Add the contribution of the derivative of all atoms with respect to the electric field
            # THIS VIOLATES THE ACUSTIC SUM RULE AND HAS A NEGATIVE SIGN
            forces[:, :] += np.einsum("a, ba, cdb -> cd", self.u_disps[i, :], self.zeff[:, 3*i: 3*i+3], Efield_diff)
            total_energy -= Efield.dot( self.zeff[:, 3*i : 3*i+3]).dot(self.u_disps[i, :])
        
        total_energy *= __MYUNITS_TO_EV__
        forces *= __MYUNITS_TO_EV__

        return total_energy, forces


    def get_electric_field(self, r, exclude_atom = None, verbose = False):
        """
        Get the electric field of a given position

        Parameters
        ----------
            r : ndarray(size = 3)
                The position in which you want the electric field
            exclude_atom : int
                If present, the charges produced by the given atom is not accounted
        
        Results
        -------
            E(r) : ndarray(size = 3)
                The electric field in that position.
        """
        assert self.charges is not None, "Error, setup the charges before computing the electric field."
        n = len(self.charges)

        # Prepare the summation over the periodic boundary condition
        n_replica = self.lattice_vectors.shape[0]
        if verbose:
            print("Using {} replicas".format(n_replica))

        total_Efield = 0
        
        for jrep in range(n_replica):
            r_lattice = CC.Methods.cryst_to_cart(self.fixed_supercell.unit_cell, self.lattice_vectors[jrep, :])

            # Dipole has a self interaction between its two charges
            # We need to exclude this interaction to avoid violating the ASR (no need actually)
            good_mask =np.ones(n, dtype = bool)
            if exclude_atom is not None:
                good_mask[exclude_atom] = False
                good_mask[exclude_atom + n//2] = False

            charges = self.charges[good_mask]
            charge_coords = self.charge_coords[good_mask]
            new_n = np.sum(good_mask.astype(int))
            
            # TODO: sum over the periodic replica 
            # It should only require to iterate the following code by adding to the disp vector
            # a lattice vector

            disp = np.tile(r, (new_n, 1)) - charge_coords - np.tile(r_lattice, (new_n, 1))
            if verbose:
                print(" ------- ELECTRIC FIELD -------")
                print("   charge pos: {}".format(charge_coords))
                print("   r: {}".format(r))
                print("   charge displacement: {}".format(disp))
            dist = np.sqrt(np.sum(disp**2, axis = 1))

            q_inner_sphere = self.charges[:] * (scipy.special.erf(dist / (np.sqrt(2) * self.eta)) - \
                np.sqrt(2) * dist * np.exp(- dist**2 / (2 * self.eta**2))/ (np.sqrt(np.pi) * self.eta) )

            #print("   q in sphere: {}".format(q_inner_sphere))
            q_over_dist = q_inner_sphere / dist**3
            #print("   q over dist: {}".format(q_over_dist))

            epsilon_inv_r = disp.dot( np.linalg.inv(self.dielectric_tensor).T)
            #print("   dist with tensor: {}".format(epsilon_inv_r))
            efield_contrib = epsilon_inv_r * np.tile(q_over_dist, (3, 1)).T
            #print("   E field contrib: {}".format(efield_contrib))
            

            Efield = np.sum( efield_contrib, axis = 0)
            total_Efield += Efield

        if verbose:
            print("   e field: {}".format(Efield))
            print("--------- END FIELD ---------")
        return total_Efield

    def get_derivative_efield(self, r, atom_deriv = None):
        """
        Get the derivative of the electric field
        ----------------------------------------

        Parameters
        ----------
            r : ndarray(size = 3)
                The Cartesian position on which to compute the value of the derivative of the electric field
            atom_deriv : int
                If provided, then it also add the contribution of the change in the r position due
                to the derivative of the atom_deriv index

        Results
        -------
            dEk_dR_ij : ndarray(size = (nat, 3, 3))
                The derivative of the electric field with respect to each atomic position.
                The first two indices (nat, 3) indicates the atomic index and the Cartesian direction of the
                atom which we derive. The last index (3) indicates the direction of the electric field. 
        """

        # TODO: add the sum over the periodic boundary conditions.
        #       it should be sufficinet to add to the dist vector the lattice vector of the corresponding cell.
        #       and simply sum the final results.


        # Prepare the summation over the periodic boundary condition
        n_replica = self.lattice_vectors.shape[0]
        nat = self.fixed_supercell.N_atoms

        total_Efield_diff = np.zeros((nat, 3, 3), dtype = np.double)
        
        for jrep in range(n_replica):
            r_lattice = CC.Methods.cryst_to_cart(self.fixed_supercell.unit_cell, self.lattice_vectors[jrep, :])

            
            # Get the electric field modulus derivative with respect to each modulus of r for each charge post
            dist = r - self.charge_coords[:, :] - np.tile(r_lattice, (nat*2, 1))
            r_mod = np.linalg.norm(dist, axis = 1)

            dEtilde_dr_tmp = np.sqrt(2) * (r_mod**2 +3 * self.eta**2 ) * np.exp(-r_mod**2 / (2*self.eta**2)) / \
                (np.sqrt(np.pi) * r_mod**3 * self.eta**3)
            dEtilde_dr_tmp -= 3 * scipy.special.erf(r_mod / (np.sqrt(2) * self.eta)) / r_mod**4
            dEtilde_dr_tmp *= self.charges

            # Get the modulus of the electric field
            E_modulus = scipy.special.erf(r_mod/ (np.sqrt(2) * self.eta)) - \
                np.sqrt(2)*r_mod*np.exp(-r_mod**2/ (2*self.eta**2)) / (np.sqrt(np.pi) * self.eta)
            E_modulus *= self.charges / r_mod**3


            # Compose the first part of the derivative
            r_over_r = np.einsum("ab, a -> ab", dist,  1/r_mod)
            epsilon_r = np.einsum("bi, ai -> ab", np.linalg.inv(self.dielectric_tensor), dist)
            dE_dr_first = np.einsum("a, ab, ac-> abc", dEtilde_dr_tmp, r_over_r, epsilon_r)
            # First index is the atom id, second index is the cartesian coordinate of the derivative
            # last index is the electric field component

            # Compose the second part
            dE_dr_second = np.einsum("a, cb -> abc", E_modulus, np.linalg.inv(self.dielectric_tensor))

            # Size (ncharges, 3, 3), last index electric field direction, middle the charge cartesian coordinate
            dE_drtilde = dE_dr_first + dE_dr_second


            # Create the derivative of the charges with respect to the atomic positions
            I_coord = np.eye(3)
            I_atms = np.zeros((2*nat, nat), dtype = np.double)
            I_atms[:nat, :] = np.eye(nat)
            I_atms[nat:,:] = np.eye(nat)
            asr =  np.ones((2*nat, nat), dtype=np.double) / nat

            # Build the Z/q with the correct shape
            qravel = np.tile(self.charges[: nat], (3, 1)).T.ravel()
            z_over_q = self.zeff / np.tile(2* qravel, (3,1))
            z_over_q = z_over_q.reshape((3, nat, 3))

            I_full = np.einsum("ij, ab ->iajb", I_atms, I_coord)
            drcharge_dr = I_full / np.double(2) + np.einsum("ab, ij -> iajb", I_coord/2, asr)
            drcharge_dr[:nat, :,:,:] += np.einsum("aib, ij -> iajb", z_over_q, I_atms[:nat, :] - asr[:nat,:])
            drcharge_dr[nat:, :,:,:] -= np.einsum("aib, ij -> iajb", z_over_q, I_atms[:nat, :] - asr[:nat,:])

            # The minus sign comes that dE_drtilde has a minus sign when we derive with respect to the charge position
            dE_dR_pos = - np.einsum("iab, iajc-> jcb", dE_drtilde, drcharge_dr)
            
            # Now add the contribution if required to the derivative of the position on which the electric field is computed
            if atom_deriv is not None:
                dE_dr_target = np.sum(dE_drtilde, axis = 0)
                dE_dR_pos[atom_deriv, :, :] += dE_dr_target / 2
                for k in range(nat):
                    dE_dR_pos[k, :, :] += dE_dr_target / (2*nat)

            # Now pass from derivative of charge coordinates into derivative of atomic positions
            # Create a fake identity
            #I = np.einsum("a,bc -> bac", np.ones(nat, dtype = np.double), np.eye(3))


            # Now convert the derivative of the charge position to the derivative of the atomic position
            #dE_dR_pos = -np.einsum("abc, dab -> adc", dE_drtilde[:nat,:,:], I/2 + z_over_q)
            #dE_dR_pos += -np.einsum("abc, dab -> adc", dE_drtilde[nat:,:,:], I/2 - z_over_q)

            total_Efield_diff += dE_dR_pos

        return total_Efield_diff



    def calculate(self, atoms=None, *args, **kwargs):
        super().calculate(atoms, *args, **kwargs)

        cc_struct = convert_to_cc_structure(atoms)
        # TODO: Check if the unit cell differ from the fixed one
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
        



    
