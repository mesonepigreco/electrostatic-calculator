import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Methods
import cellconstructor.ForceTensor

import ase, ase.calculators, ase.calculators.calculator
import ase.visualize
import pyelectrostatic, pyelectrostatic.calculator as calculator

import numpy as np
import sys, os

ACCEPTED_MODES = ["add", "remove"]

class FourierCalculator(ase.calculators.calculator.Calculator):
    def __init__(self, *args, **kwargs):
        ase.calculators.calculator.Calculator.__init__(self, *args, **kwargs)

        self.centroids = None
        self.tensor = None 
        
        self.fixed_supercell_structure = None
        self.fixed_supercell = None
        self.fixed_itau = None 
        self.reference_id = None
        self.u_disps = None

        self.nac_fc = None


        self.results = {}
        self.implemented_properties = ["energy", "forces"]

    def is_initialized(self):
        """
        Check if the structure is initialized.
        """
        return True

    def init_from_dyn(self, dyn):
        """
        Initialize the calculator from a dynamical matrix
        with centroid positions and effective charges
        """

        self.tensor = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure, (1,1,1))

        assert dyn.effective_charges is not None, "Error, the given dynamical matrix must have effective charges."
        
        # Use a white dynamical matrix
        new_dyn = dyn.Copy()
        for iq, q in enumerate(dyn.q_tot):
            new_dyn.dynmats[iq][:,:] = 0
        self.tensor.SetupFromPhonons(new_dyn)

        # Remove the subtraction of the long-range forces from the tensor
        self.remove_tensor = self.tensor.tensor[:,:,:].copy()

        self.centroids = dyn.structure.copy()
        self.centroids.coords[:,:] -= self.centroids.coords[0, :]
        self.reference_supercell = self.centroids.copy()  # Reference supercell must match centroids in the correct supercell

    def setup_nac_fc(self, mesh_grid = (1,1,1)):
        """
        Setup the nonanalitical correction in the specified mesh grid (supercell)
        """

        q_grid = CC.symmetries.GetQGrid(self.centroids.unit_cell, mesh_grid)

        nat = self.centroids.N_atoms
        nat_sc = nat * np.prod(mesh_grid)
        self.fixed_supercell = mesh_grid

        self.dynq = np.zeros( (len(q_grid), 3*nat, 3*nat), dtype = np.complex128)

        self.tensor.tensor[:,:,:] = 0

        iq0 = 0
        for iq, q in enumerate(q_grid):
            # Leave gamma unchanged
            #if np.max(np.abs(q)) > 1e-6:
            self.dynq[iq, :, :] = self.tensor.Interpolate(-q, q_direct = np.zeros(3))


            if np.max(np.abs(q)) < 1e-6:
                iq0 = iq
        
        # Perform an uniform shift over the complete dynamical matrix to get something that is zero at gamma
        # This is a short range change, it should not affect the long range contribution
        for iq in range(len(q_grid)):
            self.dynq[iq,:,:] -=  self.dynq[iq0, :, :]
            
        
        self.fc = np.real(CC.Phonons.GetSupercellFCFromDyn(self.dynq, np.array(q_grid), self.centroids, self.reference_supercell))


    def get_energy_forces(self, structure):
        """
        Assuming the structure has been initialized and the fourier transform already performed,
        compute the energy and forces.
        """

        assert self.is_initialized(), "Error, initialize the calculator before using it,"
        nat_sc = structure.N_atoms
        u_disps = np.zeros_like(structure.coords)
        for i in range(nat_sc):
            u_disps[i, :] =  CC.Methods.get_closest_vector(structure.unit_cell, 
                                                           structure.coords[self.reference_id[i],:] - self.fixed_supercell_structure.coords[i, :])

            print('IDENTIFICATING {} WITH {}; c1 = {}; c2 = {}; dist = {}'.format(i, self.reference_id[i], self.fixed_supercell_structure.coords[i, :],
                                                                                  structure.coords[self.reference_id[i],:], np.linalg.norm(u_disps[i,:])))
        
        #ase.visualize.view(structure.get_ase_atoms())
        #ase.visualize.view(self.fixed_supercell_structure.get_ase_atoms())
        u_disps *= CC.Units.A_TO_BOHR
        print("U DISPS:")
        print(u_disps)

        # Forces now are in Ry/Bohr
        forces = -self.fc.dot(u_disps.ravel()).reshape((nat_sc, 3))
        
        # Shuffle the forces to match the correct itau
        new_forces = forces.copy()
        for i in range(nat_sc):
            new_forces[self.reference_id[i], :] = forces[i, :]

        # Energy in Ry
        energy = - forces.ravel().dot(u_disps.ravel()) / 2

        self.u_disps = u_disps

        self.results["forces"] = new_forces * CC.Units.RY_TO_EV / CC.Units.BOHR_TO_ANGSTROM
        self.results["energy"] = energy * CC.Units.RY_TO_EV

    def calculate(self, atoms = None, *args, **kwargs):
        """
        This is the actual function called by the ASE calculator.
        """

        ase.calculators.calculator.Calculator.calculate(self, atoms, *args, **kwargs)

        cc_struct = calculator.convert_to_cc_structure(atoms)
        cc_struct.coords[:, :] -= cc_struct.coords[0, :]
        #cc_struct.fix_coords_in_unit_cell()

        # Initialize the supercell if not already done.
        # TODO: check what happens if the atoms are ordered in a different way.
        if self.fixed_supercell_structure is None:
            self.fix_supercell(cc_struct)
        else:
            # Check if the supercell is different
            if np.max(np.abs(cc_struct.unit_cell - self.fixed_supercell_structure.unit_cell)) > 1e-6:
                self.fix_supercell(cc_struct)        

        self.get_energy_forces(cc_struct)


    def fix_supercell(self, supercell_structure, max_cell_value = 100):
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
        self.reference_supercell = new_cell.copy()
        first_itau = new_cell.get_itau(self.centroids) - 1
        new_cell.change_unit_cell(supercell_structure.unit_cell)
        
        # Suffle the atomic order in the correct way

        #shuffle_itau = supercell_structure.get_itau(new_cell) - 1
        
        #new_cell.coords = new_cell.coords[shuffle_itau, :]
        #new_cell.atoms = [new_cell.atoms[x] for x in shuffle_itau]
        #self.reference_supercell.coords = self.reference_supercell.coords[shuffle_itau, :]
        #self.reference_supercell.atoms = [self.reference_supercell.atoms[x] for x in shuffle_itau]

        # Prepare a correct identification between the input structure and the current structure
        self.reference_id = new_cell.get_itau(supercell_structure) - 1

        # Check that the identification of the atoms worked
        good_itau = np.all( [new_cell.atoms[i] == supercell_structure.atoms[self.reference_id[i]] for i in range(new_cell.N_atoms)]) 
        
        if not good_itau or (len(np.unique(self.reference_id)) != new_cell.N_atoms):
            ERR = '''
Error while assigning the effective charges to atoms.
    try to initialize the calculator with a less distorted structure.
'''
            raise ValueError(ERR)

        self.fixed_supercell_structure = new_cell
        self.setup_nac_fc(new_supercell)













