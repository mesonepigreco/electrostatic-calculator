import ase, ase.calculators
from ase.calculators.calculator import Calculator

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Methods
import cellconstructor.Phonons

import numpy as np
import sys, os
import warnings

import scipy, scipy.special
from typing import List, Union


__JULIA_EXT__ = False
JULIA_ERROR = """
Error in initializing the Julia extension.
    Please install Julia and the required modules with:

    $ pip install julia

    The code will run without the Julia extension, but it will be slower (10-100 times).

    Details on error: {}
"""
try:
    import julia, julia.Main

    # Install the missing packages if required
    julia.Main.include(os.path.join(os.path.dirname(__file__), "fast_calculator.jl"))
    __JULIA_EXT__ = True
except Exception as e:
    try:
        import julia
        try:
            from julia.api import Julia
            jl = Julia(compiled_modules=False)
            import julia.Main
            julia.Main.include(os.path.join(os.path.dirname(__file__),
                "fast_calculator.jl"))
            __JULIA_EXT__ = True
        except:
            # Install the required modules
            julia.install()
            try:
                julia.Main.include(os.path.join(os.path.dirname(__file__),
                    "fast_calculator.jl"))
                __JULIA_EXT__ = True
            except Exception as e:
                warnings.warn(JULIA_ERROR.format(e))
    except Exception as e:
        warnings.warn(JULIA_ERROR.format(e))

DEBUG = False


def is_julia_available():
    return __JULIA_EXT__


def convert_to_cc_structure(item):
    """Convert any structure into a Cellconstructor Structure."""
    if isinstance(item, ase.Atoms):
        ret = CC.Structure.Structure()
        ret.generate_from_ase_atoms(item)
        return ret
    else:
        return item


class CompositeCalculator(Calculator):
    """
    Create a calculator that combines two different calculators.

    This calculator adds the energy and forces of two or more different calculators.
    """

    def __init__(self, list_of_calculators: list, *args, **kwargs):
        """
        Initialize the calculator.

        :param list_of_calculators: The list of calculators to combine.
        """
        Calculator.__init__(self, *args, **kwargs)

        self.list_of_calculators = list_of_calculators

        # Setup the implemented properties that are in common between all calculators
        self.implemented_properties = []

        # Fill self.implemented_properties with the intersection
        # of the variable implemented_properties of each element in list_of_calculators
        for calc in self.list_of_calculators:
            if len(self.implemented_properties) == 0:
                self.implemented_properties = calc.implemented_properties
            else:
                self.implemented_properties = \
                    list(set(self.implemented_properties)
                         .intersection(calc.implemented_properties))

        self.implemented_properties = list(set(self.implemented_properties))

    def calculate(self, atoms=None, *args, **kwargs):
        """
        Calculate the energy and forces of the system.

        :param atoms: The atoms object to calculate.
        """
        # Call the calculate method of all the calculators
        super().calculate(atoms, *args, **kwargs)

        # Copy the atoms to avoid calculator override
        ss = CC.Structure.Structure()
        ss.generate_from_ase_atoms(atoms)
        tmp_atoms = ss.get_ase_atoms()

        for i, calc in enumerate(self.list_of_calculators):
            tmp_atoms.set_calculator(calc)
            if "energy" in self.implemented_properties:
                if i == 0:
                    self.results["energy"] = tmp_atoms.get_potential_energy()
                else:
                    self.results["energy"] += tmp_atoms.get_potential_energy()
            if "forces" in self.implemented_properties:
                if i == 0:
                    self.results["forces"] = tmp_atoms.get_forces()
                else:
                    self.results["forces"] += tmp_atoms.get_forces()
            if "stress" in self.implemented_properties:
                if i == 0:
                    self.results["stress"] = tmp_atoms.get_stress()
                else:
                    self.results["stress"] += tmp_atoms.get_stress()

        atoms.set_calculator(self)

BASIC_PROPERTIES = ["energy", "forces"]

class ElectrostaticCalculator(Calculator):
    """
    Calculator for long-range electrostatic interaction.

    Long range calculator.
    """
    def __init__(self, *args, **kwargs):
        Calculator.__init__(self, *args, **kwargs)

        self.eta = 6  # Default value is 6 Angstrom
        self.reference_structure = None 
        self.effective_charges = None
        self.work_charges = None  # The actually initialized effective charges
        self.dielectric_tensor = None
        self.reciprocal_vectors = None
        self.cutoff = 5  # Stop the sum when k > cutoff / eta
        self.kpoints = None
        self.julia_speedup = True  
        self.initialized = False
        self.implemented_properties = ["energy", "forces", "stress"]

        self.compute_stress = is_julia_available()

    def __setattr__(self, __name: str, __value) -> None:
        if __name in ["eta", "cutoff"]:
            self.initialized = False
        if __name == "compute_stress":
            if __value and not is_julia_available():
                raise ValueError("Julia is not available. Please install Julia and the required modules to evaluate the stress tensor")

            if __value:
                self.implemented_properties = BASIC_PROPERTIES + ["stress"]
            else:
                self.implemented_properties = BASIC_PROPERTIES
        return super().__setattr__(__name, __value)

    def init(self, reference_structure: CC.Structure.Structure,
             effective_charges: np.ndarray,
             dielectric_tensor: np.ndarray,
             unique_atom_element : str = None,
             supercell: tuple[int, int, int] = (1, 1, 1)) -> None:
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
            unique_atom_element : str
                The atomic name of the species used to identify the origin of the structure.
                There must be just one per cell
                If None, the first atom is used.
                TODO: find a better way!
        """

        self.uc_structure = reference_structure.copy()
        self.unique_atom_element = unique_atom_element
        if self.unique_atom_element is None:
            self.unique_atom_element = self.uc_structure.atoms[0]

        # Shift the structure to have the unique element at the center
        unique_index = -1
        for i in range(self.uc_structure.N_atoms):
            if self.uc_structure.atoms[i] == self.unique_atom_element:
                unique_index = i
                break

        if unique_index == -1:
            raise ValueError("The unique atom element was not found in the structure!")

        self.uc_structure.coords[:, :] -= np.tile(self.uc_structure.coords[unique_index, :],
                                                  (self.uc_structure.N_atoms, 1))

        self.uc_effective_charges = effective_charges.copy()

        self.reference_structure = self.uc_structure.generate_supercell(supercell)
        self.supercell = supercell

        n_atoms = self.reference_structure.N_atoms
        n_atoms_uc = self.uc_structure.N_atoms
        self.effective_charges = np.zeros((n_atoms, 3, 3), dtype=np.double)
        self.dielectric_tensor = dielectric_tensor.copy()

        for i in range(np.prod(supercell)):
            self.effective_charges[i * n_atoms_uc: (i+1) * n_atoms_uc, :, :] = effective_charges

        self.work_charges = np.zeros((3, 3*n_atoms), dtype=np.double)
        for i in range(3):
            self.work_charges[i, :] = self.effective_charges[:, i, :].ravel()

        self.reciprocal_vectors = self.reference_structure.get_reciprocal_vectors()
        #CC.Methods.get_reciprocal_vectors(self.reference_structure.unit_cell)

        self.init_kpoints()

    def setup_structure(self, target_structure: CC.Structure.Structure):
        """Set the effective_charge and reference structure."""
        new_target = target_structure.copy()

        if target_structure.N_atoms != self.uc_structure.N_atoms * np.prod(self.supercell):
            raise ValueError("The target structure has a different number of atoms! Check if the supercell is {}".format(self.supercell))

        # Shift the structure to have the unique element at the center
        unique_index = -1
        for i in range(new_target.N_atoms):
            if new_target.atoms[i] == self.unique_atom_element:
                unique_index = i
                break

        if unique_index == -1:
            raise ValueError("The unique atom element {} was not found in the structure!".format(self.unique_atom_element))

        new_target.coords[:, :] -= np.tile(new_target.coords[unique_index, :],
                                           (new_target.N_atoms, 1))

        uc_target_cell = target_structure.unit_cell.copy()
        for i in range(3):
            uc_target_cell[i, :] /= self.supercell[i]

        # Adjust the unit cell
        self.uc_structure.change_unit_cell(uc_target_cell)

        # Match the target structure
        #target_structure.get_itau(self.uc_structure) - 1

        target_cov = CC.Methods.covariant_coordinates(self.uc_structure.unit_cell,
                                                      new_target.coords)
        self_cov = CC.Methods.covariant_coordinates(self.uc_structure.unit_cell,
                                                    self.uc_structure.coords)

        itau = julia.Main.get_equivalent_atoms(self_cov,
                                               self.uc_structure.atoms,
                                               target_cov,
                                               new_target.atoms)
        itau -= 1  # Convert julia to python indexing

        # Assert that itau array of int contains each element the same number of times
        if not np.all(np.bincount(itau) == np.bincount(itau)[0]):
            new_target.save_scf("target_structure.scf")
            self.uc_structure.save_scf("reference_structure.scf")
            raise ValueError(f"The target structure does not match the reference structure.")

        self.reference_structure = new_target
        nat_sc = new_target.N_atoms
        for i in range(nat_sc):
            # Identify the correct vector
            delta_vector = [round(x)
                            for x in list(target_cov[i, :] - self_cov[itau[i], :])]

            self.reference_structure.coords[i, :] = \
                self.uc_structure.coords[itau[i], :] + \
                np.dot(delta_vector, self.uc_structure.unit_cell)

            # print("Atom:", target_structure.atoms[i], i, itau[i],
            #       delta_vector, target_cov[i, :], self_cov[itau[i], :])

            # Prepare also the effective charges
            self.work_charges[:, 3*i: 3*i+3] = \
                self.uc_effective_charges[itau[i], :, :]

    def init_kpoints(self):
        r"""
        INITIALIZE THE K POINTS
        =======================

        Define the k points within the cutoff so that

        .. math::

            \left| \vec k \right| < \frac{C}{\eta}

        where :math:`C` is the cutoff and :math:`\eta` is the size of the gaussian charge core.
        
        We store the k vectors in Bohr^-1 to have a consistent units
        """

        # Initialize the sum over k
        max_values = [1 + int(x) for x in np.floor(.5 + self.cutoff / (self.eta * np.linalg.norm(self.reciprocal_vectors, axis = 1)))]
        
        self.kpoints = []

        for l in range(-max_values[0], max_values[0] + 1):
            for m in range(-max_values[1], max_values[1] + 1):
                for n in range(-max_values[2], max_values[2] + 1):
                    kvector = l * self.reciprocal_vectors[0, :] 
                    kvector += m * self.reciprocal_vectors[1, :] 
                    kvector += n * self.reciprocal_vectors[2, :] 

                    knorm = np.linalg.norm(kvector) 
                    if knorm < self.cutoff / self.eta and knorm > 1e-6:
                        self.kpoints.append(kvector / CC.Units.A_TO_BOHR)
        
        if len(self.kpoints) == 0:
            warnings.warn("WARNING, no k-points for the sum, the cell is too small to compute long-range interaction with eta = {}".format(self.eta))
            self.kpoints = np.zeros((0, 0), dtype=np.float64)
        else:
            self.kpoints = np.array(self.kpoints)

        self.energy = None
        self.force = None
        self.stress = None
        self.results = {}
        self.initialized = True

    def init_from_phonons(self, dynamical_matrix: CC.Phonons.Phonons,
                          unique_atom_element : str = None) -> None:
        """
        INITIALIZE THE CALCULATOR
        =========================

        Uses the reference of the dynamical matrix to initialize the phonons. 
        Everything is read from the dynamical matrix, included the supercell.

        see init documentation for more details


        Parameters
        ----------

        - dynamical_matrix: 
            the dynamical matrix of the system
        - unique_atom_element: 
            the string of unique atom element in the structure 
            (default: None, the first atom is used)

        """

        assert dynamical_matrix.effective_charges is not None, \
            "Error, the provided dynamical matrix has no effective charges"
        assert dynamical_matrix.dielectric_tensor is not None, \
            "Error, the provided dynamical matrix has no dielectric tensor"

        un = unique_atom_element
        if unique_atom_element is None:
            un = dynamical_matrix.structure.atoms[0]

        self.init(dynamical_matrix.structure,
                  dynamical_matrix.effective_charges,
                  dynamical_matrix.dielectric_tensor,
                  unique_atom_element=un,
                  supercell=dynamical_matrix.GetSupercell())

    def check_asr(self, threshold: float = 1e-6) -> None:
        """
        Check if the acoustic sum rule is enforced on the effective charges.

        This is very important to properly compute the forces on the structure.
        """
        nat = self.reference_structure.N_atoms
        for i in range(3):
            for j in range(3):
                asr_thr = np.sum(self.effective_charges[:, i, j])

                if asr_thr > threshold:
                    raise ValueError("Error, effective charge (atom {}, electric field {}) does not satisfy the ASR by {}".format(j, i, asr_thr))

        v = np.random.uniform(size = 3)
        total_shift = np.zeros(3)
        for i in range(nat):
            total_shift += self.effective_charges[i, :, :].dot(v)

        if np.linalg.norm(total_shift) > threshold:
            raise ValueError("Error, a translation is giving problems")

    def get_longrange_phonons(self, q_point: np.ndarray, struct: CC.Structure.Structure, convert_from_cc = True) -> np.ndarray:
        """
        PHONONS
        =======

        Use the dipole moment to return the nonanalitic phonons.
        This evaluation is correct in the limit eta -> oo 

        Parameters
        ----------
            q_point : ndarray
                The q vector at which the dynamical matrix is evaluated.
                By default accepts cellconstructor units (rad / A)
                To pass into Bohr^-1 in units of 2pi, use  convert_from_cc = False
            struct : CC.Structure.Structure
                The structure used to perform the calculation
            convert_from_cc : bool
                If true (default) use rad / A (the q points as they are stored into
                cellconstructor Phonons)
                Otherwhise, use the default units of this calculator (2pi / Bohr)

        Results
        -------
            force_constant_matrix :: ndarray(size = (3*nat, 3*nat), dtype = np.complex128)
                The force constant matrix at the provided q point.
                The result is in Ry/Bohr^2
        """
        if not self.initialized:
            raise ValueError("Error, calculator not initialized (must be redone after setting eta or cutoff)")


        if not __JULIA_EXT__:
            raise NotImplementedError("Error, subroutine get_longrange_phonons works only with julia available.")

    
        atomic_pos = struct.coords * CC.Units.A_TO_BOHR
        volume = struct.get_volume() * CC.Units.A_TO_BOHR**3

        new_q = np.copy(q_point)
        if convert_from_cc:
            new_q *= 2 * np.pi / CC.Units.A_TO_BOHR


        fc_q = julia.Main.get_phonons_q(new_q, 
            atomic_pos,
            self.reciprocal_vectors, 
            self.work_charges,
            self.dielectric_tensor,
            self.eta * CC.Units.A_TO_BOHR,
            np.double(self.cutoff),
            volume)

        fc_q *= 2 # Ha to Ry

        # Perform the division by the masses
        #_m_ = struct.get_masses_array()
        #sqrt_mass = np.sqrt(np.tile(_m_, (3,1)).T.ravel())
        #fc_q /= np.outer(sqrt_mass, sqrt_mass)

        return fc_q


    def get_supercell_fc(self, structure : CC.Structure.Structure, ase_units = False) -> np.ndarray:
        """
        SUPERCELL FC MATRIX
        ===================

        Return the force constant matrix of the given supercell structure.
        It works in the limit eta -> oo

        Results
        -------
            fc_matrix : np.ndarray(size = (3*nat, 3*nat), dtype = np.double)
                The force constant matrix in the supercell.
                It is not divided by the masses.
                Units are Ry/Bohr if ase_units is false,
                eV/A otherwise
        """

        if not self.initialized:
            raise ValueError("Error, calculator not initialized (must be redone after setting eta or cutoff)")


        if not __JULIA_EXT__:
            raise NotImplementedError("Error, subroutine get_longrange_phonons works only with julia available.")



        atomic_pos = structure.coords * CC.Units.A_TO_BOHR
        volume = structure.get_volume() * CC.Units.A_TO_BOHR**3

        fc = julia.Main.get_realspace_fc(self.kpoints, 
            atomic_pos,
            self.work_charges,
            self.dielectric_tensor,
            self.eta * CC.Units.A_TO_BOHR,
            volume)

        # Perform the ase conversion
        if ase_units:
            fc *= CC.Units.HA_TO_EV / CC.Units.BORH_TO_ANGSTROM**2
        else:
            fc *= 2 # Ha to Ry


        return fc

    def _get_energy_force(self, struct: CC.Structure.Structure) -> None:
        """
        Working function that evaluates the force and energy on the given configuration.

        The results are stored in self.energy and self.forces
        """
        if not self.initialized:
            raise ValueError("Error, calculator not initialized (must be redone after setting eta or cutoff)")

        self.setup_structure(struct)

        if __JULIA_EXT__ and self.julia_speedup:
            atomic_pos = struct.coords * CC.Units.A_TO_BOHR
            volume = struct.get_volume() * CC.Units.A_TO_BOHR**3
            ref_structure = self.reference_structure.coords * CC.Units.A_TO_BOHR
            stress = None

            if self.compute_stress:
                energy, force, stress = julia.Main.get_energy_forces_stress(self.kpoints,
                                                                            atomic_pos,
                                                                            ref_structure,
                                                                            self.work_charges,
                                                                            self.dielectric_tensor,
                                                                            self.eta * CC.Units.A_TO_BOHR,
                                                                            volume)
            else:
                energy, force = julia.Main.get_energy_forces(self.kpoints, 
                                                             atomic_pos,
                                                             ref_structure,
                                                             self.work_charges,
                                                             self.dielectric_tensor,
                                                             self.eta * CC.Units.A_TO_BOHR,
                                                             volume)

            energy *= CC.Units.HA_TO_EV
            force *= CC.Units.HA_TO_EV / CC.Units.BOHR_TO_ANGSTROM
            stress *= CC.Units.HA_TO_EV / CC.Units.BOHR_TO_ANGSTROM**3
            self.energy = energy
            self.force = force
            self.stress = stress
            return

        # Fallback to the slow python code
        n_atoms = self.reference_structure.N_atoms

        delta_r = struct.coords - self.reference_structure.coords

        # Remove the global translations (ASR)
        asr_delta_r = np.mean(delta_r, axis=0)
        delta_r[:, :] -= asr_delta_r
        delta_r *= CC.Units.A_TO_BOHR

        n_kpoints = self.kpoints.shape[0]
        volume = struct.get_volume() * CC.Units.A_TO_BOHR**3

        energy = 0 + 0j
        force = np.zeros_like(struct.coords)

        if DEBUG:
            print("Energy calculation:")
            print("-------------------")

        for kindex in range(n_kpoints):
            kvect = self.kpoints[kindex, :]

            # Discard Gamma
            if np.linalg.norm(kvect) < 1e-6:
                continue

            k2 = kvect.dot(kvect)
            
            k_eps_k = kvect.dot(self.dielectric_tensor .dot(kvect))
            kk_matrix = np.outer(kvect, kvect) * np.exp(-(self.eta * CC.Units.A_TO_BOHR)**2 * k2 / 2) / k_eps_k

            #ZkkZ = np.einsum("ai, bj, ab -> ij", self.work_charges, self.work_charges, kk_matrix)
            ZkkZ = self.work_charges.T.dot(kk_matrix.dot(self.work_charges))
            
            if DEBUG:
                print()
                print("k point: ", kvect)
                #print("ZkkZ:", ZkkZ)


            for i in range(n_atoms):
                for j in range(n_atoms):
                    #if i == j:
                    #    continue
                    delta_rij = struct.coords[j, :] - struct.coords[i, :]
                    delta_rij *= CC.Units.A_TO_BOHR


                    exp_factor = np.exp(np.complex128(-1j)* kvect.dot(delta_rij)) / 2.
                    cos_factor = np.real(np.complex128(exp_factor + np.conj(exp_factor)))
                    sin_factor = np.real(np.complex128(1j) * (exp_factor - np.conj(exp_factor)))

                    ZkkZr = ZkkZ[3*i:3*i+3, 3*j:3*j+3].dot(delta_r[j, :])

                    energy += delta_r[i, :].dot(ZkkZr) * exp_factor 

                    #print("Energy k:",  delta_r[i, :].dot(ZkkZr) * exp_factor )
                    #print("Force k:", ZkkZr * cos_factor + delta_r[i, :] * ZkkZr  * kvect *  sin_factor)
                    if np.isnan(energy):
                        print("Error, energy is NaN")
                        print("i: {};  j: {}".format(i, j))
                        print("exp: ", exp_factor)
                        print("k: ", kvect)
                        print("delta R_ij: ", struct.coords[j, :] - struct.coords[i, :])
                        raise ValueError("Error, energy is NaN")


                    if DEBUG:
                        print()
                        print("i = {}; j = {}; k = {};".format(i, j, kvect))
                        print("tr(ZkkZ_exp) = ", np.einsum("aa", ZkkZ[3*i:3*i+3, 3*j:3*j+3] * exp_factor))
                        print("tr(d/dx ZkkZ_exp) = ", np.einsum("aa", ZkkZ[3*i:3*i+3, 3*j:3*j+3]* sin_factor) * kvect / CC.Units.BOHR_TO_ANGSTROM) 
                        print("delta r_i = ", delta_r[i, :], "delta r_j = ", delta_r[j, :])
                        print("delta_rij = ", delta_rij)
                        print("sin_factor = ", sin_factor)
                        print("cos_factor = ", cos_factor)
                        print("exp_factor = ", exp_factor)
                        print("ZkkZ r_j = ", ZkkZr)
                        print("r_i ZkkZ r_j = ", delta_r[i, :].dot(ZkkZr))

                    force[i, :] -=  ZkkZr * cos_factor
                    force[i, :] -= delta_r[i, :].dot(ZkkZr)  * kvect *  sin_factor

                    if DEBUG:
                        print("Current force:")
                        print(force) 
                        print()

        assert np.imag(energy) < 1e-6, "Error, the energy has an imaginary part: {}".format(energy)
        energy = np.real(energy)

        # Remove from the forces the global translations
        force[:,:] -= np.mean(force, axis = 0) 

        if DEBUG:
            print()
            print("Total Energy: {}  | Total force:".format(energy))
            print(force)
            print()
            print()

        self.energy = energy * 4 * np.pi / volume * CC.Units.HA_TO_EV
        self.force = force *  4 * np.pi / volume * CC.Units.HA_TO_EV / CC.Units.BOHR_TO_ANGSTROM


    def calculate(self, atoms=None, *args, **kwargs):
        """
        The actual function called by the ASE calculator
        """
        super().calculate(atoms, *args, **kwargs)
        self.atoms = atoms

        if self.kpoints is None:
            raise ValueError("Error, calculator not initialized.")

        cc_struct = convert_to_cc_structure(atoms)

        # perform the actual calculation
        self._get_energy_force(cc_struct)


        self.results["energy"] = self.energy
        self.results["forces"] = self.force

        if self.compute_stress:
            self.results["stress"] = self.stress

        #self.results["dipole"] = self.get_dipole() 


