Minimal Theory Guide
====================

This guide provides the essential physics background needed to use py-electrostatic effectively. For a complete theoretical treatment, see the paper by Monacelli and Marzari [1]_.

The Problem
-----------

In polar materials (like ferroelectrics), atoms carry effective charges that create long-range electrostatic interactions. These interactions decay slowly with distance (as 1/r³) and are crucial for:

* Phonon frequencies at long wavelengths
* Ferroelectric phase transitions
* Dielectric properties
* Accurate force fields

Born Effective Charges
----------------------

The **Born effective charge tensor** Z* describes how an atom's charge distribution responds to an electric field:

.. math::

   Z^*_{i,\alpha\beta} = \frac{\partial p_\alpha}{\partial r_{i,\beta}}

where:

* :math:`i` is the atom index
* :math:`\alpha, \beta` are Cartesian directions (x, y, z)
* :math:`p_\alpha` is the polarization in direction :math:`\alpha`
* :math:`r_{i,\beta}` is the position of atom :math:`i` in direction :math:`\beta`

**Key properties:**

* Z* is a 3×3 tensor for each atom (9 numbers)
* It can be computed from first principles (DFT) using codes like Quantum ESPRESSO
* The acoustic sum rule requires: :math:`\sum_i Z^*_{i,\alpha\beta} = 0` for each :math:`\alpha, \beta`

Dielectric Tensor
-----------------

The **high-frequency dielectric tensor** :math:`\varepsilon_\infty` describes how the material polarizes in response to an electric field when atoms are held fixed:

.. math::

   P_\alpha = \varepsilon_0 \sum_\beta (\varepsilon_{\infty,\alpha\beta} - \delta_{\alpha\beta}) E_\beta

This is typically a 3×3 symmetric matrix obtained from DFT calculations.

Ewald Summation
---------------

Computing long-range interactions in periodic systems requires special techniques. The **Ewald summation** splits the interaction into:

1. **Real-space part**: Short-range, converges quickly
2. **Reciprocal-space part**: Long-range, computed in Fourier space

py-electrostatic uses an optimized k-space summation with Gaussian charge screening.

Key Parameters
--------------

eta (Å)
~~~~~~~

The Gaussian screening parameter. Controls the balance between real-space and k-space convergence:

* **Smaller η**: More k-points needed, faster real-space convergence
* **Larger η**: Fewer k-points, slower real-space convergence
* **Typical values**: 0.5-1.0 Å for most materials

cutoff (dimensionless)
~~~~~~~~~~~~~~~~~~~~~~

Determines how many k-points to include:

* Include all k-points with :math:`|k| < \text{cutoff}/\eta`
* **Typical values**: 3-5
* Higher values = more accurate but slower

use_nufft
~~~~~~~~~

Enable Non-Uniform FFT for O(N²) scaling:

* **True** (default): Faster for large systems (>50 atoms)
* **False**: Standard O(N³) method, good for small systems

Where to Get Parameters
-----------------------

Born charges and dielectric tensors are typically computed with:

* **Quantum ESPRESSO**: Use ``ph.x`` with ``epsil=.true.``
* **VASP**: Set ``LEPSILON = .TRUE.``
* **ABINIT**: Use ``rfelfd 1``

These are saved in dynamical matrix files that CellConstructor can read.

References
----------

.. [1] L. Monacelli and N. Marzari, *Electrostatic interactions in atomistic and machine-learned potentials for polar materials*, Phys. Rev. B 107, 245427 (2023).
