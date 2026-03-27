"""
Particle-Mesh Ewald (PME) electrostatic calculator.

Uses B-spline interpolation on a uniform mesh in fractional coordinates
and FFT convolution to compute electrostatic energy and forces.
This achieves O(N log N) scaling instead of O(Nk × N) for NUFFT or O(Nk × N²) for standard.

The energy formula is:
E = (4π/Ω) Σ_{G≠0} [exp(-η²G²/2) / (G·ε·G)] |Σᵢ (G·zᵢ) exp(-iG·rᵢ)|² / 2

In PME, we approximate ρ̃_a(G) = Σᵢ z_{i,a} exp(-iG·rᵢ) by placing z_{i,a}
on a mesh using B-splines and FFT-ing, with a B-spline structure factor
correction |B(G)|⁻² to compensate for the approximation.
"""
module PMECalculatorModule

using LinearAlgebra
using FFTW
using DiffResults
using ForwardDiff

export get_energy_forces_pme, get_energy_forces_pme_stress

# ============================================================
# B-spline utilities
# ============================================================

"""
    cardinal_bspline(u, p)

Evaluate the cardinal B-spline M_p(u) of order p.
Recursive definition:
  M_1(u) = 1 if 0 ≤ u < 1, else 0
  M_p(u) = u/(p-1) * M_{p-1}(u) + (p-u)/(p-1) * M_{p-1}(u-1)
"""
function cardinal_bspline(u::T, p::Int) where {T}
    if p == 1
        return (u >= zero(T) && u < one(T)) ? one(T) : zero(T)
    end
    pm1 = T(p - 1)
    return u / pm1 * cardinal_bspline(u, p - 1) + (T(p) - u) / pm1 * cardinal_bspline(u - one(T), p - 1)
end

"""
    cardinal_bspline_deriv(u, p)

Derivative of M_p: dM_p/du = M_{p-1}(u) - M_{p-1}(u-1)
"""
function cardinal_bspline_deriv(u::T, p::Int) where {T}
    return cardinal_bspline(u, p - 1) - cardinal_bspline(u - one(T), p - 1)
end

"""
    bspline_structure_factor_inv2(m, N, p)

Compute |B(m, N, p)|⁻² for the B-spline structure factor correction.
B(m) = exp(2πi(p-1)m/N) / Σ_{k=0}^{p-2} M_p(k+1) exp(2πi m k / N)

We return |B(m)|⁻² which corrects the PME approximation.
"""
function bspline_structure_factor_inv2(m::Int, N::Int, p::Int, ::Type{T}) where {T}
    if m == 0
        return one(T)
    end

    # Compute the denominator sum: Σ_{k=0}^{p-2} M_p(k+1) exp(2πi m k / N)
    denom = zero(Complex{T})
    for k in 0:(p - 2)
        Mp_val = cardinal_bspline(T(k + 1), p)
        denom += Mp_val * exp(Complex{T}(0, 2 * T(π) * m * k / N))
    end

    abs2_denom = abs2(denom)
    if abs2_denom < T(1e-30)
        return one(T)
    end

    return one(T) / abs2_denom
end

"""
    precompute_bspline_correction(mesh_sizes, spline_order, T)

Precompute the 3D B-spline structure factor correction array |B(m₁,m₂,m₃)|⁻².
This is the product of 1D corrections: |B₁(m₁)|⁻² × |B₂(m₂)|⁻² × |B₃(m₃)|⁻²
"""
function precompute_bspline_correction(mesh_sizes::Vector{Int}, p::Int, ::Type{T}) where {T}
    N1, N2, N3 = mesh_sizes

    # Precompute 1D corrections
    corr1 = zeros(T, N1)
    corr2 = zeros(T, N2)
    corr3 = zeros(T, N3)

    for i in 1:N1
        m = i - 1
        if m > N1 ÷ 2
            m -= N1
        end
        corr1[i] = bspline_structure_factor_inv2(m, N1, p, T)
    end
    for i in 1:N2
        m = i - 1
        if m > N2 ÷ 2
            m -= N2
        end
        corr2[i] = bspline_structure_factor_inv2(m, N2, p, T)
    end
    for i in 1:N3
        m = i - 1
        if m > N3 ÷ 2
            m -= N3
        end
        corr3[i] = bspline_structure_factor_inv2(m, N3, p, T)
    end

    # Build 3D array as outer product
    correction = zeros(T, N1, N2, N3)
    for i3 in 1:N3
        for i2 in 1:N2
            for i1 in 1:N1
                correction[i1, i2, i3] = corr1[i1] * corr2[i2] * corr3[i3]
            end
        end
    end

    return correction
end

# ============================================================
# Mesh size determination
# ============================================================

"""
    next_fft_size(n)

Round up to next efficient FFT size (product of small primes 2,3,5).
"""
function next_fft_size(n::Int)
    m = n
    while true
        k = m
        while k % 2 == 0; k ÷= 2; end
        while k % 3 == 0; k ÷= 3; end
        while k % 5 == 0; k ÷= 5; end
        if k == 1
            return m
        end
        m += 1
    end
end

"""
    compute_mesh_sizes(unit_cell, cutoff, eta)

Determine mesh sizes N₁,N₂,N₃ such that the mesh covers k-space up to cutoff/eta.
N_i ≥ 2·cutoff / (η · |b_i|) where b_i are reciprocal vectors.
"""
function compute_mesh_sizes(unit_cell::Matrix{T}, cutoff::T, eta::T) where {T}
    # Compute reciprocal vectors (row vectors): b = 2π (A^T)^{-1}
    # unit_cell rows are lattice vectors a₁, a₂, a₃
    recip = 2 * T(π) * inv(unit_cell')

    mesh_sizes = zeros(Int, 3)
    for i in 1:3
        b_norm = norm(recip[i, :])
        n_min = ceil(Int, 2 * cutoff / (eta * b_norm))
        mesh_sizes[i] = next_fft_size(max(n_min, 4))
    end

    return mesh_sizes
end

# ============================================================
# Core PME implementation
# ============================================================

"""
    _pme_impl(atomic_positions, reference_struct, unit_cell, Z, ε, η, volume, mesh_sizes, spline_order)

Core PME implementation. All inputs in atomic units (Bohr).

Parameters:
- atomic_positions: (n_atoms, 3) current positions
- reference_struct: (n_atoms, 3) reference positions
- unit_cell: (3, 3) lattice vectors as rows
- Z: (3, 3*n_atoms) Born effective charges
- ε: (3, 3) dielectric tensor
- η: Gaussian charge spread
- volume: cell volume
- mesh_sizes: [N₁, N₂, N₃] mesh dimensions
- spline_order: B-spline order p (typically 6)
"""
function _pme_impl(atomic_positions::Matrix{T}, reference_struct::Matrix{T},
                   unit_cell::Matrix{T}, Z::Matrix{T}, ε::Matrix{T},
                   η::T, volume::T, mesh_sizes::Vector{Int},
                   spline_order::Int) where {T}

    n_atoms = size(atomic_positions, 1)
    p = spline_order
    N1, N2, N3 = mesh_sizes

    # Compute displacements and apply ASR
    δr = atomic_positions .- reference_struct
    δr_asr = sum(δr, dims=1) ./ n_atoms
    for i in 1:n_atoms
        @views δr[i, :] .-= δr_asr[1, :]
    end

    # Compute dipole moments: z_i = Z_i · δr_i (3-vector for each atom)
    # Z format: Z[a, 3*(i-1)+b] → z_{i,a} = Σ_b Z[a, 3*(i-1)+b] * δr[i,b]
    dipoles = zeros(T, n_atoms, 3)
    for i in 1:n_atoms
        for a in 1:3
            for b in 1:3
                dipoles[i, a] += Z[a, 3*(i-1)+b] * δr[i, b]
            end
        end
    end

    # Convert to fractional coordinates: u = r * A⁻¹
    # unit_cell rows are lattice vectors, so r = u * A, hence u = r * A⁻¹
    inv_cell = inv(unit_cell)
    frac_coords = atomic_positions * inv_cell  # (n_atoms, 3) fractional coords

    # ========================================
    # Step 1: Charge assignment (B-spline spreading)
    # ========================================
    # Create 3 mesh fields for dipole components
    Q = zeros(T, N1, N2, N3, 3)  # Q[:,:,:,a] = mesh for dipole component a

    for i_atom in 1:n_atoms
        # Get fractional coordinate, wrap to [0, 1)
        u1 = frac_coords[i_atom, 1]
        u2 = frac_coords[i_atom, 2]
        u3 = frac_coords[i_atom, 3]

        # Grid point below atom (0-indexed): g = floor(u * N)
        g1 = floor(Int, u1 * N1)
        g2 = floor(Int, u2 * N2)
        g3 = floor(Int, u3 * N3)

        # Fractional distance in grid units: w ∈ [0, 1)
        w1 = u1 * N1 - g1
        w2 = u2 * N2 - g2
        w3 = u3 * N3 - g3

        # Spread using B-spline weights
        # M_p(w + k) at grid point (g - k) for k = 0, ..., p-1
        # Arguments w, w+1, ..., w+p-1 are all in [0, p) ⊂ support of M_p
        for k3 in 0:(p-1)
            bsp3 = cardinal_bspline(w3 + T(k3), p)
            idx3 = mod(g3 - k3, N3) + 1  # +1 for Julia 1-indexing
            for k2 in 0:(p-1)
                bsp2 = cardinal_bspline(w2 + T(k2), p)
                idx2 = mod(g2 - k2, N2) + 1
                for k1 in 0:(p-1)
                    bsp1 = cardinal_bspline(w1 + T(k1), p)
                    idx1 = mod(g1 - k1, N1) + 1
                    weight = bsp1 * bsp2 * bsp3
                    for a in 1:3
                        Q[idx1, idx2, idx3, a] += dipoles[i_atom, a] * weight
                    end
                end
            end
        end
    end

    # ========================================
    # Step 2: Forward FFT
    # ========================================
    Q_hat = zeros(Complex{T}, N1, N2, N3, 3)
    for a in 1:3
        Q_hat[:, :, :, a] = fft(Q[:, :, :, a])
    end

    # ========================================
    # Step 3: Multiply by kernel in k-space
    # ========================================
    # Reciprocal vectors: b_i (as rows) = 2π (A^T)^{-1}
    recip = 2 * T(π) * inv(unit_cell')

    # Precompute B-spline correction
    bsp_corr = precompute_bspline_correction(mesh_sizes, p, T)

    # Potential in k-space
    φ_hat = zeros(Complex{T}, N1, N2, N3, 3)

    # Energy accumulator
    energy = zero(T)

    for i3 in 1:N3
        m3 = i3 - 1
        if m3 > N3 ÷ 2; m3 -= N3; end
        for i2 in 1:N2
            m2 = i2 - 1
            if m2 > N2 ÷ 2; m2 -= N2; end
            for i1 in 1:N1
                m1 = i1 - 1
                if m1 > N1 ÷ 2; m1 -= N1; end

                # Skip G = 0
                if m1 == 0 && m2 == 0 && m3 == 0
                    continue
                end

                # G = m₁b₁ + m₂b₂ + m₃b₃
                G1 = m1 * recip[1, 1] + m2 * recip[2, 1] + m3 * recip[3, 1]
                G2 = m1 * recip[1, 2] + m2 * recip[2, 2] + m3 * recip[3, 2]
                G3 = m1 * recip[1, 3] + m2 * recip[2, 3] + m3 * recip[3, 3]

                G2_norm = G1^2 + G2^2 + G3^2

                # Apply cutoff: skip if |G| > cutoff/η
                # (This is important for convergence consistency with other methods)
                # We include all G vectors that the mesh provides - the Gaussian damping
                # handles convergence

                # Gaussian damping
                gauss = exp(-η^2 * G2_norm / 2)

                # G·ε·G
                GεG = G1 * (ε[1,1]*G1 + ε[1,2]*G2 + ε[1,3]*G3) +
                      G2 * (ε[2,1]*G1 + ε[2,2]*G2 + ε[2,3]*G3) +
                      G3 * (ε[3,1]*G1 + ε[3,2]*G2 + ε[3,3]*G3)

                if abs(GεG) < T(1e-30)
                    continue
                end

                # B-spline correction
                corr = bsp_corr[i1, i2, i3]

                # Kernel prefactor: 4π/Ω · exp(-η²G²/2) / (G·ε·G) · |B|⁻²
                prefactor = 4 * T(π) / volume * gauss / GεG * corr

                # K_{ab}(G) = prefactor * G_a * G_b
                # But we need to apply the full tensor kernel:
                # φ̃_a = Σ_b K_{ab} ρ̃_b = prefactor * G_a * (Σ_b G_b * ρ̃_b)

                G_vec = (G1, G2, G3)

                # Compute G · ρ̃ = Σ_b G_b * Q̂_b
                G_dot_rho = G_vec[1] * Q_hat[i1, i2, i3, 1] +
                            G_vec[2] * Q_hat[i1, i2, i3, 2] +
                            G_vec[3] * Q_hat[i1, i2, i3, 3]

                # Energy: (1/2) * prefactor * |G · ρ̃|²
                energy += prefactor * abs2(G_dot_rho) / 2

                # Potential: φ̃_a = prefactor * G_a * (G · ρ̃)
                for a in 1:3
                    φ_hat[i1, i2, i3, a] = prefactor * G_vec[a] * G_dot_rho
                end
            end
        end
    end

    # ========================================
    # Step 4: Inverse FFT to get real-space potential
    # ========================================
    φ = zeros(T, N1, N2, N3, 3)
    for a in 1:3
        φ[:, :, :, a] = real.(ifft(φ_hat[:, :, :, a]))
    end

    # ========================================
    # Step 5: Forces
    # ========================================
    # The force is F_{i,β} = -∂E/∂r_{i,β} = -N_total × Σ_n (∂Q_a(n)/∂r_{i,β}) × φ_a(n)
    # The N_total factor comes from: ∂E/∂Q_a(n) = N_total × φ_a(n) due to IFFT normalization
    N_total = T(N1 * N2 * N3)
    force = zeros(T, n_atoms, 3)

    for i_atom in 1:n_atoms
        u1 = frac_coords[i_atom, 1]
        u2 = frac_coords[i_atom, 2]
        u3 = frac_coords[i_atom, 3]

        g1 = floor(Int, u1 * N1)
        g2 = floor(Int, u2 * N2)
        g3 = floor(Int, u3 * N3)

        w1 = u1 * N1 - g1
        w2 = u2 * N2 - g2
        w3 = u3 * N3 - g3

        # Interpolate potential at atom position
        φ_at_atom = zeros(T, 3)

        # Gradient of potential: ∂φ_a/∂(u_γ·N_γ) (in grid units)
        # For the position derivative, we need ∂φ_a/∂r_{i,β}
        # Using chain rule: ∂/∂r_β = Σ_γ (∂(u_γ N_γ)/∂r_β) × ∂/∂(u_γ N_γ)
        # Since u_γ = Σ_β r_β (A⁻¹)_{β,γ}, we have ∂u_γ/∂r_β = (A⁻¹)_{β,γ}
        # And ∂(u_γ N_γ)/∂r_β = N_γ (A⁻¹)_{β,γ}

        dφ_dw = zeros(T, 3, 3)  # dφ_dw[a, γ] = ∂φ_a / ∂(u_γ·N_γ)

        for k3 in 0:(p-1)
            bsp3 = cardinal_bspline(w3 + T(k3), p)
            dbsp3 = cardinal_bspline_deriv(w3 + T(k3), p)
            idx3 = mod(g3 - k3, N3) + 1
            for k2 in 0:(p-1)
                bsp2 = cardinal_bspline(w2 + T(k2), p)
                dbsp2 = cardinal_bspline_deriv(w2 + T(k2), p)
                idx2 = mod(g2 - k2, N2) + 1
                for k1 in 0:(p-1)
                    bsp1 = cardinal_bspline(w1 + T(k1), p)
                    dbsp1 = cardinal_bspline_deriv(w1 + T(k1), p)
                    idx1 = mod(g1 - k1, N1) + 1

                    weight = bsp1 * bsp2 * bsp3

                    for a in 1:3
                        φ_val = φ[idx1, idx2, idx3, a]
                        φ_at_atom[a] += φ_val * weight

                        # B-spline derivative: dM(w+k)/dw at each grid point
                        # Note: negative sign because grid index g-k moves opposite to w
                        dφ_dw[a, 1] -= φ_val * dbsp1 * bsp2 * bsp3
                        dφ_dw[a, 2] -= φ_val * bsp1 * dbsp2 * bsp3
                        dφ_dw[a, 3] -= φ_val * bsp1 * bsp2 * dbsp3
                    end
                end
            end
        end

        # Force term 1: dipole derivative
        # F_{i,β}^(1) = -N_total × Σ_a Z_{a,β}^i × φ_a(r_i)
        for β in 1:3
            for a in 1:3
                force[i_atom, β] -= N_total * Z[a, 3*(i_atom-1)+β] * φ_at_atom[a]
            end
        end

        # Force term 2: position derivative
        # F_{i,β}^(2) = -N_total × Σ_a z_{i,a} × ∂φ_a/∂r_{i,β}
        # ∂φ_a/∂r_{i,β} = Σ_γ dφ_dw[a,γ] × N_γ × (A⁻¹)_{β,γ}
        for β in 1:3
            for a in 1:3
                dφ_dr = zero(T)
                for γ in 1:3
                    N_gamma = T([N1, N2, N3][γ])
                    dφ_dr += dφ_dw[a, γ] * N_gamma * inv_cell[β, γ]
                end
                force[i_atom, β] -= N_total * dipoles[i_atom, a] * dφ_dr
            end
        end
    end

    # Apply acoustic sum rule to forces
    force_asr = sum(force, dims=1) ./ n_atoms
    for i in 1:n_atoms
        @views force[i, :] .-= force_asr[1, :]
    end

    return energy, force
end

# ============================================================
# Public interface
# ============================================================

"""
    get_energy_forces_pme(atomic_positions, reference_struct, unit_cell, Z, ε, η, volume, mesh_sizes, spline_order)

Compute electrostatic energy and forces using Particle-Mesh Ewald.

All inputs in atomic units (Bohr, Hartree).
"""
function get_energy_forces_pme(atomic_positions, reference_struct, unit_cell,
                               Z, ε, η, volume, mesh_sizes, spline_order)
    T = Float64
    return _pme_impl(convert(Matrix{T}, atomic_positions),
                     convert(Matrix{T}, reference_struct),
                     convert(Matrix{T}, unit_cell),
                     convert(Matrix{T}, Z),
                     convert(Matrix{T}, ε),
                     convert(T, η),
                     convert(T, volume),
                     convert(Vector{Int}, mesh_sizes),
                     convert(Int, spline_order))
end

"""
    get_energy_forces_pme_stress(atomic_positions, reference_struct, unit_cell, Z, ε, η, volume, mesh_sizes, spline_order)

Same as get_energy_forces_pme but also computes the stress tensor via numerical differentiation.
Uses numerical strain derivatives because FFTW does not support ForwardDiff Dual types.
"""
function get_energy_forces_pme_stress(atomic_positions, reference_struct, unit_cell,
                                      Z, ε, η, volume, mesh_sizes, spline_order)
    T = Float64
    n_atoms = size(atomic_positions, 1)
    p = convert(Int, spline_order)
    ms = convert(Vector{Int}, mesh_sizes)

    ap = convert(Matrix{T}, atomic_positions)
    rs = convert(Matrix{T}, reference_struct)
    uc = convert(Matrix{T}, unit_cell)
    Zc = convert(Matrix{T}, Z)
    εc = convert(Matrix{T}, ε)
    ηc = convert(T, η)
    vc = convert(T, volume)

    # Compute energy and forces at zero strain
    energy, force = _pme_impl(ap, rs, uc, Zc, εc, ηc, vc, ms, p)

    # Compute stress via numerical strain derivatives
    δ = T(1e-5)
    stress = zeros(T, 3, 3)

    for i in 1:3
        for j in 1:3
            ε_strain = zeros(T, 3, 3)
            ε_strain[i, j] = δ

            strain_matrix = Matrix{T}(I, 3, 3) + ε_strain
            factor = one(T) + ε_strain[1, 1] + ε_strain[2, 2] + ε_strain[3, 3]

            new_ap = ap * strain_matrix
            new_rs = rs * strain_matrix
            new_uc = uc * strain_matrix
            new_volume = vc * factor

            e_plus, _ = _pme_impl(new_ap, new_rs, new_uc, Zc, εc, ηc, new_volume, ms, p)

            stress[i, j] = -(e_plus - energy) / (δ * vc)
        end
    end

    return energy, force, stress
end

end  # module

# Export functions to Main module for Python access
get_energy_forces_pme = PMECalculatorModule.get_energy_forces_pme
get_energy_forces_pme_stress = PMECalculatorModule.get_energy_forces_pme_stress
compute_mesh_sizes = PMECalculatorModule.compute_mesh_sizes
