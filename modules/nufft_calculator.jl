"""
NUFFT-based electrostatic calculator
Uses Non-Uniform FFT to compute Fourier transforms of dipole moments efficiently.
This reduces the complexity from O(Nₖ×N²) to O(Nₖ×N) for the transform + O(Nₖ) for the convolution.
"""
module NUFFTCalculatorModule

using LinearAlgebra
using FFTW
using DiffResults
using ForwardDiff

export get_energy_forces_nufft, get_energy_forces_nufft_stress


@doc raw"""
    get_energy_forces_nufft(k_points, atomic_positions, reference_struct, Z, ε, η, volume)

Compute the electrostatic energy and forces using NUFFT.
This achieves O(Nₖ × N) complexity for the transform + O(Nₖ) for the convolution,
instead of O(Nₖ × N²) for the naive implementation.

Z format: (3, 3*n_atoms) where:
- Row a corresponds to electric field direction a
- Column 3*(i-1)+b corresponds to atom i, cartesian direction b

The formula is:
E = (4π/Ω) Σₖ K(k) · |Σᵢ (k·zᵢ) · exp(-ik·rᵢ)|² / 2

where zᵢ = Z · δrᵢ is the dipole moment (3-vector).
"""
function get_energy_forces_nufft(k_points, atomic_positions, reference_struct, Z, ε, η, volume)
    T = float(promote_type(eltype(k_points), eltype(atomic_positions), eltype(reference_struct), eltype(Z), eltype(ε), typeof(η)))
    return _nufft_impl(convert(Matrix{T}, k_points), 
                       convert(Matrix{T}, atomic_positions),
                       convert(Matrix{T}, reference_struct),
                       convert(Matrix{T}, Z),
                       convert(Matrix{T}, ε),
                       convert(T, η),
                       convert(T, volume))
end

function _nufft_impl(k_points::Matrix{T}, atomic_positions::Matrix{T}, 
                      reference_struct::Matrix{T}, Z::Matrix{T}, ε::Matrix{T},
                      η::T, volume::T) where {T}
    
    n_atoms = size(atomic_positions, 1)
    n_k = size(k_points, 1)
    I = Complex{T}(im)
    
    # Compute displacements δr = r - r_ref (in Bohr)
    δr = atomic_positions .- reference_struct
    
    # Apply acoustic sum rule (ASR) to displacements
    δr_asr = sum(δr, dims=1)[1, :] / n_atoms
    @views for i in 1:n_atoms
        δr[i, :] .-= δr_asr
    end
    
    # Precompute k·z for each atom (3 scalar products)
    k_dot_z = zeros(Complex{T}, n_atoms)
    
    energy = zero(T)
    force = zeros(T, n_atoms, 3)
    
    @inbounds for (i_k, k_vec) in enumerate(eachrow(k_points))
        k2 = k_vec[1]^2 + k_vec[2]^2 + k_vec[3]^2
        
        # Skip k = 0
        if k2 < 1e-10
            continue
        end
        
        # Compute k·z for each atom and the sum Σᵢ (k·zᵢ) exp(-ik·rᵢ)
        k_dot_z_sum = zero(Complex{T})
        
        for i_atom in 1:n_atoms
            # Compute k·z_i = Σ_a k[a] * Σ_b Z[a, 3*(i-1)+b] * δr[i, b]
            kdz = zero(T)
            for a in 1:3
                for b in 1:3
                    kdz += k_vec[a] * Z[a, 3*(i_atom-1)+b] * δr[i_atom, b]
                end
            end
            
            # Compute exp(-ik·r_i)
            phase = exp(-I * (k_vec[1] * atomic_positions[i_atom, 1] + 
                             k_vec[2] * atomic_positions[i_atom, 2] + 
                             k_vec[3] * atomic_positions[i_atom, 3]))
            
            k_dot_z[i_atom] = kdz
            k_dot_z_sum += kdz * phase
        end
        
        # Compute kernel K(k) = exp(-η²k²/2) / (k·ε·k)
        k_eps_k = k_vec[1] * (ε[1,1] * k_vec[1] + ε[1,2] * k_vec[2] + ε[1,3] * k_vec[3]) +
                  k_vec[2] * (ε[2,1] * k_vec[1] + ε[2,2] * k_vec[2] + ε[2,3] * k_vec[3]) +
                  k_vec[3] * (ε[3,1] * k_vec[1] + ε[3,2] * k_vec[2] + ε[3,3] * k_vec[3])
        
        K_k = exp(-η^2 * k2 / 2) / k_eps_k
        
        # Energy contribution: K(k) * |Σᵢ (k·zᵢ) exp(-ik·rᵢ)|² / 2
        energy_contrib = K_k * abs2(k_dot_z_sum) / 2
        energy += energy_contrib
        
        # Force contribution: F_i = -∂E/∂r_i
        conj_sum = conj(k_dot_z_sum)
        
        for i_atom in 1:n_atoms
            phase = exp(-I * (k_vec[1] * atomic_positions[i_atom, 1] + 
                             k_vec[2] * atomic_positions[i_atom, 2] + 
                             k_vec[3] * atomic_positions[i_atom, 3]))
            
            for beta in 1:3
                # d(k·z_i)/d(δr[i, beta]) = Σ_a k[a] * Z[a, 3*(i-1)+beta]
                d_kdz = k_vec[1] * Z[1, 3*(i_atom-1)+beta] +
                        k_vec[2] * Z[2, 3*(i_atom-1)+beta] +
                        k_vec[3] * Z[3, 3*(i_atom-1)+beta]
                
                d_phase = -I * k_vec[beta] * phase
                
                # d(Σⱼ (k·zⱼ) exp(-ik·rⱼ))/d(δr[i, beta]) = d_kdz * phase + k_dot_z[i_atom] * d_phase
                d_sum = d_kdz * phase + k_dot_z[i_atom] * d_phase
                
                # Force contribution: -K(k) * Re[conj_sum * d_sum]
                force[i_atom, beta] -= K_k * real(conj_sum * d_sum)
            end
        end
    end
    
    energy *= 4.0 * π / volume
    force .= force .* (4.0 * π / volume)
    
    # Apply acoustic sum rule to forces
    force_asr = sum(force, dims=1)[1, :] / n_atoms
    @views for i in 1:n_atoms
        force[i, :] .-= force_asr
    end
    
    return real(energy), force
end


function get_energy_forces_nufft_stress(k_points, atomic_positions, reference_struct, Z, ε, η, volume)
    T = float(promote_type(eltype(k_points), eltype(atomic_positions), eltype(reference_struct), eltype(Z), eltype(ε), typeof(η)))
    
    n_atoms = size(atomic_positions, 1)
    
    function aux_diff(ε_strain::Matrix{T}) where {T}
        strain_matrix = Matrix{T}(I, 3, 3) + ε_strain
        inverse_strain = Matrix{T}(I, 3, 3) - ε_strain
        factor = 1.0 + ε_strain[1, 1] + ε_strain[2, 2] + ε_strain[3, 3]
        
        # Convert all captured variables to type T and apply strain
        new_k_points = convert(Matrix{T}, k_points) * inverse_strain
        new_atomic_positions = convert(Matrix{T}, atomic_positions) * strain_matrix
        new_reference_struct = convert(Matrix{T}, reference_struct) * strain_matrix
        new_Z = convert(Matrix{T}, Z)
        new_ε = convert(Matrix{T}, ε)
        new_η = convert(T, η)
        new_volume = convert(T, volume) * factor
        
        energy, force = _nufft_impl(new_k_points, new_atomic_positions,
                                   new_reference_struct, new_Z, new_ε, new_η, new_volume)
        
        output = zeros(T, n_atoms * 3 + 1)
        output[1] = energy
        output[2:end] = vec(force)
        return output
    end
    
    value = zeros(T, n_atoms * 3 + 1)
    start_strain = zeros(T, 3, 3)
    jacob_results = DiffResults.JacobianResult(value, start_strain)
    jacob_results = ForwardDiff.jacobian!(jacob_results, aux_diff, start_strain)
    
    energy = DiffResults.value(jacob_results)[1]
    force = reshape(DiffResults.value(jacob_results)[2:end], n_atoms, 3)
    stress = reshape(DiffResults.jacobian(jacob_results)[1, :], 3, 3)
    stress *= -1.0 / volume
    
    return energy, force, stress
end

end  # module

# Export functions to Main module for Python access
get_energy_forces_nufft = NUFFTCalculatorModule.get_energy_forces_nufft
get_energy_forces_nufft_stress = NUFFTCalculatorModule.get_energy_forces_nufft_stress
