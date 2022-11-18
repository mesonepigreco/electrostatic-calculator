try
    using LinearAlgebra
catch
    using Pkg 
    Pkg.add("LinearAlgebra")
    using LinearAlgebra
end

@doc raw"""
    get_energy_forces(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T})


Compute the electrostatic energy and forces.
The input must be in Ha atomic units and the output will be in Ha atomic units.
"""
function get_energy_forces(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, reference_struct :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T}, η:: T, volume :: T) where {T <: AbstractFloat}

    n_atoms = size(atomic_positions, 1)
    n_ks = size(k_points, 1)


    δr = atomic_positions .- reference_struct
    δr_asr = sum(δr, dims = 1)
    @views for i in 1:n_atoms
        δr[i, :] .-= δr_asr
    end

    energy = zeros(Complex{T})  
    I = Complex{T}(1.0im)
    force = zeros(T, (n_atoms, 3))
    kk_matrix = zeros(T, (3,3))
    ZkkZ = zeros(T, (3*n_atoms, 3*n_atoms))
    ZkkZr = zeros(T, (3, 3))
    δrⱼᵢ = zeros(T, 3)

    for kindex ∈ 1:n_ks
        k_vect = @view k_points[kindex, :]
        k² = k_vect' * k_vect
        kϵk = k_vect' * ϵ * k_vect

        kk_matrix .= k_vect * k_vect'
        kk_matrix .*= exp.( - η^2 * k² / 2)
        kk_matrix ./= kϵk

        ZkkZ .= Z' * kk_matrix * Z

        for i ∈ 1:n_atoms
            for j ∈ 1:n_atoms
                @views δrⱼᵢ .= atomic_positions[j, :] - atomic_positions[i, :]

                exp_factor = exp(I * (k_vect' * δrⱼᵢ))
                cos_factor = real(exp_factor + conj(exp_factor))
                sin_factor = real(I * (exp_factor - conj(exp_factor)))

                @views ZkkZr .= ZkkZ[3*(i-1)+1 : 3*(i-1)+3, 3*(j-1)+1 : 3*(j-1)+3] * δr[j, :]

                energy -= @views ( δr[i, :]' * ZkkZr) * exp_factor

                force[i, :] .+= ZkkZr .* cos_factor
                force[i, :] .+= @views k_vect .* (( δr[i, :]' * ZkkZr) * sin_factor) 
            end
        end
    end

    # Apply the acoustic sum rule
    force_asr = sum(force, dims=1) ./ n_atoms
    for i ∈ n_atoms
        force[i, :] .-= force_asr
    end

    @assert isapprox(imag(energy), 0, atol = 1e-6)

    # Get the total energy and forces
    energy = real(energy) * 4 * π / volume
    force .*= 4 * π / volume

    return energy, force
end