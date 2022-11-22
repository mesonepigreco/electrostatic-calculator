try
    using LinearAlgebra
catch
    using Pkg 
    Pkg.add("LinearAlgebra")
    using LinearAlgebra
end

# Enforce blas to be executed on a single thread
using LinearAlgebra.BLAS
LinearAlgebra.BLAS.set_num_threads(1)

@doc raw"""
    get_energy_forces(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T})


Compute the electrostatic energy and forces.
The input must be in Ha atomic units and the output will be in Ha atomic units.
"""
function get_energy_forces(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, reference_struct :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T}, η:: T, volume :: T) where {T <: AbstractFloat}

    n_atoms = size(atomic_positions, 1)
    n_ks = size(k_points, 1)


    δr = atomic_positions .- reference_struct
    δr_asr = sum(δr, dims = 1)[1,:] / n_atoms
    @views for i in 1:n_atoms
        δr[i, :] .-= δr_asr
    end

    energy = zero(Complex{T})  
    I = Complex{T}(1.0im)
    force = zeros(T, (n_atoms, 3))
    kk_matrix = zeros(T, (3,3))
    ZkkZ = zeros(T, (3*n_atoms, 3*n_atoms))
    ZkkZr = zeros(T, 3)
    δrⱼᵢ = zeros(T, 3)


    for kindex ∈ 1:n_ks
        k_vect = @view k_points[kindex, :]
        k² = k_vect' * k_vect
        kϵk = k_vect' * ϵ * k_vect
        
        mul!(kk_matrix, k_vect, k_vect', exp.( - η^2 * k² / 2) / kϵk, 0)

        ZkkZ .= Z' * kk_matrix * Z

        #println()
        #println("k point: $k_vect")
        #println("ZkkZ: $ZkkZ")

        for i ∈ 1:n_atoms
            for j ∈ 1:n_atoms
                @views δrⱼᵢ .= atomic_positions[j, :] - atomic_positions[i, :]

                exp_factor = exp(-I * (k_vect' * δrⱼᵢ))
                cos_factor = real(exp_factor + conj(exp_factor))
                sin_factor = real(I * (exp_factor - conj(exp_factor)))

                #println("i = $i, j = $j")
                #println("sin_factor = $sin_factor")
                #println("cos_factor = $cos_factor")
                #println("exp_factor = $exp_factor")
                #println("δr_i = $(δr[i, :]);  δr_j = $(δr[j, :])")

                @views mul!(ZkkZr, ZkkZ[3*(i-1)+1 : 3*(i-1)+3, 3*(j-1)+1 : 3*(j-1)+3], δr[j, :])
                rZkkZr = @views δr[i, :]' * ZkkZr

                energy -= rZkkZr * exp_factor

                #println("r_i ZkkZ r_j = $rZkkZr")

                force[i, :] .+= ZkkZr .* cos_factor
                force[i, :] .+= k_vect .* (rZkkZr * sin_factor) 

                #println("Current force:")
                #println("$force")
                #println()
            end
        end
    end

    # Apply the acoustic sum rule
    force_asr = sum(force, dims=1)[1,:] ./ n_atoms
    for i ∈ 1:n_atoms
        force[i, :] .-= force_asr
    end

    @assert isapprox(imag(energy), 0, atol = 1e-6)

    #println("total energy = $(real(energy))")
    #println("total force = ")
    #println("$force")

    # Get the total energy and forces
    energy = real(energy) * 4 * π / volume
    force .*= 4 * π / volume

    return energy, force
end