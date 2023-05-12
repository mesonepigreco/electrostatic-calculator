# Load the correct modules even if not installed at the first startup of the code
try
    using LinearAlgebra
    using LoopVectorization
catch
    using Pkg 
    Pkg.add("LinearAlgebra")
    Pkg.add("LoopVectorization")
    using LinearAlgebra
    using LoopVectorization
end

# Enforce blas to be executed on a single thread
using LinearAlgebra.BLAS
LinearAlgebra.BLAS.set_num_threads(1)


@doc raw"""
    get_phonons_lowq(q_point :: Vector{T}, coords :: Matrix{T}, reciprocal_vectors :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T}, η :: T, volume :: T) where {T <: AbstractFloat}


Return the phonons using the limit q → 0 and the harmonic part (no dependence on the position).
This is good for testing purpouses and eventually also a simple way to define the forces, to allow for a faster calculation.

This subroutine implements the equation (not correct, there is an exponential)
```math
D^{ij}_{\alpha\beta}(q) = \frac{1}{\Omega} \sum_{\substack{k\mu\nu\\ k = q + G}} \fracPk_\nu k_\mu Z_{j\nu\beta} Z_{i\mu\alpha} e^{-\frac{\eta^2 k^2}{2}}}{\sum_{\alpha\beta} k_\alpha\epsilon_{\alpha\beta}k_\beta }
```

Note that the results dynamical matrix is of complex type.
The coordinates are in the primitive cell
"""
function get_phonons_q(q_point :: Vector{T}, coords :: Matrix{T}, reciprocal_vectors :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T}, η :: T, cutoff :: T, volume :: T) :: Matrix{Complex{T}} where {T <: AbstractFloat}

    # Loop over the k points 
    max_value = zeros(Int, 3)
    n_atoms = size(coords, 1)

    # Extract the bonds of the loop
    for i ∈ 1:3
        max_value[i] = @views 1 + Int(floor(.5 + cutoff / (η * LinearAlgebra.norm(reciprocal_vectors[i, :]))))
    end
    
    
    k_vect = zeros(T, 3)
    kk_matrix = zeros(T, (3,3))
    ZkkZ = zeros(Complex{T}, (3*n_atoms, 3*n_atoms))
    I = Complex{T}(1.0im)
    output_dyn = zeros(Complex{T}, (3* n_atoms, 3*n_atoms))
    for l ∈ -max_value[1] : max_value[1]
        for m ∈ -max_value[2] : max_value[2]
            for n ∈ -max_value[3] : max_value[3]
                k_vect .= l .* @view reciprocal_vectors[1, :] 
                k_vect .+= m .* @view reciprocal_vectors[2, :] 
                k_vect .+= n .* @view reciprocal_vectors[3, :] 
                k_vect .+= q_point

                # Discard the k vector if not relevant
                k_norm = LinearAlgebra.norm(k_vect)
                if k_norm < cutoff / η && k_norm > 1e-6
                    # Initialize the variables for the standard numerator
                    k² = k_vect' * k_vect
                    kϵk = k_vect' * ϵ * k_vect
                    mul!(kk_matrix, k_vect, k_vect', exp.( - η^2 * k² / 2) / kϵk, 0)
                    ZkkZ .= Z' * kk_matrix * Z


                    for i ∈ 1:n_atoms
                        for j ∈ 1:n_atoms
                            δRᵢⱼ = @views coords[i, :] - coords[j, :]
                            exp_factor = exp(I * (k_vect' * δRᵢⱼ))/ 2
                            ZkkZ[3*(i-1)+1 : 3*(i-1)+3, 3*(j-1)+1 : 3*(j-1)+3] .*= exp_factor
                        end
                    end
                    output_dyn .+= ZkkZ
                end

                # Go to the other direction
                k_vect .-= q_point
                k_vect .-= q_point

                # Discard the k vector if not relevant
                k_norm = LinearAlgebra.norm(k_vect)
                if k_norm < cutoff / η && k_norm > 1e-6
                    # Initialize the variables for the standard numerator
                    k² = k_vect' * k_vect
                    kϵk = k_vect' * ϵ * k_vect
                    mul!(kk_matrix, k_vect, k_vect', exp.( - η^2 * k² / 2) / kϵk, 0)
                    ZkkZ .= Z' * kk_matrix * Z

                    for i ∈ 1:n_atoms
                        for j ∈ 1:n_atoms
                            δRᵢⱼ = @views coords[i, :] - coords[j, :]
                            exp_factor = exp(-I * (k_vect' * δRᵢⱼ))/ 2
                            ZkkZ[3*(i-1)+1 : 3*(i-1)+3, 3*(j-1)+1 : 3*(j-1)+3] .*= exp_factor
                        end
                    end
                    output_dyn .+= ZkkZ
                end
            end
        end
    end

    return output_dyn .* (4 * π / volume)
end 


@doc raw"""
    get_realspace_fc(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T}, η :: T, volume :: T) :: Matrix{T} where {T <: AbstractFloat}



Return the real space force constant matrix for a specific structure, 
only keeping the simplest term which does not mix the position.
"""
function get_realspace_fc(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T}, η :: T, volume :: T) :: Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(atomic_positions, 1)
    n_ks = size(k_points, 1)


    I = Complex{T}(1.0im)
    kk_matrix = zeros(T, (3,3))
    ZkkZ = zeros(T, (3*n_atoms, 3*n_atoms))
    δrⱼᵢ = zeros(T, 3)


    fc_matrix = zeros(T, (3*n_atoms, 3*n_atoms))

    for kindex ∈ 1:n_ks
        k_vect = @view k_points[kindex, :]
        k² = k_vect' * k_vect
        kϵk = k_vect' * ϵ * k_vect
        
        mul!(kk_matrix, k_vect, k_vect', exp.( - η^2 * k² / 2) / kϵk, 0)

        ZkkZ .= Z' * kk_matrix * Z

        for i ∈ 1:n_atoms
            for j ∈ 1:n_atoms   #SPEEDUP, possibly a factor two running on j > i (but not compatible with turbo yet)
                δrⱼᵢ .= atomic_positions[j, :] 
                δrⱼᵢ .-= atomic_positions[i, :]

                exp_factor = exp(-I * (k_vect' * δrⱼᵢ))
                cos_factor = real(exp_factor + conj(exp_factor)) / 2

                
                ZkkZ[3*(i-1)+1 : 3*(i-1)+3, 3*(j-1)+1 : 3*(j-1)+3] .*= cos_factor
            end
        end
        fc_matrix .+= ZkkZ
    end

    return fc_matrix .* (4 * π / volume)
end



@doc raw"""
    get_energy_forces(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T})


Compute the electrostatic energy and forces.
The input must be in Ha atomic units and the output will be in Ha atomic units.

It returns energy, forces and stress tensor.
"""
get_energy_forces(k_points :: Matrix{T}, atomic_positions :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T})
    nat = shape(atomic_positions, 1)
    forces = zeros(T, (nat, 3))
    stress = zeros(T, (3,3))

    energy = get_energy_forces!(forces, stress,
        k_points, atomic_positions, Z, ϵ)

    return energy, forces, stress
end



@doc raw"""
    get_energy_forces!(forces:: Matrix{T}, stress :: Matrix{T}, k_points :: Matrix{T}, atomic_positions :: Matrix{T}, Z :: Matrix{T}, ϵ :: Matrix{T})


Compute the electrostatic energy, forces and stress tensor.
The input must be in Ha atomic units and the output will be in Ha atomic units.

It returns the energy as a value, while forces and stress are computed inplace.
"""
function get_energy_forces!(force :: Matrix{T}, stress_tensor :: Matrix{T}, 
    k_points :: Matrix{T}, atomic_positions :: Matrix{T}, 
    reference_struct :: Matrix{T}, Z :: Matrix{T}, 
    ϵ :: Matrix{T}, η:: T, volume :: T) :: T where {T <: AbstractFloat}

    n_atoms = size(atomic_positions, 1)
    n_ks = size(k_points, 1)


    δr = atomic_positions .- reference_struct
    δr_asr = sum(δr, dims = 1)[1,:] / n_atoms
    @views for i in 1:n_atoms
        δr[i, :] .-= δr_asr
    end

    energy = zero(Complex{T})  
    I = Complex{T}(1.0im)
    kk_matrix = zeros(T, (3,3))
    ZkkZ = zeros(T, (3*n_atoms, 3*n_atoms))
    ZkkZr = zeros(T, 3)
    ZrrZ = zeros(T, (3,3))
    rr_mat = zeros(T, (3,3))
    δrⱼᵢ = zeros(T, 3)

    # Clean force and stress tensor
    force .= 0
    stress_tensor .= 0


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
            for j ∈ 1:n_atoms   #SPEEDUP, possibly a factor two running on j > i (but not compatible with turbo yet)
                @views δrⱼᵢ .= atomic_positions[j, :] - atomic_positions[i, :]

                exp_factor = exp(-I * (k_vect' * δrⱼᵢ)) / 2
                cos_factor = real(exp_factor + conj(exp_factor))
                sin_factor = real(I * (exp_factor - conj(exp_factor)))

                #println("i = $i, j = $j")
                #println("sin_factor = $sin_factor")
                #println("cos_factor = $cos_factor")
                #println("exp_factor = $exp_factor")
                #println("δr_i = $(δr[i, :]);  δr_j = $(δr[j, :])")

                @views mul!(ZkkZr, ZkkZ[3*(i-1)+1 : 3*(i-1)+3, 3*(j-1)+1 : 3*(j-1)+3], δr[j, :])
                rZkkZr = @views δr[i, :]' * ZkkZr

                rr_mat .= δr * δr'

                @views ZrrZ .= Z[:, 3*(i-1)+1 : 3*(i-1)+3] * rr_mat * Z[:, 3*(j-1)+1 : 3*(j-1)+3]'

                energy += rZkkZr * exp_factor
                for α in 1:3
                    for β in 1:3
                        @views stress_tensor[β, α] .-= δr[i, α] * ZkkZr[β] / (volume) * cos_factor
                        @views stress_tensor[β, α] .+= ZrrZ[β, :]' * kk_matrix[:, α] * cos_factor
                        @views stress_tensor[β, α] .-= rZkkZr * cos_factor * k_vect[α] * k_vect[β] * η^2 / 2
                        @views stress_tensor[β, α] .-= (rZkkZr * k_vect[α]) .* (ϵ[β, :]' * k_vect) * cos_factor / kϵk
                    end
                end 

                #println("r_i ZkkZ r_j = $rZkkZr")

                force[i, :] .-= ZkkZr .* cos_factor
                force[i, :] .-= k_vect .* (rZkkZr * sin_factor) 

                #println("Current force:")
                #println("$force")
                #println()
            end
        end
    end

    # Prepare the calculation of the stress_tensor
    stress_tensor .+= stress_tensor'
    for α in 1:3
        stress_tensor[α,α] += real(energy) / volume
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

    # Complete the calculation of the stress tensor 
    stress_tensor .+= stress_tensor'
    for μ in 1:3
        stress_tensor[μ, μ] += energy 


    # Get the total energy and forces and stress
    energy = real(energy) * 4 * π / volume
    force .*= 4 * π / volume
    stress_tensor .*= 4 * π / volume


    return energy
end


@doc raw"""
    setup_effective_charges(reference_coords :: Matrix{T},
                            reference_types :: Vector{Int},
                            reference_eff_charges:: Array{T, 3},
                            current_coords :: Matrix{T},
                            current_types :: Vector{Int};) where {T <: AbstractFloat}


Setup the reference structure and effective charges to match the minimum distance.
reference coordinates must be provided in crystal coordinates


TODO: Not yet implemented
"""
function setup_effective_charges(reference_coords :: Matrix{T},
                                 reference_types :: Vector{Int},
                                 reference_eff_charges:: Array{T, 3},
                                 current_coords :: Matrix{T},
                                 current_types :: Vector{Int}; far_away :: Int = 3) where {T <: AbstractFloat}

    nat_sc = size(current_coords, 1)
    nat = size(reference_coords, 1)

    r_eq = zeros(T, 3)

    @assert floor(nat_sc /nat) ≈ nat_sc / nat

    # Get the maximum vector size
    for i ∈ 1:nat_sc
        # Get the equivalent coordinates in the unit cell
        @views r_eq .= current_coords[i, :]
        @views r_eq .-= floor.(current_coords[i, :])

        for j ∈ -far_away : far_away

        end
    end
end



@doc raw"""
    get_equivalent_atoms(reference_coords :: Matrix{T},
                         reference_types :: Vector{String},
                         current_coords :: Matrix{T},
                         current_types :: Vector{String}; far_away :: Int = 2) :: Vector{Int} where {T <: AbstractFloat}

Return the index of each atom in the current_coords which is equivalent to the reference_coords.
Both reference_coords and current_coords must be in crystal coordinates.
reference_types and current_types are the species of each atom.

The function returns a vector of indices of the equivalent atoms in the current_coords.
"""
function get_equivalent_atoms(reference_coords::Matrix{T},
                              reference_types::Vector{String},
                              current_coords::Matrix{T},
                              current_types::Vector{String}; far_away::Int=3) :: Vector{Int} where {T<:AbstractFloat}

    nat_sc = size(current_coords, 1)
    nat = size(reference_coords, 1)

    r_eq = zeros(T, 3)
    r_eq2 = zeros(T, 3)

    @assert floor(nat_sc / nat) ≈ nat_sc / nat

    distances = zeros(T, nat)

    itau = zeros(Int, nat_sc)

    # Get the maximum vector size
    for i ∈ 1:nat_sc
        # Get the equivalent coordinates in the unit cell
        @views r_eq .= current_coords[i, :]
        @views r_eq .-= floor.(current_coords[i, :])

        distances .= 1000
        for j ∈ 1:nat
            if current_types[i] != reference_types[j]
                continue
            end

            for h ∈ -far_away:far_away
                for k ∈ -far_away:far_away
                    for l ∈ -far_away:far_away
                        @views r_eq2 .= r_eq .+ [h, k, l]
                        my_dist = norm(r_eq2 - reference_coords[j, :])
                        distances[j] = min(distances[j], my_dist)
                    end
                end
            end
        end
        itau[i] = argmin(distances)
    end

    return itau
end
