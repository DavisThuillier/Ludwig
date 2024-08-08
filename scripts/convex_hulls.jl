using Ludwig
using StaticArrays
using LinearAlgebra
using CairoMakie
using StatsBase

function convexhull(S)
    P = Vector{SVector{2,Float64}}(undef, 0)
    hullpoint = S[1]
    for i in eachindex(S)
        push!(P, hullpoint)
        endpoint = S[1]

        for j in eachindex(S)
            Δe = endpoint - P[i]
            Δj = S[j] - P[i]
            if endpoint == hullpoint || Δj[1] * Δe[2] - Δj[2] * Δe[1] > 0
                endpoint = S[j] 
            end
        end

        endpoint == P[1] && break 
        hullpoint = endpoint
    end
    return P
end

function get_ebounds(p)
    return [p.energy - p.de / 2, p.energy + p.de / 2]
end

function centroid(S)
    center = zeros(Float64, 2)
    for s in S
        center += s
    end
    return center / length(S)
end

function diameter(S)
    d = 0.0
    
    for i in eachindex(S)
        i == 1 && continue
        distance = norm(S[i] - S[1])
        if distance > d
            d = distance
        end
    end
    return d
end

function main(T::Real, n_ε::Int, n_θ::Int, α)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ; α = α)
    ℓ = length(mesh.patches)
    corner_ids = map(x -> x.corners, mesh.patches)

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> mesh.corners[x], corner_ids[i]))
    end

    i, j = rand(1:ℓ, 2)

    Pi = map(x -> mesh.corners[x], mesh.patches[i].corners)
    Pj = map(x -> mesh.corners[x], mesh.patches[j].corners)
    circshift!(Pi, 1 - argmin(first.(Pi)))
    circshift!(Pj, 1 - argmin(first.(Pj)))

    S = Vector{SVector{2,Float64}}(undef, 16)
    for i in eachindex(Pi)
        for j in eachindex(Pj)
            S[4 * (i-1) + j] = Pi[i] + Pj[j] 
        end
    end
    
    Pij = convexhull(S)

    eij = mesh.patches[i].energy + mesh.patches[j].energy
    ijbounds = get_ebounds(mesh.patches[i]) + get_ebounds(mesh.patches[j]) 

    counter = 0
    f = Figure(size = (1000, 1000))
    ax = Axis(f[1,1], aspect = 1.0, limits = ((-0.5, 0.5), (-0.5, 0.5)))
    poly!(ax, quads, color = :gold, strokecolor = :black, strokewidth = 0.1)
    for m in eachindex(mesh.patches)
        # em = mesh.patches[m].energy
        # for n in m:ℓ
        #     emn = em + mesh.patches[n].energy
        #     if emn > ijbounds[1] && emn < ijbounds[2]
        #         counter += 1
        #     end
        # end
        scatter!(ax, Ludwig.map_to_first_bz(mesh.patches[i].momentum + mesh.patches[j].momentum - mesh.patches[m].momentum), color = :blue)

        Pm = map(x -> mesh.corners[x], mesh.patches[m].corners)
        circshift!(Pm, 1 - argmin(first.(Pm)))

        S = Vector{SVector{2,Float64}}(undef, 4 * length(Pij))
        for i in eachindex(Pij)
            for j in eachindex(Pm)
                S[4 * (i-1) + j] = Pij[i] - Pm[j] 
            end
        end

        Pijm = convexhull(S)
        kn = centroid(Pijm)

        energies = Vector{Float64}(undef, length(bands))
        for μ in eachindex(bands)
            energies[μ] = bands[μ](kn)
        end
        μn = argmin(abs.(energies))

        bounds = ijbounds - get_ebounds(mesh.patches[m]) .- energies[μn]      

        if sign(bounds[1]) != sign(bounds[2])
            println("In bounds.")
            counter += 1

            d = diameter(Pijm)

            
            poly!(ax, Ludwig.map_to_first_bz.(Pijm), color = :black)
            # display(f)
        end

           
    end
    
    display(f)
    @show counter / ℓ
    

end

T   = 12.0
n_ε = 12
n_θ = 38

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

main(T, n_ε, n_θ, 6.0)
