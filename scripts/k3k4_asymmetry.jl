using Ludwig
using StaticArrays
using LinearAlgebra
using CairoMakie

function main(T::Real, n_ε::Int, n_θ::Int, α)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ; α = α)
    grid = mesh.patches
    ℓ = length(mesh.patches)
    e_max = α * T

    N = 101
    kx = LinRange(-0.5, 0.5, N)
    uniform_grid = Vector{SVector{2, Float64}}(undef, N^2)
    for i in eachindex(kx)
        for j in eachindex(kx)
            uniform_grid[(i - 1) * N + j] = [kx[i], kx[j]]
        end
    end
    step = sqrt(2) / N
    @show step

    
    counter = 0
    while counter == 0
        i,j = rand(1:ℓ, 2)

        
        kij = mesh.patches[i].momentum - mesh.patches[j].momentum
        eij = mesh.patches[i].energy - mesh.patches[j].energy
        w0 = f0(mesh.patches[i].energy, T) * (1 - f0(mesh.patches[j].energy, T))
        @show w0

        corner_ids = map(x -> x.corners, mesh.patches)

        quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
        for i in 1:size(corner_ids)[1]
            push!(quads, map(x -> mesh.corners[x], corner_ids[i]))
        end

        band_colors = [:red, :blue, :green]

        for m in eachindex(uniform_grid)
            for n in eachindex(uniform_grid)
                Δk = Ludwig.map_to_first_bz(kij + uniform_grid[m] - uniform_grid[n])
                if norm(Δk) < step
                    for (r, m_band) in enumerate(bands)
                        for (s, n_band) in enumerate(bands)
                            em = m_band(uniform_grid[m])
                            en = n_band(uniform_grid[n])
                            δ = eij + em - en
                            if abs(δ) < Δε
                                counter += 1
                                w = f0(m_band(uniform_grid[m]), T) * (1 - f0(n_band(uniform_grid[n]), T))
                                if w > w0 && (abs(em) > e_max || abs(en) > e_max)
                                    @show w, en / e_max, em / e_max
                                    f = Figure(size = (1000,1000))
                                    ax = Axis(f[1,1],
                                            aspect = 1.0,
                                            limits = (-0.5,0.5,-0.5,0.5),
                                    )

                                    
                                    poly!(ax, quads, color = map(x -> band_colors[x.band_index], mesh.patches))
                                    scatter!(ax, uniform_grid, color = :black, markersize = 4)
                                    
                                    xs = [0.0, 0.0, 0.0, 0.0]
                                    ys = [0.0, 0.0, 0.0, 0.0]
                                    us = map(x -> x[1], [mesh.patches[i].momentum, mesh.patches[j].momentum, uniform_grid[m], uniform_grid[n]])
                                    vs = map(x -> x[2], [mesh.patches[i].momentum, mesh.patches[j].momentum, uniform_grid[m], uniform_grid[n]])
                                    arrows!(ax, xs, ys, us, vs, lengthscale = 1, arrowsize = 0, linewidth = 2, linecolor = [:black, :black, band_colors[r], band_colors[s]], arrowcolor = [:black, :black, band_colors[r], band_colors[s]])
            
                                    
                                    # scatter!(ax, uniform_grid[n], color = :black)
                                    display(f)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
end

T   = 12.0
n_ε = 20
n_θ = 60

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

main(T, n_ε, n_θ, 6.0)