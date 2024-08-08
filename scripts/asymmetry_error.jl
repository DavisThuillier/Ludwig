using Ludwig
using StaticArrays
import LinearAlgebra: dot
using CairoMakie
using StatsBase

function main(T::Real, n_ε::Int, n_θ::Int, α)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ; α = α)
    ℓ = length(mesh.patches)

    scale = 1.0
    integration_mesh, _ = Ludwig.multiband_mesh(bands, orbital_weights, T, ceil(Int, scale * n_ε), n_θ; α = scale * α)

    vertex_model = ""
    # vertex_model = "Constant"

    if vertex_model == "Constant" 
        Fpp(p1, p2) = 1.0
        Fpk(p1, k, μ) = 1.0
        title = L"F_{\mathbf{k}_1\mathbf{k}_2}^{\mu_1\mu_2} = 1"
    else
        Fpp = vertex_pp
        Fpk = vertex_pk
        title = L"F_{\mathbf{k}_1\mathbf{k}_2}^{\mu_1\mu_2} = (W_{\mathbf{k}_1}^\dagger W_{\mathbf{k}_1})^{\mu_1\mu_2}"
    end

    N = 1000
    errors = Vector{Float64}(undef, N)
    counter = 0

    while true
        i,j = rand(1:ℓ, 2)

        Lij = Ludwig.electron_electron(mesh.patches, integration_mesh.patches, i, j, bands, Δε, T, Fpp, Fpk, mesh.n_bands, integration_mesh.α)

        Lji = Ludwig.electron_electron(mesh.patches, integration_mesh.patches, j, i, bands, Δε, T, Fpp, Fpk, mesh.n_bands, integration_mesh.α)

        counter += 1
        errors[counter] = abs( 2 * (Lij - Lji) / (Lij + Lji) )
        if mod(counter, 100) == 0
            @show counter
        end
        counter == N && break

    end

    ## Fit histogram of errors ##
    x_lower = 0.0
    x_upper = 1.0
    n_bins  = 50
    step    = (x_upper - x_lower) / n_bins 
    bins = LinRange(x_lower, x_upper, n_bins)

    h = StatsBase.fit(Histogram, errors, bins)
    h = StatsBase.normalize(h, mode = :pdf) 

    midpoints = ((h.edges[1] .+ circshift(h.edges[1], -1)) / 2.0)[1:end-1]
    mean = dot(midpoints, h.weights) * step

    # expmodel(t, p) = p[1] .* exp.(-p[1] .* t)
    # fit = LsqFit.curve_fit(expmodel, midpoints, h.weights, [0.1])
    println("Average Asymmetry Error: ", mean)

    f = Figure()
    ax = Axis(f[1,1], ylabel = "PDF", xlabel = "Asymmetry Error", title = title)
    # xlims!(ax, x_lower, x_upper)
    hist!(ax, errors, bins = bins, normalization = :pdf)
    # domain = LinRange(x_lower, x_upper, 100)
    # lines!(ax, domain, expmodel(domain, fit.param), color = :red)
    display(f)

end

T   = 12.0
n_ε = 12
n_θ = 38

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

main(T, n_ε, n_θ, 6)