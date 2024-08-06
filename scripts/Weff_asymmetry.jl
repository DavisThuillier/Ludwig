using Ludwig

function main(T::Real, n_ε::Int, n_θ::Int, α)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ; α = α)
    grid = mesh.patches
    ℓ = length(mesh.patches)
    e_max = α * T

    ζ  = MVector{6,Float64}(undef)
    η::SVector{6,Float64} = (Δε / 2.0) * [1.0, 0.0, 1.0, 0.0, -1.0, 0.0]

    i,j = rand(1:ℓ, 2)

    Fpp = vertex_pp
    Fpk = vertex_pk

    energies = Vector{Float64}(undef, 3)

    Kimj = 0
    Kjmi = 0
    Kijm = 0
    Kjim = 0
    
    for m in eachindex(grid)
        kijm = Ludwig.map_to_first_bz(grid[i].momentum + grid[j].momentum - grid[m].momentum)
        qimj = Ludwig.map_to_first_bz(grid[i].momentum - grid[j].momentum + grid[m].momentum)
        qjmi = Ludwig.map_to_first_bz(grid[j].momentum - grid[i].momentum + grid[m].momentum)

        for μ in eachindex(bands)
            energies[μ] = abs(bands[μ](kijm))
        end
        μijm = argmin(energies) # Band to which k4 belongs
        
        for μ in eachindex(bands)
            energies[μ] = abs(bands[μ](qimj))
        end
        μimj = argmin(energies) # Band to which k4 belongs

        for μ in eachindex(bands)
            energies[μ] = abs(bands[μ](qjmi))
        end
        μjmi = argmin(energies) # Band to which k4 belongs

        wij = Ludwig.Weff_squared_123(grid[i], grid[j], grid[m], Fpp, Fpk, kijm, μijm) 
        Kijm += Ludwig.Γabc!(ζ, η, grid[i], grid[j], grid[m], T, Δε, bands[μijm], kijm, e_max) * (1 - f0(grid[m].energy, T))# * wij

        wji = Ludwig.Weff_squared_123(grid[j], grid[i], grid[m], Fpp, Fpk, kijm, μijm) 
        Kjim += Ludwig.Γabc!(ζ, η, grid[j], grid[i], grid[m], T, Δε, bands[μijm], kijm, e_max) * (1 - f0(grid[m].energy, T))# * wji

        w123 = Ludwig.Weff_squared_123(grid[i], grid[m], grid[j], Fpp, Fpk, qimj, μimj)
        w124 = Ludwig.Weff_squared_124(grid[i], grid[m], grid[j], Fpp, Fpk, qimj, μimj)
        wij = w123 + w124

        Kimj += Ludwig.Γabc!(ζ, η, grid[i], grid[m], grid[j], T, Δε, bands[μimj], qimj, e_max) * f0(grid[m].energy, T) #* wij

        w123 = Ludwig.Weff_squared_123(grid[j], grid[m], grid[i], Fpp, Fpk, qjmi, μjmi)
        w124 = Ludwig.Weff_squared_124(grid[j], grid[m], grid[i], Fpp, Fpk, qjmi, μjmi)
        wji = w123 + w124

        Kjmi += Ludwig.Γabc!(ζ, η, grid[j], grid[m], grid[i], T, Δε, bands[μjmi], qjmi, e_max) * f0(grid[m].energy, T) #* wji

    end
    Kimj *= f0(grid[i].energy, T) * (1 - f0(grid[j].energy, T))
    Kjmi *= f0(grid[j].energy, T) * (1 - f0(grid[i].energy, T))

    @show Kimj / Kjmi
    @show Kijm / Kjim

end

T   = 12.0
n_ε = 20
n_θ = 60

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

main(T, n_ε, n_θ, 6.0)
