using Pkg
pkg"add Oceananigans, CairoMakie"
using Oceananigans
using SpecialFunctions
using JLD2
# Environmental parameters
# Linear background stratification (in z)
N=.5
N_temporary=0.
θ = 0 # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates


grid = RectilinearGrid(size=32, z=(0, 1), topology=(Flat, Flat, Bounded))



bottomimmerse = 0
grid_immerse = ImmersedBoundaryGrid(grid, GridFittedBottom(bottomimmerse)) 

buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -[ĝ...])
# @inline κ(x, y, z) = 1000 * exp(z / 1)
closure = ScalarDiffusivity(κ=.5,)
# @inline constant_stratification(z, t, p) = p.N² * z
# b̄_field = BackgroundField(constant_stratification, parameters=(; N² = N^2))
@inline ẑ(z, ĝ) = z*ĝ[3]
@inline constant_stratification(z, t, p) = p.N² * ẑ(z, p.ĝ)
b̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

normal = -N^2*cos(θ)    # normal slope 
normal_top = 0.
cross = -N^2*sin(θ)     # cross slope

b_immerse = ImmersedBoundaryCondition(bottom=GradientBoundaryCondition(normal),
                    west = GradientBoundaryCondition(cross), east = GradientBoundaryCondition(-cross))
b_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(normal),
                    top = GradientBoundaryCondition(normal_top), immersed=b_immerse);


model = NonhydrostaticModel(; grid=grid_immerse, closure, tracers=:b,
        background_fields = (b = b̄_field,),
        boundary_conditions=(; b = b_bcs),
        buoyancy=buoyancy,
        ) 


# define a forcing function for background buoyancy diffusion
    #1) continuous function
    # @inline ẑ(z, ĝ) = z*ĝ[3]
    # # @inline background_stratification(z, t) = N^2 * ẑ(z, ĝ)
    # @inline background_stratification(z, p, t) = p.N^2 * ẑ(z, p.ĝ)
    # func(z, t, κ) = ∂z(κ * background_stratification(z, t))    # ∂z is not working
    # κ = diffusivity(model.closure, model.diffusivity_fields, Val(:b))
    # b_forcing = Forcing(func; parameters=κ,)        

    #2) discrete forcing
import Oceananigans.TurbulenceClosures
# κzᶠᶜᶜ = Oceananigans.TurbulenceClosures.κzᶠᶜᶜ
κzᶜᶜᶠ = Oceananigans.TurbulenceClosures.κzᶜᶜᶠ



using Oceananigans.Operators: ∂zᶠᶜᶠ, ℑxzᶠᵃᶜ, ∂zᶜᶜᶠ, ℑzᵃᵃᶜ, ℑxzᶠᵃᶜ

function b_forcing_func(i, j, k, grid, clock, model_fields)
    
    # [κN²cosθ](z+Δz/2) - [κN²cosθ](z-Δz/2)     
diffusive_flux = @inbounds (κzᶜᶜᶠ(i, j, k+1, grid, model.closure, model.diffusivity_fields, Val(:b), clock, model_fields) *
                    ℑzᵃᵃᶜ(i, j, k+1, grid, ∂zᶜᶜᶠ, model.background_fields.tracers.b) * cos(θ)) -
                    (κzᶜᶜᶠ(i, j, k, grid, model.closure, model.diffusivity_fields, Val(:b), clock, model_fields) *
                    ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, model.background_fields.tracers.b) * cos(θ))    
    # C,C,C ➡ C,C,F ➡ C,C,C
    return diffusive_flux
end

b_forcing = Forcing(b_forcing_func, discrete_form=true)

model = NonhydrostaticModel(; grid=grid_immerse, closure, tracers=:b,
        background_fields = (b = b̄_field,),
        boundary_conditions=(; b = b_bcs),
        buoyancy=buoyancy,
        forcing = (b = b_forcing,)
        ) 

width = 0.1
initial_buoyancy(z) = 0  # or exp(-z^2 / (2width^2))
set!(model, b=initial_buoyancy)
using CairoMakie

z = znodes(model.tracers.b)

# Diagnostics
b = model.tracers.b
b̄ = model.background_fields.tracers.b
B_total = b̄ + b # total buoyancy field
custom_diags = (; B_total = B_total, b_mean = b̄)

# lines(T_total, z; label, axis)

# Time-scale for diffusion across a grid cell
min_Δz = minimum_zspacing(model.grid)
diffusion_time_scale = min_Δz^2 / model.closure.κ.b

simulation = Simulation(model, Δt = 0.1 * diffusion_time_scale, stop_iteration = 500)

simulation.output_writers[:buoyancy] =
                    JLD2OutputWriter(model, merge(model.tracers,custom_diags),
                     filename = "one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_Nz32_forcing.jld2",
                     schedule=IterationInterval(1),
                     overwrite_existing = true)


run!(simulation)

## solve for analytical solution:
# general solution for the heat equation with no-flux BC and N²z as initial condition
N=.5;
κ = 0.5;

# b_anal(t) = N^2*√(4*κ*t) .* erf.(z / √(4*κ*t))
b_anal(t) = N^2*z .* (erf.(z / √(4*κ*t)) ) .+ 2*N^2*√(κ*t)/√π*exp.(-z.^2/(4*κ*t))

using Printf
filename = "one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_Nz32_forcing.jld2"
B_timeseries = FieldTimeSeries(filename, "B_total")
b_timeseries = FieldTimeSeries(filename, "b")
b̄_timeseries = FieldTimeSeries(filename, "b_mean")
times = B_timeseries.times
# b_diff = b_analytical - b_timeseries;
# using JLD2
# T_total2 = jldopen("one_dimensional_diffusion_constN2.jld2","T_total")
n = Observable(1)

fig = Figure()
ax1 = Axis(fig[2, 1]; xlabel = "total buoyancy B  ", ylabel = "z")
xlims!(ax1, -.1, .4)
B = @lift interior(B_timeseries[$n], 1, 1, :)
L1 = lines!(ax1,B, z) 
b_analytical = @lift b_anal(times[$n])
L2 = lines!(ax1, b_analytical, z, linestyle=:dash, color=:red)

axislegend(ax1,[L1, L2],
    [L"B_{num}",
     L"B_{an}"],
     position = :lt)

ax2 = Axis(fig[2, 2]; xlabel = "buoyancy perturbation b x 0.001", yticklabelsvisible=false)
xlims!(ax2, -2, 2)
b = @lift interior(b_timeseries[$n], 1, 1, :)*1e3
lines!(ax2,b, z)


ax3 = Axis(fig[2, 3]; xlabel = L"(B_{num}-B_{an})× 0.001", yticklabelsvisible=false)
xlims!(ax3, -0.5, 0.5)
b_diff = @lift (interior(B_timeseries[$n], 1, 1, :) -  b_anal(times[$n]))*1e3
lines!(ax3,b_diff, z)
# ax3 = Axis(fig[2, 3]; xlabel = "background buoyancy", yticklabelsvisible=false)
# xlims!(ax3, -.25, .25)
# b̄ = @lift interior(b̄_timeseries[$n], 1, 1, :)
# lines!(ax3,b̄, z)

label = @lift "t = " * string(round(times[$n], digits=3))
Label(fig[1, 1], label, tellwidth=false)

fig

frames = 1:length(times)

@info "Making an animation..."

record(fig, "one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_Nz32_forcing.mp4", frames, framerate=24) do i
    n[] = i
end