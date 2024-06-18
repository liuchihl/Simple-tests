using Pkg
pkg"add Oceananigans, CairoMakie"
using Oceananigans

# Environmental parameters
# Linear background stratification (in z)
N=.5
θ = 1 # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates


grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))



bottomimmerse = -0.25
grid_immerse = ImmersedBoundaryGrid(grid, GridFittedBottom(bottomimmerse)) 

buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -[ĝ...])

closure = ScalarDiffusivity(κ=.5)
# @inline constant_stratification(z, t, p) = p.N² * z
# b̄_field = BackgroundField(constant_stratification, parameters=(; N² = N^2))
@inline ẑ(z, ĝ) = z*ĝ[3]
@inline constant_stratification(z, t, p) = p.N² * ẑ(z, p.ĝ)
b̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

# with background field, gradient boundary condition needs to be specified to make buoyancy flux at the boundary =0
# value = -N^2 # 0
# b_immerse = ImmersedBoundaryCondition(bottom=GradientBoundaryCondition(value))
# b_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(value),
#         top=GradientBoundaryCondition(value),immersed=b_immerse);

# for tilted coordinate

# by default, Oceananigans sets the normal and cross boundaries to be 0 gradient
normal = 0    # normal slope 
cross = 0     # cross slope

# normal = -N^2*cos(θ)    # normal slope 
# cross = -N^2*sin(θ)     # cross slope

b_immerse = ImmersedBoundaryCondition(bottom=GradientBoundaryCondition(normal),
                    west = GradientBoundaryCondition(cross), east = GradientBoundaryCondition(-cross))
b_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(normal),
                    top = GradientBoundaryCondition(normal), immersed=b_immerse);


model = NonhydrostaticModel(; grid=grid_immerse, closure, tracers=:b,
        background_fields = (b = b̄_field,),
        boundary_conditions=(; b = b_bcs),
        buoyancy=buoyancy,
        ) 

# define a forcing function for background buoyancy diffusion
func(z, t, p, κ) = ∂z(κ * constant_stratification(z, t, p))
κ = diffusivity(model.closure, model.diffusivity_fields, Val(:b))
b_forcing = Forcing(func; parameters=(κ=κ,))        

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
# set_theme!(Theme(fontsize = 24, linewidth=3))

# fig = Figure()
# axis = (xlabel = "Temperature (ᵒC)", ylabel = "z")
# label = "t = 0"

z = znodes(model.tracers.b)
# T = interior(model.tracers.T, 1, 1, :)
# T̄ = model.background_fields.tracers.T
# T_total = T̄[:] + T # total buoyancy field


# Diagnostics
b = model.tracers.b
b̄ = model.background_fields.tracers.b
B_total = b̄ + b # total buoyancy field
custom_diags = (; B_total = B_total, b_mean = b̄)

# lines(T_total, z; label, axis)

# Time-scale for diffusion across a grid cell
min_Δz = minimum_zspacing(model.grid)
diffusion_time_scale = min_Δz^2 / model.closure.κ.b

simulation = Simulation(model, Δt = 0.1 * diffusion_time_scale, stop_iteration = 10000)

simulation.output_writers[:buoyancy] =
                    JLD2OutputWriter(model, merge(model.tracers,custom_diags),
                     filename = "one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_forcing.jld2",
                     schedule=IterationInterval(100),
                     overwrite_existing = true)


run!(simulation)

using Printf

# label = @sprintf("t = %.3f", model.clock.time)
# lines!(interior(model.tracers.T, 1, 1, :), z; label)
# axislegend()


# simulation.stop_iteration += 10000
# run!(simulation)

# file = jldopen("one_dimensional_diffusion_constN2.jld2")

# ig = file["timeseries/T_total"]
# ug = ig.underlying_grid
# ĝ = file["serialized/buoyancy"].gravity_unit_vector


B_timeseries = FieldTimeSeries("one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_forcing.jld2", "B_total")
b_timeseries = FieldTimeSeries("one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_forcing.jld2", "b")
b̄_timeseries = FieldTimeSeries("one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_forcing.jld2", "b_mean")
times = B_timeseries.times
# using JLD2
# T_total2 = jldopen("one_dimensional_diffusion_constN2.jld2","T_total")
n = Observable(1)

fig = Figure()
ax1 = Axis(fig[2, 1]; xlabel = "total buoyancy", ylabel = "z")
xlims!(ax1, -.1, .1)
B = @lift interior(B_timeseries[$n], 1, 1, :)
lines!(ax1,B, z)

ax2 = Axis(fig[2, 2]; xlabel = "buoyancy perturbation", yticklabelsvisible=false)
xlims!(ax2, -.1, .1)
b = @lift interior(b_timeseries[$n], 1, 1, :)
lines!(ax2,b, z)

# ax3 = Axis(fig[2, 3]; xlabel = "background buoyancy", yticklabelsvisible=false)
# xlims!(ax3, -.25, .25)
# b̄ = @lift interior(b̄_timeseries[$n], 1, 1, :)
# lines!(ax3,b̄, z)

label = @lift "t = " * string(round(times[$n], digits=3))
Label(fig[1, 1], label, tellwidth=false)

fig

frames = 1:length(times)

@info "Making an animation..."

record(fig, "one_dimensional_diffusion_withbackgroundfield_zerograd_noinitial_tilt_forcing.mp4", frames, framerate=24) do i
    n[] = i
end