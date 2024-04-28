using Pkg
pkg"add Oceananigans, CairoMakie"
using Oceananigans


grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))
closure = ScalarDiffusivity(κ=.5)
# Linear background stratification (in z)
N=.5
@inline constant_stratification(z, t, p) = p.N² * z
T̄_field = BackgroundField(constant_stratification, parameters=(; N² = N^2))

# with background field, gradient boundary condition needs to be specified to make buoyancy flux at the boundary =0
value = -N^2 # 0
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(value),
                                bottom = GradientBoundaryCondition(value));


model = NonhydrostaticModel(; grid, closure, tracers=:T,
        background_fields = (T=T̄_field,)
        ,boundary_conditions=(; T = T_bcs))
width = 0.1
initial_temperature(z) = 0  # or exp(-z^2 / (2width^2))
set!(model, T=initial_temperature)
using CairoMakie
# set_theme!(Theme(fontsize = 24, linewidth=3))

# fig = Figure()
# axis = (xlabel = "Temperature (ᵒC)", ylabel = "z")
# label = "t = 0"

z = znodes(model.tracers.T)
# T = interior(model.tracers.T, 1, 1, :)
# T̄ = model.background_fields.tracers.T
# T_total = T̄[:] + T # total buoyancy field


# Diagnostics
T = model.tracers.T
T̄ = model.background_fields.tracers.T
T_total = T̄ + T # total buoyancy field
custom_diags = (; T_total = T_total)

# lines(T_total, z; label, axis)

# Time-scale for diffusion across a grid cell
min_Δz = minimum_zspacing(model.grid)
diffusion_time_scale = min_Δz^2 / model.closure.κ.T

simulation = Simulation(model, Δt = 0.1 * diffusion_time_scale, stop_iteration = 10000)

simulation.output_writers[:temperature] =
                    JLD2OutputWriter(model, merge(model.tracers,custom_diags),
                     filename = "one_dimensional_diffusion_withbackgroundfield_nonzeroflux_noinitial.jld2",
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


T_timeseries = FieldTimeSeries("one_dimensional_diffusion_withbackgroundfield_nonzeroflux_noinitial.jld2", "T_total")
times = T_timeseries.times
# using JLD2
# T_total2 = jldopen("one_dimensional_diffusion_constN2.jld2","T_total")
fig = Figure()
ax = Axis(fig[2, 1]; xlabel = "Temperature (ᵒC)", ylabel = "z")
xlims!(ax, -.75, .75)

n = Observable(1)

T = @lift interior(T_timeseries[$n], 1, 1, :)
lines!(T, z)

label = @lift "t = " * string(round(times[$n], digits=3))
Label(fig[1, 1], label, tellwidth=false)

fig

frames = 1:length(times)

@info "Making an animation..."

record(fig, "one_dimensional_diffusion_withbackgroundfield_nonzeroflux_noinitial.mp4", frames, framerate=24) do i
    n[] = i
end