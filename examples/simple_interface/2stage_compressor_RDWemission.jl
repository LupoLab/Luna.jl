#= In this example we simulate a first stage to compress 35 fs pulses to ca 7.5 fs.
We then optimise the pulse compression, plot the resulting compressed pulse,
and pass the compressed pulse on to a second simulation for UV RDW emission.
=#
using Luna

λ0 = 800e-9
τfwhm = 35e-15
energy = 1e-3

gas = :He

a_stg1 = 225e-6 # core radius of compressor stage
p_stg1 = 4 # gas pressure in compressor stage
flength_stg1 = 1.5 # fibre length of compressor stage

λlims_stg1 = (200e-9, 4e-6) # grid wavelength limits
trange_stg1 = 1e-12 # time grid size

modes = 4 # number of modes

compressor = prop_capillary(a_stg1, flength_stg1, gas, p_stg1; λ0, τfwhm, energy, λlims=λlims_stg1, trange=trange_stg1, modes)

##
# propagator function which propagates the field from the 1st to the 2nd stage
# this mutates its input as required for the LunaPulse
function prop!(Eω, grid)
    Fields.prop_mirror!(Eω, grid, :PC70, 10) # 10 chirped mirror bounces
    Fields.prop_material!(Eω, grid, :Air, 5, λ0) # 5 m of air path total
    Fields.prop_material!(Eω, grid, :SiO2, 2e-3, λ0) # 2 1-mm windows
    # optimise compression with silica insertion (i.e. wedges)
    _, Eωopt = Fields.optcomp_material(Eω, grid, :SiO2, λ0, -1e-2, 1e-2)
    Eω .= Eωopt
end

# non-mutating version of prop! as required for plotting
function prop(grid, Eω)
    Eωout = copy(Eω)
    prop!(Eωout, grid)
    Eωout
end

##
# Plot the compressed pulse
Plotting.time_1D(compressor; modes=:sum, propagate=prop)

##
# Parameters for 2nd stage
a_stg2 = 100e-6
p_stg2 = 1.75
flength_stg2 = 1.5
energy_stg2 = 125e-6

pulse = Pulses.LunaPulse(compressor; energy=energy_stg2, propagator=prop!)

λlims_stg2 = (130e-9, 4e-6)
trange_stg2 = 500e-15

rdw = prop_capillary(a_stg2, flength_stg2, gas, p_stg2; λ0, pulses=pulse, modes, λlims=λlims_stg2, trange=trange_stg2)

##
Plotting.prop_2D(rdw; modes=:sum)
Plotting.spec_1D(rdw; λrange=(130e-9, 1.3e-6))
Plotting.spec_1D(rdw; modes=:sum, λrange=(130e-9, 1.3e-6))