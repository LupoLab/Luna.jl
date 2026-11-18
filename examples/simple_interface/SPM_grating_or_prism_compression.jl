#= In this example we simulate single-stage SPM broadening of a Yb-laser-like pulse
(1030 nm, 300 fs, 600 µJ) in an argon-filled hollow capillary fibre, followed by
compression using either a four-grating compressor or a double-pass fused-silica
prism pair, and compare the two.

The fibre parameters (125 µm core radius, 1 m length, 4 bar Ar) give significant SPM
broadening, and the resulting chirp is approximately linear—well suited for grating
or prism compression.

The grating compressor uses 300 lines/mm gratings at Littrow angle (order m = -1).
The prism compressor uses Brewster-cut fused-silica prisms at minimum deviation. =#
using Luna, PyPlot

## -- Fibre parameters --
a = 125e-6      # core radius [m]
flength = 1.0   # fibre length [m]
gas = :Ar       # fill gas
pressure = 4.0  # gas pressure [bar]

## -- Pulse parameters --
λ0 = 1030e-9    # central wavelength [m]
τfwhm = 300e-15 # pulse duration FWHM [s]
energy = 600e-6 # pulse energy [J]

## -- Run the mode-averaged simulation --
output = prop_capillary(a, flength, gas, pressure;
    λ0, τfwhm, energy,
    λlims=(800e-9, 1300e-9), trange=2e-12)

## -- Grating compressor parameters --
Λ = 1/(300e3)             # grating period [m] (300 lines/mm)
m = -1                     # diffraction order
θi_g = Fields.littrow_angle(λ0, Λ; m)  # Littrow angle of incidence [rad]

## -- Prism compressor parameters --
material = :SiO2                              # fused silica
α_p = Fields.mindev_apex(material, λ0)        # apex angle for minimum deviation [rad]
θi_p = Fields.brewster_angle(material, λ0)    # Brewster angle of incidence [rad]

## -- Optimal compression: gratings --
function prop_gratings(grid, Eω)
    L_opt, Eωcomp = Fields.optcomp_gratings(Eω, grid, Λ, m, θi_g; λ0=λ0)
    @info "Optimal grating separation: $(round(L_opt*1000, digits=2)) mm"
    Eωcomp
end

## -- Optimal compression: double-pass prism pair --
function prop_prisms(grid, Eω)
    L_opt, Eωcomp = Fields.optcomp_prisms(Eω, grid, material, α_p, θi_p; λ0=λ0)
    @info "Optimal prism separation: $(round(L_opt*1000, digits=2)) mm"
    Eωcomp
end

## -- Plotting --
# Spectral evolution along the fibre
Plotting.prop_2D(output, :λ, λrange=(800e-9,1200e-9), trange=(-1000e-15, 1000e-15))

# Input and output spectra
Plotting.spec_1D(output, λrange=(800e-9,1200e-9))

# Compare compressed temporal profiles on the same axes
trange = (-500e-15, 500e-15)
t_g,  Et_g,  _ = Processing.getEt(output, flength; propagate=prop_gratings, trange, oversampling=4)
t_p,  Et_p,  _ = Processing.getEt(output, flength; propagate=prop_prisms,   trange, oversampling=4)
t_ftl, Et_ftl, _ = Processing.getEt(output, flength; FTL=true, trange, oversampling=4)

Pt_g   = abs2.(Et_g)
Pt_p   = abs2.(Et_p)
Pt_ftl = abs2.(Et_ftl)

# Normalise to the transform-limited peak power
Pmax = maximum(Pt_ftl)

fig, ax = PyPlot.subplots()
ax.plot((t_ftl .- t_ftl[argmax(Pt_ftl)]) .* 1e15, Pt_ftl ./ Pmax, "k--"; label="Transform limited")
ax.plot((t_g .- t_g[argmax(Pt_g)]) .* 1e15, Pt_g ./ Pmax;           label="Grating compressed")
ax.plot((t_p .- t_p[argmax(Pt_p)]) .* 1e15, Pt_p ./ Pmax;           label="Prism compressed")
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Normalised power")
ax.set_xlim(extrema(trange) .* 1e15)
ax.legend(; frameon=false)
fig.tight_layout()
