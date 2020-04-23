using Luna

a = 13e-6
gas = :Ar
pres = 5
flength = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

modes = (
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
)

grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)

energyfun, energyfunω = Fields.energyfuncs(grid)
normfun = NonlinearRHS.norm_modal(grid.ω)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            #  Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs,
                              modes, :y; full=false)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.energy_λ(grid, energyfunω, (150e-9, 300e-9), label="RDW"),
                               Stats.peakpower(grid),
                               Stats.peakintensity(grid, modes),
                               Stats.fwhm_t(grid),
                               Stats.fwhm_r(grid, modes),
                               Stats.electrondensity(grid, ionrate, densityfun, modes)
                               )
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

import FFTW
import PyPlot:pygui, plt

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Etout = FFTW.irfft(Eout, length(grid.t), 1)
It = abs2.(Maths.hilbert(Etout))

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

pygui(true)
##
for i = 1:length(modes)
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog[:,i,:]))
    plt.clim(-6, 0)
    plt.xlim(0,2.0)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(t.*1e15, zout, transpose(It[:,i,:]))
    plt.xlim(-30.0,100.0)
    plt.colorbar()
end

##
plt.figure()
for i = 1:length(modes)
    plt.semilogy(100*output["stats"]["z"], 1e6*output["stats"]["energy"][i, :],
                 label="Mode $i") 
end
plt.xlabel("Distance (cm)")
plt.ylabel("Energy (μJ)")
plt.legend()

##
plt.figure()
for i = 1:length(modes)
    plt.plot(100*output["stats"]["z"], 1e6*output["stats"]["energy_RDW"][i, :],
    label="Mode $i") 
end
plt.xlabel("Distance (cm)")
plt.ylabel("RDW Energy (μJ)")
plt.legend()

##
plt.figure()
for i = 1:length(modes)
    plt.plot(100*output["stats"]["z"], 1e-9*output["stats"]["peakpower"][i, :],
    label="Mode $i") 
end
plt.xlabel("Distance (cm)")
plt.ylabel("Peak power (GW)")
plt.legend()

##
plt.figure()
plt.plot(output["stats"]["z"].*1e2, output["stats"]["peakintensity"].*1e-4.*1e-12)
plt.xlabel("Distance (cm)")
plt.ylabel("Peak intensity (TW/cm\$^2\$)")

##
plt.figure()
for i = 1:length(modes)
    plt.plot(100*output["stats"]["z"], 1e15*output["stats"]["fwhm_t_min"][i, :],
    label="Mode $i (min)")
    plt.plot(100*output["stats"]["z"], 1e15*output["stats"]["fwhm_t_min"][i, :],
    label="Mode $i (max)")
end
plt.xlabel("Distance (cm)")
plt.ylabel("FWHM (fs)")
plt.legend()

##
if haskey(output["stats"], "electrondensity")
    plt.figure()
    plt.plot(output["stats"]["z"].*1e2, output["stats"]["electrondensity"]*1e-6)
    plt.xlabel("Distance (cm)")
    plt.ylabel("Electron Density (cm\$^{-3}\$)")
end

##
plt.figure()
plt.plot(output["stats"]["z"].*1e2, output["stats"]["fwhm_r"]*1e6)
plt.axhline(0.93675*a*1e6, linestyle="--", label="FWHM of EH11")
plt.xlabel("Distance (cm)")
plt.ylabel("Radial FWHM (μm)")
plt.legend()

