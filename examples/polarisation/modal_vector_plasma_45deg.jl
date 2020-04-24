using Luna

a = 9e-6
gas = :Ar
pres = 5
flength = 25e-2

τfwhm = 30e-15
λ0 = 1500e-9
energy = 1.7e-6

grid = Grid.RealGrid(flength, λ0, (200e-9, 3000e-9), 2e-12)

modes = (
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=π/2, loss=false),
)
nmodes = length(modes)

energyfun, energyfunω = Fields.energyfuncs(grid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
plasma = Nonlinear.PlasmaCumtrapz(grid.to, Array{Float64}(undef, length(grid.to), 2),
                                  ionrate, ionpot)
                                  
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             plasma)

normfun = NonlinearRHS.norm_modal(grid.ω)

inputs = ((mode=1, fields=(Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy/2),)),
          (mode=2, fields=(Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy/2),)))

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs,
                               modes, :xy; full=false)

statsfun = Stats.collect_stats(grid, Eω,
                              Stats.ω0(grid),
                              Stats.energy(grid, energyfunω),
                              Stats.peakpower(grid),
                              Stats.peakintensity(grid, modes),
                              Stats.fwhm_t(grid),
                              Stats.fwhm_r(grid, modes),
                              Stats.electrondensity(grid, ionrate, densityfun, modes)
                              )
output = Output.HDF5Output("modalvector.h5", 0, grid.zmax, 201, statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

import FFTW
import PyPlot:pygui, plt

ω = grid.ω
t = grid.t

zout = output["z"]
Eout = output["Eω"]

Etout = FFTW.irfft(Eout, length(grid.t), 1)
It = abs2.(Maths.hilbert(Etout))

pygui(true)

λ = PhysData.wlfreq.(ω)[2:end]
Ilog = log10.(Maths.normbymax(abs2.(Eout[2:end,:,:]) ./ λ.^2) .+ 1e-30)

for i = 1:nmodes
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(λ, zout, transpose(Ilog[:,i,:]))
    plt.clim(-3, 0)
    plt.xlim(550e-9,1800e-9)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(t.*1e15, zout, transpose(It[:,i,:]))
    plt.xlim(-500.0,100.0)
    plt.colorbar()
end
