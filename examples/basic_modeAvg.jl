using Luna

const N0 =  2.5000013629442974e25

folder = "Ar_2bar_15um_4uJ_kerr_plasma(stat)\\"
dir = raw"C:\Users\mo_is\Dropbox (Heriot-Watt University Team)\RES_EPS_Lupo\Projects\Mohammed\phd\simulation data\new\\"*folder

a = 15e-6
gas = :Ar
pres = 2.0
PP = [1.0]
flength = 0.225

τfwhm = 30e-15
λ0 = 800e-9
energy = 4e-6

grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)
g = grid

rfg = PhysData.ref_index_fun([:Ar], pres, PP)
rfs = PhysData.ref_index_fun(:SiO2)
m = Capillary.MarcatilliMode(a, rfg, rfs, PP, loss=false)

aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

energyfun, energyfunω = Fields.energyfuncs(grid)

densityfun(z) = N0*pres

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

ionpots2 = [:O2dis]
weights2 = fill(0.1, (Int(g.zmax*1000+1)))
ionrate2 = Ionisation.inFiber_rate(ionpots2, weights2)
dissO = Nonlinear.DissCumtrapz(grid.to, grid.to, ionrate2, ionpots2, g.zmax; weights=weights2)

plasma = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)
responses = (
             Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             plasma,
            #  Nonlinear.RamanPolarField(grid.to, Raman.raman_response(gas)),
             dissO,
            )

linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

# statsfun = Stats.default(grid, Eω, m, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
ozoneStat = Stats.ozoneStat(grid, Eω, m, linop, transform; gas=gas, windows=((150e-9, 300e-9),)) 
output = Output.HDF5Output(dir*"file.h5", 0, grid.zmax, 201, ozoneStat)

Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(false)
Plotting.stats(dir, output)
Plotting.prop_2D(dir, output)
Plotting.time_1D(dir, output, [5e-2, 10e-2, 11e-2])
Plotting.spec_1D(dir, output, [5e-2, 10e-2, 11e-2])
##
Plotting.spectrogram(dir, output, 9.8e-2, :λ; trange=(-50e-15, 50e-15), λrange=(160e-9, 1200e-9),
                     N=512, fw=3e-15,
                     cmap=Plotting.cmap_white("viridis", n=48))