import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, Modes, RK45
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
import LinearAlgebra: mul!, ldiv!
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

modes = ((1,1,:HE), (1,2,:HE))#, (1,3,:HE), (1,4,:HE))
nmodes = length(modes)

grid = Grid.RealGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)

energyfun = Modes.energy_modal()

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

function get_linop(grid, a, n, m, kind, vel)
    #β1const = Capillary.dispersion(1, a; λ=λ0, gas=gas, pressure=pres, n=n, m=m)
    βconst = zero(grid.ω)
    βconst[2:end] = Capillary.β(a, grid.ω[2:end], gas=gas, pressure=pres, n=n, m=m)
    βconst[1] = 1
    βfun(ω, m, n, z) = βconst
    frame_vel(z) = vel
    αfun(ω, m, n, z) = 0.0
    Luna.make_linop(grid, βfun, αfun, frame_vel)
end

β1const = Capillary.dispersion(1, a; λ=λ0, gas=gas, pressure=pres, n=1, m=1)
linops = zeros(ComplexF64, length(grid.ω), nmodes)
for i = 1:nmodes
    linops[:,i] = get_linop(grid, a, modes[i][1], modes[i][2], modes[i][3], 1/β1const)
end


densityfun(z) = PhysData.std_dens * pres

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.χ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

in1 = (func=gausspulse, energy=1e-6, m=1, n=1)
inputs = (in1, )

xt = Array{Float64}(undef, length(grid.t))
FTt = FFTW.plan_rfft(xt, 1, flags=FFTW.MEASURE)

Eω = zeros(ComplexF64, length(grid.ω), nmodes)
Eω[:,1] .= Luna.make_init(grid, inputs, energyfun, FTt)

Exy = Capillary.fields(a, modes; components=:Exy)

x = Array{Float64}(undef, length(grid.t), nmodes)
FT = FFTW.plan_rfft(x, 1, flags=FFTW.MEASURE)

xo1 = Array{Float64}(undef, length(grid.to), 2)
FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.MEASURE)

transform = Modes.TransModalRadialMat(grid, a, Exy, FTo1, responses, densityfun; rtol=1e-3, atol=0.0, mfcn=300, full=true)

Et = FT \ Eω

z = 0
dz = 1e-3
zmax = grid.zmax
saveN = 201

window! = let window=grid.ωwin, twindow=grid.twin, FT=FT, Et=Et
    function window!(Eω)
        Eω .*= window
        ldiv!(Et, FT, Eω)
        Et .*= twindow
        mul!(Eω, FT, Et)
    end
end

zout, Eout, steps = RK45.solve_precon(
        transform, linops, Eω, z, dz, zmax, saveN, stepfun=window!)

ω = grid.ω
t = grid.t

Etout = FFTW.irfft(Eout, length(grid.t), 1)
It = abs2.(Maths.hilbert(Etout))

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

pygui(true)

for i = 1:nmodes
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
