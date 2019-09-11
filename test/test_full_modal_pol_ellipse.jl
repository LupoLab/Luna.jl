import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, RK45, Stats, Output, LinearOps
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
import LinearAlgebra: mul!, ldiv!
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

a = 50e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 400e-9
energy = 30e-6

modes = (Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0),
         Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=π/2))
nmodes = length(modes)

grid = Grid.RealGrid(10e-2, 400e-9, (160e-9, 3000e-9), 1e-12)

energyfun = NonlinearRHS.energy_modal()
normfun = NonlinearRHS.norm_modal(grid.ω)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

Etlin = gausspulse(grid.t)
cenergy = energyfun(grid.t, Etlin)
Etlin = sqrt(energy)/sqrt(cenergy) .* Etlin

Et = [Etlin zero(grid.t)]
Ew = FFTW.rfft(Et, 1);
Ewd = size(Ew)
Ewp = Array{ComplexF64,2}(undef, Ewd[2], Ewd[1])
permutedims!(Ewp, Ew, [2, 1])

"Make arbitrary wave plate Jones matrix"
function WP(η, θ; ϕ=0.0)
    m = exp(-im*η/2.0) .* [cos(θ)^2 + exp(im*η)*sin(θ)^2                (1.0 -  exp(im*η))*exp(-im*ϕ)*cos(θ)*sin(θ)  ;
                           (1.0 -  exp(im*η))*exp(im*ϕ)*cos(θ)*sin(θ)   sin(θ)^2 + exp(im*η)*cos(θ)^2                ]
    SMatrix{2,2}(m)
end

CP = WP(π/2, π/8+0.046, ϕ=π/2)*WP(π, π/8.0, ϕ=0.0)
Ewp .= CP * Ewp
permutedims!(Ew, Ewp, [2, 1])
Et = FFTW.irfft(Ew, length(grid.t), 1);

densityfun(z) = PhysData.std_dens * pres

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

in1 = (func=gausspulse, energy=1e-6)
inputs = ((1,(in1,)),)

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                              modes, :Exy; full=false)

Eω .= Ew

statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Eoutd = size(Eout)
Ewp = Array{ComplexF64,3}(undef, Eoutd[2], Eoutd[1], Eoutd[3])
permutedims!(Ewp, Eout, [2, 1, 3])

Ewp .= WP(π/2, π/2 + π/8+0.046, ϕ=π/2) * Ewp

S = Array{Float64,3}(undef, 4, Eoutd[1], Eoutd[3])
elip = Array{Float64,2}(undef, Eoutd[1], Eoutd[3])

function Stokes(E)
    Ex = E[1]
    Ey = E[2]
    I = abs(Ex)^2 + abs(Ey)^2
    Q = abs(Ex)^2 - abs(Ey)^2
    U = 2*real(Ex*conj(Ey))
    V = -2*imag(Ex*conj(Ey))
    SVector(I, Q, U, V)
end

function ellipse(S)
    I = S[1]
    Q = S[2]
    U = S[3]
    V = S[4]
    aL = sqrt(Q^2 + U^2)
    θ = angle(aL)/2
    A = sqrt((I + aL)/2)
    B = sqrt((abs(I - aL))/2)
    h = sign(V)
    A, B, θ, h
end

function ellipticity(S)
    A, B, θ, h = ellipse(S)
    r = A/B
    r > 1 ? 1/r : r
end

for i = 1:Eoutd[3]
    for j = 1:Eoutd[1]
        S[:,j,i] .= Stokes(Ewp[:,j,i])
        elip[j,i] = ellipticity(S[:,j,i])
    end
end

pygui(true)
plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, S[4,:,:]./S[1,:,:])
plt.colorbar()

plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, elip)
plt.colorbar()

Etout = FFTW.irfft(Eout, length(grid.t), 1)
It = abs2.(Maths.hilbert(Etout))

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

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
