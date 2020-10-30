using Luna
import Luna.PhysData: wlfreq
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
import LinearAlgebra: mul!, ldiv!
Logging.disable_logging(Logging.BelowMinLevel)

import PyPlot:pygui, plt

a = 225e-6
gas = :Ar
pres = 0.4

τfwhm = 30e-15
λ0 = 1800e-9
energy = 500e-6

modes = (Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0),
         Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=π/2))
nmodes = length(modes)

grid = Grid.RealGrid(250e-2, λ0, (200e-9, 3000e-9), 1e-12)

energyfun, energyfunω = Fields.energyfuncs(grid)

gausspulse(t) = sqrt.(Maths.gauss.(t; fwhm=τfwhm)) .* cos.(wlfreq(λ0).*t)

Etlin = gausspulse(grid.t)
cenergy = energyfun(Etlin)
Etlin = sqrt(energy)/sqrt(cenergy) .* Etlin

Et = [Etlin zero(grid.t)]
#Etd = size(Et)
#Etp = Array{ComplexF64,2}(undef, Etd[2], Etd[1])
#permutedims!(Etp, Et, [2, 1])

Ew = FFTW.rfft(Et, 1);
Ewd = size(Ew)
Ewp = Array{ComplexF64,2}(undef, Ewd[2], Ewd[1])
permutedims!(Ewp, Ew, [2, 1])

γ = deg2rad(10.0)
CP = Polarisation.rotate(Polarisation.WP(π/2), π/4 - γ)
Ewp .= CP * Ewp
permutedims!(Ew, Ewp, [2, 1])
#Et = FFTW.irfft(Ew, length(grid.t), 1);

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes, :xy; full=false)

Eω .= Ew

statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Eoutd = size(Eout)
Ewp = Array{ComplexF64,2}(undef, Eoutd[2], Eoutd[1])
permutedims!(Ewp, Eout[:,:,end], [2, 1])

CP = Polarisation.rotate(Polarisation.WP(π/2), γ)
LP = Polarisation.rotate(Polarisation.LP(), π/2)
Ewp .= LP * CP * Ewp

Eoutp = Array{ComplexF64,2}(undef, Eoutd[1], Eoutd[2])
permutedims!(Eoutp, Ewp, [2, 1])

pygui(true)
plt.figure()
plt.plot(ω./2π.*1e-15, log10.(abs2.(Eout[:,:,end])))
plt.plot(ω./2π.*1e-15, log10.(abs2.(Eoutp)))
#plt.ylim(-6, 0)
plt.xlim(0,2.0)

##
plt.figure()
It = abs2.(Maths.hilbert(FFTW.irfft(Eout[:,:,end], length(grid.t), 1)))
plt.plot(t/1e-15, It[:,1]./maximum(It[:,1]))
plt.plot(t/1e-15, It[:,2]./maximum(It[:,2]))
Itp = abs2.(Maths.hilbert(FFTW.irfft(Eoutp, length(grid.t), 1)))
plt.plot(t/1e-15, Itp[:,1]./maximum(Itp[:,1]))
plt.plot(t/1e-15, Itp[:,2]./maximum(Itp[:,2]))
plt.xlim(-50.0,50.0)

println("$(sum(Itp)/sum(It))")
##
Etout = FFTW.irfft(Eout, length(grid.t), 1)
It = Maths.normbymax(abs2.(Maths.hilbert(Etout)))
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

S = Array{Float64,3}(undef, 4, Eoutd[1], Eoutd[3])
elip = Array{Float64,2}(undef, Eoutd[1], Eoutd[3])
for i = 1:Eoutd[3]
    permutedims!(Ewp, Eout[:,:,i], [2, 1])
    for j = 1:Eoutd[1]
        S[:,j,i] .= Polarisation.Stokes(Ewp[:,j])
        elip[j,i] = Polarisation.ellipticity(S[:,j,i])
    end
end

plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, S[3,:,:]./S[1,:,:])
plt.colorbar()

plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, S[2,:,:]./S[1,:,:])
plt.colorbar()

plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, elip)
plt.colorbar()

Eenvo = Maths.hilbert(Etout)
Etoutd = size(Eenvo)
Etp = Array{ComplexF64,2}(undef, Etoutd[2], Etoutd[1])

St = Array{Float64,3}(undef, 4, Etoutd[1], Etoutd[3])
elipt = Array{Float64,2}(undef, Etoutd[1], Etoutd[3])
Ss = Array{Float64,2}(undef, 4, Etoutd[3])
Elipt = Array{Float64,1}(undef, Etoutd[3])
CP = Polarisation.rotate(Polarisation.WP(π/2), γ)
for i = 1:Etoutd[3]
    permutedims!(Etp, Eenvo[:,:,i], [2, 1])
    for j = 1:Etoutd[1]
        St[:,j,i] .= Polarisation.Stokes(CP * Etp[:,j])
        elipt[j,i] = Polarisation.ellipticity(St[:,j,i])
    end
    k = argmax(St[1,:,i])
    Ss[:,i] .= St[:,k,i] ./ St[1,k,i]
    Elipt[i] = elipt[k,i]
end

plt.figure()
plt.plot(zout, Ss[1,:], label="S1")
plt.plot(zout, Ss[2,:], label="S2")
plt.plot(zout, Ss[3,:], label="S3")
plt.plot(zout, Ss[4,:], label="S4")
plt.plot(zout, Elipt, label="ε")
plt.legend()


