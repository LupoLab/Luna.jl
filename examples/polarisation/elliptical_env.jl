using Luna
import StaticArrays: SMatrix, SVector
import LinearAlgebra: mul!, ldiv!

a = 50e-6
gas = :Ar
pres = 5

τfwhm = 30e-15
λ0 = 400e-9
energy = 30e-6

modes = (Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0),
         Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=π/2))
nmodes = length(modes)

grid = Grid.EnvGrid(10e-2, λ0, (160e-9, 3000e-9), 1e-12)

energyfun = Fields.energyfuncs(grid)[1]
normfun = NonlinearRHS.norm_modal(grid.ω)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    Et = @. sqrt(It)
end

field = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
Etlin = gausspulse(grid.t)
cenergy = energyfun(Etlin)
Etlin = sqrt(energy)/sqrt(cenergy) .* Etlin

Et = [Etlin zero(grid.t)] .+ 0im
Ew = FFTW.fft(Et, 1);
Ewd = size(Ew)
Ewp = Array{ComplexF64,2}(undef, Ewd[2], Ewd[1])
permutedims!(Ewp, Ew, [2, 1])

CP = Polarisation.rotate(Polarisation.WP(π/2), π/6 + π/8)
Ewp .= CP * Ewp
permutedims!(Ew, Ewp, [2, 1])
Et = FFTW.ifft(Ew, 1);

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

# dummy, we don't use these
inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs,
                              modes, :xy; full=false)

Eω .= Ew

statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

import FFTW
import PyPlot:pygui, plt

ω = FFTW.fftshift(grid.ω)
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]
Eoutd = size(Eout)
Ewp = Array{ComplexF64,3}(undef, Eoutd[2], Eoutd[1], Eoutd[3])
permutedims!(Ewp, Eout, [2, 1, 3])
S = Array{Float64,3}(undef, 4, Eoutd[1], Eoutd[3])
elip = Array{Float64,2}(undef, Eoutd[1], Eoutd[3])

for i = 1:Eoutd[3]
    for j = 1:Eoutd[1]
        S[:,j,i] .= Polarisation.Stokes(Ewp[:,j,i])
        elip[j,i] = Polarisation.ellipticity(S[:,j,i])
    end
end

pygui(true)
plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, FFTW.fftshift(S[3,:,:]./S[1,:,:],1))
plt.colorbar()

plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, FFTW.fftshift(S[2,:,:]./S[1,:,:],1))
plt.colorbar()

plt.figure()
plt.pcolormesh(zout, ω./2π.*1e-15, FFTW.fftshift(elip,1))
plt.colorbar()

Etout = FFTW.ifft(Eout, 1)
It = abs2.(Etout)

Ilog = FFTW.fftshift(log10.(Maths.normbymax(abs2.(Eout))), 1)



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
