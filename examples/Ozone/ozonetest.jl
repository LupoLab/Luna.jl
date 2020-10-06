using Luna
import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes, Raman
import Logging
import FFTW, HDF5
import NumericalIntegration: integrate, SimpsonEven
import DifferentialEquations: ODEProblem, solve, Tsit5
import LinearAlgebra: Tridiagonal
using Dierckx
using Formatting

const N0 =  2.5000013629442974e25

dataDir = cd(pwd, string(@__DIR__)*"/../../src/data/")
# table = HDF5.h5open(joinpath(pwd(), "15um_12bar_He-O2_noLoss"), "r+")

a = 15e-6
gas = :HeO2
pres = 12.0
gases = [:He, :O2]
PP = [0.79, 0.21]
flength = 0.225

τfwhm = 30e-15
λ0 = 800e-9
energy = 2.5e-6

const factor = 1e-3*pi*a^2 # used to transfere from concentration to actual number of molecules inside the fiber. #Mo = Mc*factor
const M0 = N0*pres*factor # number of air molecules inside the fiber (1e-3 for 1 mm, pi*a^2 is fiber area)
const Mc = N0*pres # or N0*pres # gas concentration inside the fiber (particles/m^3)
externalDens = nothing

iterations = 5 # number of chapman calculation using the same pulse profile
count = 5 # number of pulse propagation calculation
tlim=(0.0,1e-3)

folder = format("PCF_HeO2_{}um_{}bar_{}fs_{}uJ_{}it(correct_diffusion)\\", a*1e6, pres, Int(τfwhm*1e15), energy*1e6, iterations)
dir = raw"C:\Users\mo_is\Dropbox (Heriot-Watt University Team)\RES_EPS_Lupo\Projects\Mohammed\phd\simulation data\new\\"*folder

# Diffusion coefficients in He
# values are from 
# 1-DIFFUSION MASS TRANSFER IN FLUID SYSTEMS THIRD EDITION
# 2-https://pubs.acs.org/doi/abs/10.1021/jp066558w
#= 
For He-O2 mixture
=#
DO = 756*1/760/pres*1e-4 # 756 original from 2, 1/760 to convert from torr to bar, /pres to scale with pressure, 1e-4 to convert to SI
DO2 =  0.718*1/1.01325/pres*1e-4 # 0.718 original from 1, 1/1.01325 to convert from atm to bar, /pres to scale with pressure, 1e-4 to convert to SI
DO3 = 410*1/760/pres*1e-4 # 410 original from 2, 1/760 to convert from torr to bar, /pres to scale with pressure, 1e-4 to convert to SI

grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)
g = grid
gw = Maths.planck_taper(grid.ω, 0, 2pi*3e8/3000e-9, 2pi*3e8/160e-9, 1.1*2pi*3e8/140e-9)

loss0 = LinearOps.load_ozone_loss(g, gw)

function calculate_adkdis(zout, t, ω, aeff, factor, adkO2, adkO3, It, Iw, loss, rO, rO2, rO3)
    dz = zout[2] - zout[1]
    dω = ω[2] - ω[1]
    phEn = PhysData.ħ*ω # photons energy
    dIw = zeros(Float64, size(Iw)) # the absorbed intensity for each 1mm
    for ii in 1:size(Iw)[2]
        dIw[:,ii] .= Iw[:,ii] .* (exp.(-loss*rO3[ii]/factor*dz).-1)
    end
    dPw = dIw .* aeff # the absorbed power for each 1mm
    dNw = dPw.*dω./phEn # number of photon absorbed in the UV as function of frequancy
    dNw[.~isfinite.(dNw)] .= 0.0
    dNw[dNw.<0.0] .= 0.0
    numPhoton = zeros(Float64, length(zout)) # total number of UV photon absorbed per 1 mm
    for ii in 1:size(dNw)[2]
        numPhoton[ii] = sum(dNw[:,ii])
    end
    O = @. adkO2*rO2 
    O3 = @. adkO3*rO3 + numPhoton
    disO = rO .+ 2.0.*O .+ O3  # O2 dissociation  # O3 dissociation
    disO2 = rO2 .- O .+ O3
    disO3 = rO3 .- O3
    return disO, disO2, disO3
end

function chapmanDiffu!(du,u,p,t)
    #=
    u[1] = O, u[2] = O2, u[3] = O3, du[4] = dM = 0
    =#
    k1,k2,k3,k4,M,DO,DO2,DO3,Ax = p
    O = @view u[:,1]
    O2 = @view u[:,2]
    O3 = @view u[:,3]
    du1 = @view du[:,1]
    du2 = @view du[:,2]
    du3 = @view du[:,3]
    D1 = DO*Ax*O
    D2 = DO2*Ax*O2
    D3 = DO3*Ax*O3

    D1[1] = 0
    D1[end] = 0
    D2[1] = 0
    D2[end] = 0
    D3[1] = 0
    D3[end] = 0
    
    du1 .= @. D1 + 2*k1*O2-k2*O*O2*M+k3*O3-k4*O*O3
    du2 .= @. D2 - k1*O2 - k2*O2*M*O + k3*O3 + 2*k4*O*O3 
    du3 .= @. D3 + k2*O*O2*M - k3*O3- k4*O*O3 
end

zlength = length(0:1e-3:grid.zmax)

function work(dens, rfg, rfs, iterations, count; tlim=tlim)
    L = zeros(ComplexF64, (length(grid.ω), zlength))
    for jj in 1:zlength
        ppO2 = round.(dens[:O2][jj]/Mc, digits=4)
        ppO3 = round.(dens[:O3][jj]/Mc, digits=4)
        gases = [:He, :O2, :O3]
        PP = [0.79, ppO2, ppO3]

        global m
        m = Capillary.MarcatilliMode(a, rfg, rfs, PP, loss=false)
        
        linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
        linop .-= loss0.*dens[:O3][jj]/2
        L[:,jj] = linop
    end
    aeff = let m=m
        z -> Modes.Aeff(m, z=z)
    end

    Linop0 = LinearOps.make_linop_from_data(grid, transpose(L))

    energyfun, energyfunω = Fields.energyfuncs(grid)

    densityfun(z) = N0*pres

    ionpots1 = [:O2m]
    weights1 = dens[:O2]./Mc
    ionrate1 = Ionisation.inFiber_rate(ionpots1, weights1)
    plasmaO = Nonlinear.DissCumtrapz(grid.to, grid.to, ionrate1, ionpots1, g.zmax; weights=weights1, includephase=true)

    ionpots2 = [:O2dis]
    weights2 = dens[:O2]./Mc
    ionrate2 = Ionisation.inFiber_rate(ionpots2, weights2)
    dissO2 = Nonlinear.DissCumtrapz(grid.to, grid.to, ionrate2, ionpots2, g.zmax; weights=weights2)

    ionpots3 = [:O3dis]
    weights3 = dens[:O3]./Mc
    ionrate3 = Ionisation.inFiber_rate(ionpots3, weights3)
    dissO3 = Nonlinear.DissCumtrapz(grid.to, grid.to, ionrate3, ionpots3, g.zmax; weights=weights3)

    responses = (
                Nonlinear.Kerr_field(PhysData.γ3_gas(gas, source=:Mix)),
                #  plasma,
                #  the resp below does not work since it does not have mo-adk implementation
                #  Nonlinear.scaled_response(Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot), 0.21, size(grid.to)),
                #  Nonlinear.RamanPolarField(grid.to, Raman.raman_response(gas)),
                plasmaO,
                dissO2,
                dissO3,
                )

    linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

    inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
    Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

    ozoneStat = Stats.ozoneStat(grid, Eω, m, linop, transform; gas=gas, windows=((150e-9, 300e-9),)) 
    global output
    output = Output.HDF5Output(dir*"file$count.h5", 0, grid.zmax, Int(g.zmax*1000+1), ozoneStat)

    Luna.run(Eω, grid, Linop0, transform, FT, output)

    adkO20 = output["stats"]["O2dis"]
    adkO30 = output["stats"]["O3dis"]
    zs = output["stats"]["z"]

    csplO2 = Maths.CSpline(zs, adkO20)
    csplO3 = Maths.CSpline(zs, adkO30)

    ω = grid.ω;
    t = grid.t;

    zout = collect(range(0, stop=g.zmax, length=Int(g.zmax*1000+1)))
    Eout = output["Eω"];

    adkO2 = csplO2.(zout)
    adkO3 = csplO3.(zout)

    dt = g.t[2]-g.t[1]

    Etout = FFTW.irfft(Eout, length(grid.t), 1)./sqrt(2/(PhysData.ε_0*PhysData.c*aeff(0.0)));

    to, Eto = Maths.oversample(t, Etout, factor=1);

    It = abs2.(Maths.hilbert(Eto))./(pi*a^2).*1e-4;

    Pw = zeros(Float64, size(Eout)) 
    for i in 1:length(zout)
        Eth = Maths.hilbert(Etout[:,i])
        Ew = FFTW.fft(Eth)*dt
        Pw[:,i] = abs2.(Ew[1:Int(size(Ew)[1]/2+1)])/(2pi)
    end

    Iw = Pw./aeff(0.0)

    dx = zout[2]-zout[1]
    Ax = Array(Tridiagonal([1.0 for i in 1:length(zout)-1],[-2.0 for i in 1:length(zout)],[1.0 for i in 1:length(zout)-1]))
    Ax .= Ax./dx^2

    for jj in 1:iterations
        rO, rO2, rO3 = dens[:O].*factor, dens[:O2].*factor, dens[:O3].*factor # rO is the number of O atoms in 1mm inside the fiber
        disO, disO2, disO3 = calculate_adkdis(zout, t, ω, aeff(0.0), factor, adkO2, adkO3, It, Iw, loss0, rO, rO2, rO3)
       
        dens[:O] .= disO/factor
        dens[:O2] .= disO2/factor
        dens[:O3] .= disO3/factor   

        if any(isnan(x) for x in dens[:O2])
            error("NaN encountered from the spline")
        end

        u0 = zeros(length(zout),3)
        u0[:,1] .= dens[:O]
        u0[:,2] .= dens[:O2]
        u0[:,3] .= dens[:O3]

        # Chapman and Diffusion constants
        p = [0.0,  6.35E-46, 0.0, 7E-21, Mc, DO, DO2, DO3, Ax]
        tspan = tlim
        prob1 = ODEProblem(chapmanDiffu!, u0, tspan, p)

        sol = solve(prob1, Tsit5(), saveat=2)
        dens[:O] = round.(sol.u[end][:,1])
        dens[:O2] = round.(sol.u[end][:,2])
        dens[:O3] = round.(sol.u[end][:,3])

        dens[:O][1] = 0
        dens[:O][end] = 0
        dens[:O2][1] = 0.21*Mc
        dens[:O2][end] = 0.21*Mc
        dens[:O3][1] = 0
        dens[:O3][end] = 0
    end
    return dens
end

function run(stop, iterations)
    count = 1
    if count == 1
        if isnothing(externalDens)
            dens = Dict{Symbol, Array{Float64,1}}(
                :O => zeros(Float64, zlength),
                :O2 => fill(0.21*Mc, zlength),
                :O3 => fill(0.00*Mc, zlength)
            )
        else
            densfile = joinpath(dataDir, "$externalDens")
            fid2 = HDF5.h5open(densfile, "r")
            densO3 = read(fid2["O3"])
            densO2 = read(fid2["O2"])
            close(fid2)
            dens = Dict{Symbol, Array{Float64,1}}(
                :O => zeros(Float64, zlength),
                :O2 => densO2,
                :O3 => densO3
            )
        end
    end
            
    rfg = PhysData.ref_index_fun(gases, pres, PP)
    rfs = PhysData.ref_index_fun(:SiO2)
    m = Capillary.MarcatilliMode(a, rfg, rfs, PP, loss=false)
    while count != stop    
        # work(dens, m, iterations, count; tlim=tlim)
        ndens = work(dens, rfg, rfs, iterations, count; tlim=tlim)
        dens[:O] = ndens[:O]
        dens[:O2] = ndens[:O2]
        dens[:O3] = ndens[:O3]
        fid = HDF5.h5open(dir*"dens$count", "w")
        fid["O"] = dens[:O]
        fid["O2"] = dens[:O2]
        fid["O3"] = dens[:O3]
        close(fid)
        count +=1
    end
end

run(count, iterations)
close()

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