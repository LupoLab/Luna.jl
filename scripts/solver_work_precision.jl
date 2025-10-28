# Calculate work-precision plots for various NLSE solvers

using DifferentialEquations, SciMLOperators
import FFTW
import LinearAlgebra: inv, mul!, ldiv!, norm, Diagonal
using PyPlot
import Luna
import Printf: @sprintf

# NLSE grid and temporary storage
mutable struct NLSE{TFT}
    n::Int
    dt::Float64
    dΩ::Float64
    T::Vector{Float64}
    Ω::Vector{Float64}
    ut::Vector{ComplexF64}
    utmp::Vector{ComplexF64}
    utmp2::Vector{ComplexF64}
    dutmp::Vector{ComplexF64}
    L::Vector{ComplexF64}
    FT::TFT
    cz::Float64
    nfunc::Int
end

function NLSE(dt, trange)
    n = nextpow(2, ceil(Int, trange/dt))
    dt = trange/n
    dΩ = 2π/trange
    T = collect((-n//2:n//2-1)*dt)
    Ω = FFTW.fftshift((-n//2:n//2-1)*dΩ)
    ut = @. complex(5*sech(T))
    utmp = similar(ut)
    utmp2 = similar(ut)
    dutmp = similar(ut)
    L = @. 1im*Ω^2/2
    FT = FFTW.plan_fft(ut)
    inv(FT)
    cz = 0.0
    NLSE(n, dt, dΩ, T, Ω, ut, utmp, utmp2, dutmp, L, FT, cz, 0)
end

function reset!(nlse::NLSE)
    nlse.cz = 0.0
    nlse.nfunc = 0
end

# explicit linear operator
function f1!(du,u,p,z)
    @. du = p.L*u
end

# nonlinear operator (Kerr effect)
function f2!(du,u,p,z)
    p.nfunc += 1
    ldiv!(p.ut, p.FT, u)
    @. p.utmp = -1im*abs2(p.ut)*p.ut
    mul!(du, p.FT, p.utmp)
end

# full interaction picture, constant L, analytically integrated
function fpre!(du,u,p,z)
    @. p.utmp = u*exp(p.L*z)
    f2!(du,p.utmp,p,z)
    @. du *= exp(-p.L*z)
end

# piecewise interaction picture, constant L, analytically integrated
function fpre2!(du,u,p,z)
    @. p.utmp = u*exp(p.L*(z  - p.cz))
    f2!(du,p.utmp,p,z)
    @. du *= exp(-p.L*(z  - p.cz))
end

# full interaction picture, L numerically integrated
function fdbl!(du,u,p,z)
    uu = @view u[1:length(u)÷2]
    ll = @view u[length(u)÷2+1:end]
    duu = @view du[1:length(u)÷2]
    dll = @view du[length(u)÷2+1:end]
    @. p.utmp = uu*exp(ll)
    f2!(p.utmp2,p.utmp,p,z)
    @. duu = p.utmp2*exp(-ll)
    @. dll = p.L
end

# explicitly call both linear and nonlinear terms, this is stiff
function fall!(du,u,p,z)
    f1!(p.dutmp,u,p,z)
    f2!(du,u,p,z)
    du .+= p.dutmp
end

# 5th order soliton initial condition
function getinit(nlse::NLSE)
    ut0 = @. complex.(5*sech(nlse.T))
    FFTW.fft(ut0)
end

# reset u and cz at each step for piecewise interaction picture solver
function resetaffect!(integrator)
    integrator.u .= integrator.u .* exp.(integrator.p.L .* (integrator.t  - integrator.p.cz))
    integrator.p.cz = integrator.t
end

function noaffect!(integrator)
    # do nothing
end

function geterror(nlse, u)
    ana = @. 5*sech(nlse.T)*exp(-1im*π/4)
    norm(FFTW.ifft(u) .- ana)/norm(ana)
end

function run(prob, solver, adaptive, dt, reltol, abstol; cb=nothing)
    zs = range(0.0, π/2, length=201)
    println("building")
    @time integrator = init(prob, solver; dt, adaptive, reltol, abstol, saveat=zs, callback=cb)
    println("starting")
    @time u = solve!(integrator)
    zs, u, integrator
end

# run full interaction picture, constant L, analytically integrated
function run_fullip(nlse::NLSE; solver=Tsit5(), adaptive=true, dt=0.0002, reltol=1e-2, abstol=1e-6, fullret=false)
    reset!(nlse)
    prob = ODEProblem(fpre!, getinit(nlse), (0.0, π/2), nlse)
    zs, u, integrator = run(prob, solver, adaptive, dt, reltol, abstol)
    res = Array{Complex{Float64}}(undef, nlse.n, length(zs))
    for (i,z) in enumerate(zs)
        @. res[:,i] =  u[:,i] * exp(nlse.L * z)
    end
    err = geterror(nlse, res[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    if fullret
        return zs, res, nlse.nfunc, err, u, integrator
    end
    zs, res, nlse.nfunc, err
end

# run full interaction picture, L numerically integrated
function run_numfullip(nlse::NLSE; solver=Tsit5(), adaptive=true, dt=0.0002, reltol=1e-2, abstol=1e-6)
    reset!(nlse)
    u0 = vcat(getinit(nlse), zero(nlse.L))
    prob = ODEProblem(fdbl!, u0, (0.0, π/2), nlse)
    cb = DiscreteCallback((u,t,integrator) -> true, noaffect!, save_positions=(true,true))
    zs, u, integrator = run(prob, solver, adaptive, dt, reltol, abstol; cb=cb)
    zs = Array(u.t)
    res = Array{Complex{Float64}}(undef, nlse.n, length(zs))
    for i in 1:length(zs)
        @. res[:,i] =  u[1:nlse.n,i] * exp(u[nlse.n + 1:end,i])
    end
    err = geterror(nlse, res[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    zs, res, nlse.nfunc, err
end

# piecewise interaction picture, constant L, analytically integrated
function run_pieceip(nlse::NLSE; solver=Tsit5(), adaptive=true, dt=0.0002, reltol=1e-2, abstol=1e-6)
    reset!(nlse)
    prob = ODEProblem(fpre2!, getinit(nlse), (0.0, π/2), nlse)
    cb = DiscreteCallback((u,t,integrator) -> true, resetaffect!, save_positions=(false,true))
    _, u, integrator = run(prob, solver, adaptive, dt, reltol, abstol; cb)
    res = Array(u)
    zs = u.t
    err = geterror(nlse, res[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    zs, res, nlse.nfunc, err
end

function run_stiff(nlse::NLSE; solver=Tsit5(), adaptive=true, dt=0.0002, reltol=1e-2, abstol=1e-6)
    reset!(nlse)
    prob = ODEProblem(fall!, getinit(nlse), (0.0, π/2), nlse)
    zs, u, integrator = run(prob, solver, adaptive, dt, reltol, abstol)
    res = Array(u)
    err = geterror(nlse, res[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    zs, res, nlse.nfunc, err
end

# Exponential RK integrator
function run_splitlin(nlse::NLSE; solver=ETDRK4(), adaptive=false, dt=0.0002, reltol=1e-2, abstol=1e-6)
    reset!(nlse)
    op = DiagonalOperator(nlse.L)
    f = SplitFunction(op, f2!)
    prob = SplitODEProblem(f, getinit(nlse), (0.0, π/2), nlse)
    zs, u, integrator = run(prob, solver, adaptive, dt, reltol, abstol)
    res = Array(u)
    err = geterror(nlse, res[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    zs, res, nlse.nfunc, err
end
    
# Luna original RK45 solver
function run_Luna_weak(nlse::NLSE; solver=nothing, adaptive=true, dt=0.0002, reltol=1e-2, abstol=1e-6)
    reset!(nlse)
    z, u, steps = Luna.RK45.solve_precon((du, u, z) -> f2!(du, u, nlse, z), nlse.L, getinit(nlse), 0.0, dt, π/2;
                    rtol=reltol, atol=abstol, output=true, locextrap=true, norm=Luna.RK45.weaknorm)
    err = geterror(nlse, u[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    z, u, nlse.nfunc, err
end

# Luna original RK45 solver with better norm
function run_Luna_norm(nlse::NLSE; solver=nothing, adaptive=true, dt=0.0002, reltol=1e-2, abstol=1e-6)
    reset!(nlse)
    z, u, steps = Luna.RK45.solve_precon((du, u, z) -> f2!(du, u, nlse, z), nlse.L, getinit(nlse), 0.0, dt, π/2;
                    rtol=reltol, atol=abstol, output=true, locextrap=true, norm=Luna.RK45.normnorm)
    err = geterror(nlse, u[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    z, u, nlse.nfunc, err
end

# Luna new solver
function run_newLuna(nlse::NLSE; solver=:Tsit5, adaptive=true, dt=0.0002, reltol=1e-2, abstol=1e-6)
    reset!(nlse)
    zs = range(0.0, π/2, length=201)
    iz = 2
    res = Array{Complex{Float64}}(undef, nlse.n, length(zs))
    res[:,1] = getinit(nlse)
    function stepfun(u, z, dz, interpolant)
        while iz <= length(zs) && z >= zs[iz]
            res[:,iz] = interpolant(zs[iz])
            iz += 1
        end
    end
    sol = Luna.Propagator.propagate((du, u, z) -> f2!(du, u, nlse, z), nlse.L, res[:,1], 0, π/2, stepfun;
                    rtol=reltol, atol=abstol, init_dz=dt, max_dz=Inf, min_dz=0,
                    status_period=10, solver)
    err = geterror(nlse, res[:,end])
    println("nfunc: $(nlse.nfunc)")
    println("error: $err")
    zs, res, nlse.nfunc, err
end

function workprecision(nlse::NLSE, solvers)
    errs = []
    nfs = []
    for (i,solverset) in enumerate(solvers)
        solver, dts, reltols, abstols, label = solverset
        errsi = zeros(length(dts))
        nfsi = zeros(length(dts))
        for j in 1:length(dts)
            z, u, nfuncs, err = solver(nlse; reltol=reltols[j], abstol=abstols[j], dt=dts[j])
            errsi[j] = err
            nfsi[j] = nfuncs
        end
        if isnothing(label)
            label = string(solver)
        end
        loglog(errsi, nfsi, label=label)
        push!(errs, errsi)
        push!(nfs, nfsi)
    end
    legend()
    PyPlot.grid()
    xlabel("Error")
    ylabel("Function Evaluations")
    ylim(2e3,4e4)
    xlim(1e-6,1e-1)
    errs, nfs
end

function plot_nlse(nlse::NLSE, z, u; axs=nothing)
    IT = 10log10.(abs2.(FFTW.ifft(u,1)))
    IW = 10log10.(abs2.(FFTW.fftshift(u,1)))
    IT .-= maximum(IT)
    IW .-= maximum(IW)
    if isnothing(axs)
        fig = PyPlot.plt.figure(constrained_layout=true, figsize=(10, 6))
        axd = fig.subplot_mosaic(
            """
            ab
            """)
        axs = (axd["a"], axd["b"])
    end
    axs[1].pcolormesh(z, nlse.T, IT, clim=(-200,0), rasterized=true)
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Time")
    img = axs[2].pcolormesh(z, FFTW.fftshift(nlse.Ω), IW, clim=(-200,0), rasterized=true)
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Frequency")
    colorbar(img, ax=axs, fraction=0.05, pad=0.1, label="dB")
end

function plot_nlse_cmp(nlse::NLSE, data)
    fig = PyPlot.plt.figure(constrained_layout=true, figsize=(10, 3*length(data)))
    ax_array = fig.subplots(length(data), 2)
    for (i, (z, u, nfs, err)) in enumerate(data)
        axs = (ax_array[i,1], ax_array[i,2])
        plot_nlse(nlse, z, u; axs=axs)
        errs = @sprintf("%.2e", err)
        axs[1].set_title("nfs=$(nfs), err=$(errs)")
    end
end

nlse = NLSE(0.016, 48.0);

# run work-precision plots for various solvers
# errs, nfs = workprecision(nlse, (
#     (run_fullip, 0.0002 .* ones(30), collect(logrange(1e-5, 1e-1, 30)), 1e-6 .* ones(30), nothing),
#     (run_pieceip, 0.0002 .* ones(30), collect(logrange(1e-5, 1e-1, 30)), 1e-6 .* ones(30), nothing),
#     (run_numfullip, 0.0002 .* ones(30), collect(logrange(1e-5, 1e-1, 30)), 1e-6 .* ones(30), nothing),
#     (run_splitlin, collect(logrange(1e-4, 1e-2, 30)), 1e-6 .* ones(30), 1e-6 .* ones(30), nothing),
#     (run_Luna_weak, 0.0002 .* ones(30), collect(logrange(1e-10, 1e-3, 30)), 1e-10 .* ones(30), nothing),
#     (run_Luna_norm, 0.0002 .* ones(30), collect(logrange(1e-7, 1e-1, 30)), 1e-6 .* ones(30), nothing),
#     (run_newLuna, 0.0002 .* ones(40), collect(logrange(5e-5, 1.2e-1, 40)), 1e-6 .* ones(40), nothing)
# ))
# savefig("scripts/solver_work_precision_nofsal.svg"))

# errs, nfs = workprecision(nlse, (
#     (run_fullip, 0.0002 .* ones(30), collect(logrange(1e-5, 1e-1, 30)), 1e-6 .* ones(30), nothing),
#     (run_pieceip, 0.0002 .* ones(30), collect(logrange(1e-5, 1e-1, 30)), 1e-6 .* ones(30), nothing),
#     (run_Luna_weak, 0.0002 .* ones(30), collect(logrange(1e-10, 1e-3, 30)), 1e-10 .* ones(30), nothing),
#     (run_Luna_norm, 0.0002 .* ones(30), collect(logrange(1e-7, 1e-1, 30)), 1e-6 .* ones(30), nothing),
#     (run_newLuna, 0.0002 .* ones(40), collect(logrange(5e-5, 1.2e-1, 40)), 1e-6 .* ones(40), nothing)
# ))
# savefig(solver_work_precision_nabsbound.svg"))

# # work-precision curves for multiple atol values for new Luna solver
# errs, nfs = workprecision(nlse, (
#     (run_newLuna, 0.0002 .* ones(40), collect(logrange(1e-10, 1.2e-1, 40)), 1e-4 .* ones(40), "1e-4"),
#     (run_newLuna, 0.0002 .* ones(40), collect(logrange(1e-10, 1.2e-1, 40)), 1e-5 .* ones(40), "1e-5"),
#     (run_newLuna, 0.0002 .* ones(40), collect(logrange(1e-10, 1.2e-1, 40)), 1e-6 .* ones(40), "1e-6"),
#     (run_newLuna, 0.0002 .* ones(40), collect(logrange(1e-10, 1.2e-1, 40)), 1e-8 .* ones(40), "1e-8"),
#     (run_newLuna, 0.0002 .* ones(40), collect(logrange(1e-10, 1.2e-1, 40)), 1e-10 .* ones(40), "1e-10"),
# ))
# savefig("solver_work_precision_atolscan.svg")

# # run a comparison to visualise the error
# data = [run_newLuna(nlse; reltol=rtol, abstol=1e-6) for rtol in (1e-1, 6.9e-2, 1.2e-2, 5e-4)]
# plot_nlse_cmp(nlse, data)
# savefig("solver_work_precision_cmp.svg"), dpi=600)
