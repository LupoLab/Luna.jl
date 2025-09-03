module Propagator
import Dates
import Logging
import Printf: @sprintf
import Luna.Utils: format_elapsed
import OrdinaryDiffEq as ODE

mutable struct Printer{DT}
    status_period::Int
    start::DT
    tic::DT
    zmax::Float64
end

function Printer(status_period, zmax)
    Printer(status_period, Dates.now(), Dates.now(), zmax)
end

function printstart(p::Printer)
    p.start = Dates.now()
    p.tic = Dates.now()
    Logging.@info "Starting propagation"
end

function printstep(p::Printer, z, dz)
    if Dates.value(Dates.now() - p.tic) > 1000*p.status_period
        speed = z/(Dates.value(Dates.now() - p.start)/1000)
        eta_in_s = (p.zmax - z)/speed
        if eta_in_s > 356400
            Logging.@info @sprintf("Progress: %.2f %%, ETA: XX:XX:XX, stepsize %.2e",
                                   z/p.zmax*100, dz)
        else
            eta_in_ms = Dates.Millisecond(ceil(eta_in_s*1000))
            etad = Dates.DateTime(Dates.UTInstant(eta_in_ms))
            Logging.@info @sprintf("Progress: %.2f %%, ETA: %s, stepsize %.2e",
                                   z/p.zmax*100, Dates.format(etad, "HH:MM:SS"), dz)
        end
        flush(stderr)
        p.tic = Dates.now()
    end
end

function printstop(p::Printer, integrator)
    totaltime = Dates.now() - p.start
    dtstring = format_elapsed(totaltime)
    Logging.@info @sprintf("Propagation finished in %s", dtstring)
    Logging.@info @sprintf("Steps accepted: %d; rejected: %d",
                           integrator.stats.naccept, integrator.stats.nreject)
    Logging.@info @sprintf("Nonlinear function calls: %d", integrator.stats.nf)
end

abstract type AbstractPropagator end

# For the cases where we can integrate L(z) analytically
struct AnalyticalPropagator{LT, NLT, SFT, PT, AT<:AbstractArray} <: AbstractPropagator
    Li!::LT                 # function to get integrated linear operator at z
    nonlinop!::NLT
    stepfun::SFT
    Litmp::AT
    Eωtmp::AT
    Pωtmp::AT
    p::PT
end

function fcl!(du,u,p,z)
    p.Li!(p.Litmp, z)                     # Get integrated linear operator at z
    @. p.Eωtmp = u * exp(p.Litmp)         # Transform back from interaction picture
    p.nonlinop!(p.Pωtmp, p.Eωtmp, z)      # Apply nonlinear operator
    @. du = p.Pωtmp * exp(-p.Litmp)       # Transform to interaction picture
end

function callbackcl(integrator)
    # The output we want must be transformed back from the interaction picture
    integrator.p.Li!(integrator.p.Litmp, integrator.t)
    @. integrator.p.Eωtmp = integrator.u * exp(integrator.p.Litmp)
    interp = let integrator=integrator
        function interp(z)
            u = integrator.sol(z)
            integrator.p.Li!(integrator.p.Litmp, z)
            @. u * exp(integrator.p.Litmp)
        end
    end
    integrator.p.stepfun(integrator.p.Eωtmp, integrator.t,
                         ODE.get_proposed_dt(integrator), interp)
    printstep(integrator.p.p, integrator.t, ODE.get_proposed_dt(integrator))
    #@. integrator.u = integrator.p.Eωtmp * exp(-integrator.p.Litmp) # copy back as we modify u in stepfun (absorbing boundaries)
    ODE.u_modified!(integrator, false)  # We didn't mutate the solution, so can keep fsal
end

# For a non-constant linear operator, we need to integrate L(z) numerically along with
# the solution. We do this by simply including the integral of linear operator in the state vector.
struct NonConstPropagator{LT, NLT, SFT, PT, AT<:AbstractArray} <: AbstractPropagator
    linop!::LT              # function to get linear operator at z
    nonlinop!::NLT
    stepfun::SFT
    n::Int
    Eωtmp::AT
    Pωtmp::AT
    p::PT
end

function fncl!(du,u,p,z)
    Eω = @views u[1:p.n]                 # Actual Eω
    Li = @views u[p.n+1:end]             # Cumulatively integrated linear operator
    dEω = @views du[1:p.n]
    dLi = @views du[p.n+1:end]
    @. p.Eωtmp = Eω * exp(Li)            # Transform back from interaction picture
    p.nonlinop!(p.Pωtmp, p.Eωtmp, z)     # Apply nonlinear operator
    @. dEω = p.Pωtmp * exp(-Li)          # Transform to interaction picture
    p.linop!(dLi, z)                     # Integrate linear operator
end

function callbackncl(integrator)
    n = integrator.p.n
    Eω = @views integrator.u[1:n]        # Actual Eω
    Li = @views integrator.u[n+1:end]    # Cumulatively integrated linear operator
    # The output we want must be transformed back from the interaction picture
    @. integrator.p.Eωtmp = Eω * exp(Li)
    interp = let integrator=integrator, n=n
        function interp(z)
            u = integrator.sol(z)
            Eω = @views u[1:n]
            Li = @views u[n+1:end]
            @. Eω * exp(Li)
        end
    end
    integrator.p.stepfun(integrator.p.Eωtmp, integrator.t,
                         ODE.get_proposed_dt(integrator), interp)
    printstep(integrator.p.p, integrator.t, ODE.get_proposed_dt(integrator))
    #@. Eω = integrator.p.Eωtmp * exp(-Li) # copy back as we modify u in stepfun (absorbing boundaries)
    ODE.u_modified!(integrator, false)  # We didn't mutate the solution, so can keep fsal
end

# Constant linear operator case--linop is an array
function makeprop(f!, linop::Array{ComplexF64}, Eω0, z, zmax, stepfun, printer, rtol, atol)
    # For a constant linear operator L, the integral is just L*z
    Li! = let linop=linop
        function Li!(out, z)
            @. out = linop * z
        end
    end
    prop = AnalyticalPropagator(Li!, f!, stepfun, similar(Eω0), similar(Eω0), similar(Eω0), printer)
    prob = ODE.ODEProblem(fcl!, Eω0, (z, zmax), prop)
    prob, callbackcl, rtol, atol
end

# For a linop tuple we expect a pair of functions (linop, ilinop) where the second function provides the
# cumulatively integrated linear operator. This is mostly for testing.
function makeprop(f!, linop::Tuple, Eω0, z, zmax, stepfun, printer, rtol, atol)
    Li! = linop[2]
    prop = AnalyticalPropagator(Li!, f!, stepfun, similar(Eω0), similar(Eω0), similar(Eω0), printer)
    prob = ODE.ODEProblem(fcl!, Eω0, (z, zmax), prop)
    prob, callbackcl, rtol, atol
end

# General linop, we integrate numerically along with the solution
function makeprop(f!, linop, Eω0, z, zmax, stepfun, printer, rtol, atol)
    prop = NonConstPropagator(linop, f!, stepfun,
                              length(Eω0), similar(Eω0), similar(Eω0), printer)
    u0 = vcat(Eω0, zero(Eω0))   # Initial linear operator is zero
    #rtol = vcat(ones(length(Eω0))*rtol, ones(length(Eω0))*1e-10)
    #atol = vcat(ones(length(Eω0))*atol, ones(length(Eω0))*1e-10)
    prob = ODE.ODEProblem(fncl!, u0, (z, zmax), prop)
    prob, callbackncl, rtol, atol
end

function propagate(f!, linop, Eω0, z, zmax, stepfun;
                   rtol=1e-3, atol=1e-6, init_dz=1e-4, max_dz=Inf, min_dz=0,
                   status_period=1, solver=:Tsit5, zstops=nothing)
    printer = Printer(status_period, zmax)
    prob, cbfunc, rtol, atol = makeprop(f!, linop, Eω0, z, zmax, stepfun, printer, rtol, atol)
    # We do all saving and stats in a callback called at every step
    cb = ODE.DiscreteCallback((u,t,integrator) -> true, cbfunc, save_positions=(false,false))
    integrator = ODE.init(prob, getproperty(ODE, solver)(); adaptive=true, reltol=rtol, abstol=atol,
                          dt=init_dz, dtmin=min_dz, dtmax=max_dz, callback=cb, tstops=zstops)
    printstart(printer)
    sol = ODE.solve!(integrator)
    printstop(printer, integrator)
    sol
end

end # module
