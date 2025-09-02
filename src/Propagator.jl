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

# For a constant linear operator, we can integrate L(z) analytically
struct ConstPropagator{NLT, SFT, PT, AT<:AbstractArray} <: AbstractPropagator
    L::AT
    nonlinop!::NLT
    stepfun::SFT
    Eωtmp::AT
    Pωtmp::AT
    p::PT
end

function fcl!(du,u,p,z)
    @. p.Eωtmp = u * exp(p.L * z)       # Transform back from interaction picture
    p.nonlinop!(p.Pωtmp, p.Eωtmp, z)    # Apply nonlinear operator
    @. du = p.Pωtmp * exp(-p.L * z)     # Transform to interaction picture
end

function callbackcl(integrator)
    # The output we want must be transformed back from the interaction picture
    @. integrator.p.Eωtmp = integrator.u * exp(integrator.p.L * integrator.t)
    interp = let integrator=integrator
        function interp(z)
            u = integrator(z)
            @. u * exp(integrator.p.L * z)
        end
    end
    integrator.p.stepfun(integrator.p.Eωtmp, integrator.t,
                         ODE.get_proposed_dt(integrator), interp)
    printstep(integrator.p.p, integrator.t, ODE.get_proposed_dt(integrator))
    ODE.u_modified!(integrator, false)  # We didn't mutate the solution, so can keep fsal
end

# For a non-constant linear operator, we need to integrate L(z) numerically along with
# the solution. We do this by simply including the linear operator in the state vector.
struct NonConstPropagator{LT, NLT, SFT, PT, AT<:AbstractArray} <: AbstractPropagator
    linop!::LT
    nonlinop!::NLT
    stepfun::SFT
    n::Int
    Eωtmp::AT
    Pωtmp::AT
    p::PT
end

function fncl!(du,u,p,z)
    Eω = @views u[1:p.n]
    L = @views u[p.n+1:end]
    dEω = @views du[1:p.n]
    dL = @views du[p.n+1:end]
    @. p.Eωtmp = Eω * exp(L)            # Transform back from interaction picture
    p.nonlinop!(p.Pωtmp, p.Eωtmp, z)    # Apply nonlinear operator
    @. dEω = p.Pωtmp * exp(-L)          # Transform to interaction picture
    p.linop!(dL, z)                     # Integrate linear operator
end

function callbackncl(integrator)
    n = integrator.p.n
    Eω = @views integrator.u[1:n]
    L = @views integrator.u[n+1:end]
    # The output we want must be transformed back from the interaction picture
    @. integrator.p.Eωtmp = Eω * exp(L)
    interp = let integrator=integrator, n=n
        function interp(z)
            u = integrator(z)
            Eω = @views u[1:n]
            L = @views u[n+1:end]
            @. Eω * exp(L)
        end
    end
    integrator.p.stepfun(integrator.p.Eωtmp, integrator.t,
                         ODE.get_proposed_dt(integrator), interp)
    printstep(integrator.p.p, integrator.t, ODE.get_proposed_dt(integrator))
    ODE.u_modified!(integrator, false)  # We didn't mutate the solution, so can keep fsal
end

function makeprop(f!, linop::Array{ComplexF64,N}, Eω0, z, zmax, stepfun, printer, rtol, atol) where N
    prop = ConstPropagator(linop, f!, stepfun, similar(Eω0), similar(Eω0), printer)
    prob = ODE.ODEProblem(fcl!, Eω0, (z, zmax), prop)
    prob, callbackcl, rtol, atol
end

function makeprop(f!, linop, Eω0, z, zmax, stepfun, printer, rtol, atol)
    prop = NonConstPropagator(linop, f!, stepfun,
                              length(Eω0), similar(Eω0), similar(Eω0), printer)
    u0 = vcat(Eω0, zero(Eω0))   # Initial linear operator is zero
    #rtol = vcat(ones(length(Eω0))*rtol, ones(length(Eω0))*1e-10)
    #atol = vcat(ones(length(Eω0))*atol, ones(length(Eω0))*1e-15)
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
    ODE.solve!(integrator)
    printstop(printer, integrator)
end

end # module
