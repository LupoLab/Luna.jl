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
    steps::Int
    zmax::Float64
end

function Printer(status_period, zmax)
    Logging.@info "Starting propagation"
    Printer(status_period, Dates.now(), Dates.now(), 0, zmax)
end

function printstep(p::Printer, z, dz)
    p.steps += 1
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

function printstop(p::Printer)
    totaltime = Dates.now() - p.start
    dtstring = format_elapsed(totaltime)
    Logging.@info @sprintf("Propagation finished in %s, %d steps",
                           dtstring, p.steps)
end

abstract type AbstractPropagator end

struct ConstPropagator{NLT} <: AbstractPropagator
    L::Vector{ComplexF64}
    nonlinop!::NLT
    Eωtmp::Vector{ComplexF64}
    Pωtmp::Vector{ComplexF64}
end

#function make_propagator(f!, linop::Vector{ComplexF64}, u0::Vector{ComplexF64})
#    prop = ConstPropagator(linop, f!, similar(u0), similar(u0))
#end

function fcl!(du,u,p,z)
    @. p.Eωtmp = u * exp(p.L * z)
    p.nonlinop!(p.Pωtmp, p.Eωtmp, z)
    @. du = p.Pωtmp * exp(-p.L * z)
end

struct NonConstPropagator{LT, NLT, SFT, PT} <: AbstractPropagator
    linop!::LT
    nonlinop!::NLT
    stepfun::SFT
    n::Int
    Eωtmp::Vector{ComplexF64}
    Pωtmp::Vector{ComplexF64}
    p::PT
end

function fncl!(du,u,p,z)
    Eω = @views u[1:p.n]
    L = @views u[p.n+1:end]
    dEω = @views du[1:p.n]
    dL = @views du[p.n+1:end]
    @. p.Eωtmp = Eω * exp(L)
    p.nonlinop!(p.Pωtmp, p.Eωtmp, z)
    @. dEω = p.Pωtmp * exp(-L)
    p.linop!(dL, z)
end

function callbackncl(integrator)
    n = integrator.p.n
    Eω = @views integrator.p.u[1:n]
    L = @views integrator.p.u[n+1:end]
    @. integrator.p.Eωtmp = Eω * exp(L)
    interp = let integrator=integrator, n=n
        function interp(z)
            u = integrator(z)
            Eω = @views u[1:n]
            L = @views u[n+1:end]
            @. Eω * exp(L)
        end
    end
    integrator.p.stepfun(integrator.p.Eωtmp, integrator.t, ODE.get_proposed_dt(integrator), interp)
    printstep(integrator.p.p, integrator.t, ODE.get_proposed_dt(integrator))
    u_modified!(integrator, false)
end

function propagate(f!, linop!, Eω0, z, zmax, stepfun;
                   rtol=1e-6, atol=1e-10, max_dz=Inf, min_dz=0, status_period=1)
    p = Printer(status_period, zmax)
    prop = NonConstPropagator(linop!, f!, stepfun, length(Eω0), similar(Eω0), similar(Eω0), p)
    u0 = vcat(Eω0, zero(Eω0))
    prob = ODE.ODEProblem(fncl!, u0, (z, zmax), prop)
    cb = ODE.DiscreteCallback((u,t,integrator) -> true, callbackncl, save_positions=(false,false))
    integrator = ODE.init(prob, ODE.Tsit5(); adaptive=true, reltol=rtol, abstol=atol,
                          dtmin=min_dz, dtmax=max_dz, callback=cb)
    ODE.solve!(integrator)
    printstop(p)
end

end # module
