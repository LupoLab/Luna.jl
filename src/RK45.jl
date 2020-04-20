module RK45
import Dates
import Logging
import Printf: @sprintf

#Get Butcher tableau etc from separate file (for convenience of changing if wanted)
include("dopri.jl")

function solve(f!, y0, t, dt, tmax;
               rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true,
               norm=weaknorm,
               kwargs...)
    stepper = Stepper(f!, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    return solve(stepper, tmax; kwargs...)
end

function solve_precon(f!, linop, y0, t, dt, tmax;
                    rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true, norm=weaknorm,
                    kwargs...)
    stepper = PreconStepper(f!, linop, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    return solve(stepper, tmax; kwargs...)
end

function solve(s, tmax; stepfun=donothing!, output=false, outputN=201,
                        status_period=1, repeat_limit=10)
    if output
        yout = Array{eltype(s.y)}(undef, (size(s.y)..., outputN))
        tout = range(s.t, stop=tmax, length=outputN)
        saved = 1
        yout[fill(:, ndims(s.y))..., 1] = s.y
    end

    steps = 0
    repeated = 0
    repeated_tot = 0

    Logging.@info "Starting propagation"
    start = Dates.now()
    tic = Dates.now()
    while s.tn <= tmax
        ok = step!(s)
        steps += 1
        if Dates.value(Dates.now()-tic) > 1000*status_period
            speed = s.tn/(Dates.value(Dates.now()-start)/1000)
            eta_in_s = (tmax-s.tn)/(speed)
            if eta_in_s > 356400
                Logging.@info @sprintf("Progress: %.2f %%, ETA: XX:XX:XX, stepsize %.2e, err %.2f, repeated %d",
                s.tn/tmax*100, s.dt, s.err, repeated_tot)
            else
                eta_in_ms = Dates.Millisecond(ceil(eta_in_s*1000))
                etad = Dates.DateTime(Dates.UTInstant(eta_in_ms))
                Logging.@info @sprintf("Progress: %.2f %%, ETA: %s, stepsize %.2e, err %.2f, repeated %d",
                    s.tn/tmax*100, Dates.format(etad, "HH:MM:SS"), s.dt, s.err, repeated_tot)
            end
            tic = Dates.now()
        end
        if ok
            if output
                while (saved<outputN) && tout[saved+1] < s.tn
                    ti = tout[saved+1]
                    yout[fill(:, ndims(s.y))..., saved+1] .= interpolate(s, ti)
                    saved += 1
                end
            end
            stepfun(s.yn, s.tn, s.dt, t -> interpolate(s, t))
            repeated = 0
        else
            repeated += 1
            repeated_tot += 1
            if repeated > repeat_limit
                error("Reached limit for step repetition ($repeat_limit)")
            end
        end
    end
    Logging.@info @sprintf("Propagation finished in %.3f seconds, %d steps",
                           Dates.value(Dates.now()-start)/1000, steps)

    if output
        return collect(tout), yout, steps
    else      
        return nothing
    end
end


mutable struct Stepper{T<:AbstractArray, F, nT}
    f!::F  # RHS function
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::T  # Interpolant array (see interpolate())
    yerr::T  # solution error estimate (from embedded RK)
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    rtol::Float64  # relative tolerance on error
    atol::Float64  # absolute tolerance on error
    safety::Float64  # safety factor for stepsize control
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful
    err::Float64  # error metric to be compared to tol
    errlast::Float64  # error of the most recent successful step
    norm::nT # function to calculate error metric, defaults to RK45.weaknorm
end

function Stepper(f!, y0, t, dt; rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true, norm=weaknorm)
    k1 = similar(y0)
    f!(k1, y0, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    yerr = similar(y0)
    return Stepper(f!, copy(y0), copy(y0), similar(y0), yerr, ks,
        float(t), float(t), float(dt), float(dt),
        float(rtol), float(atol), float(safety), float(max_dt), float(min_dt),
        locextrap, false, 0.0, 0.0, norm)
end

mutable struct PreconStepper{T<:AbstractArray, F, P, nT}
    fbar!::F  # RHS callable
    prop!::P # linear propagator callable
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::T  # Interpolant array (see interpolate())
    yerr::T  # solution error estimate (from embedded RK)
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    rtol::Float64  # relative tolerance on error
    atol::Float64  # absolute tolerance on error
    safety::Float64  # safety factor for stepsize control
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful
    err::Float64  # error metric to be compared to tol
    errlast::Float64  # error of the most recent successful step
    norm::nT  # function to calculate error metric, defaults to RK45.weaknorm
end

function PreconStepper(f!, linop, y0, t, dt;
                       rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true, norm=weaknorm)
    prop! = make_prop!(linop, y0)
    fbar! = make_fbar!(f!, prop!, y0)
    k1 = similar(y0)
    fbar!(k1, y0, t, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    yerr = similar(y0)

    return PreconStepper(fbar!, prop!, copy(y0), copy(y0), similar(y0), yerr, ks,
        float(t), float(t), float(dt), float(dt), float(rtol), float(atol), float(safety),
        float(max_dt), float(min_dt), locextrap, false, 0.0, 0.0, norm)
end

function step!(s)    
    evaluate!(s)

    if s.locextrap
        s.yn .= s.y
        for jj = 1:7
            b5[jj] == 0 || (s.yn .+= s.dt*b5[jj].*s.ks[jj])
        end
    end
    
    fill!(s.yerr, 0)
    for ii = 1:7
        errest[ii] == 0 || (@. s.yerr += s.dt*s.ks[ii]*errest[ii])
    end
    s.err = s.norm(s.yerr, s.y, s.yn, s.rtol, s.atol)
    s.ok = s.err <= 1
    stepcontrolPI!(s)
    if s.ok
        s.tn = s.t + s.dt
        s.ks[1] .= s.ks[end]
    else
        s.yn .= s.y
    end
    return s.ok
end

function evaluate!(s::Stepper)
    # Set new time and stepsize values -- this happens at the beginning because
    # the interpolant still requires the old values after the step has finished
    s.dt = s.dtn
    s.t = s.tn
    s.y .= s.yn
    for ii = 1:6
        s.yn .= s.y
        for jj = 1:ii
            B[ii][jj] == 0 || (s.yn .+= s.dt*B[ii][jj].*s.ks[jj])
        end
        s.f!(s.ks[ii+1], s.yn, s.t+nodes[ii]*s.dt)
    end
end

function evaluate!(s::PreconStepper)
    # Set new time and stepsize values -- this happens at the beginning because
    # the interpolant still requires the old values after the step has finished
    s.y .= s.yn
    s.prop!(s.y, s.t, s.tn)
    s.prop!(s.ks[1], s.t, s.tn)
    s.dt = s.dtn
    s.t = s.tn
    for ii = 1:6
        s.yn .= s.y
        for jj = 1:ii
            B[ii][jj] == 0 || (s.yn .+= s.dt*B[ii][jj].*s.ks[jj])
        end
        s.fbar!(s.ks[ii+1], s.yn, s.t, s.t+nodes[ii]*s.dt)
    end
end

"Interpolate solution, aka dense output."
function interpolate(s::Stepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    elseif ti == s.tn
        return s.yn
    end
    σ = (ti - s.t)/s.dt
    σp = map(p -> σ^p, range(1, stop=4))
    b = sum(σp.*interpC, dims=1)
    fill!(s.yi, 0)
    for ii = 1:7
         s.yi .+= s.ks[ii].*b[ii]
    end
    return @. s.y + s.dt.*s.yi
end

"Interpolate solution, aka dense output."
function interpolate(s::PreconStepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    elseif ti == s.tn
        return s.yn
    end
    σ = (ti - s.t)/s.dt
    σp = map(p -> σ^p, range(1, stop=4))
    b = sum(σp.*interpC, dims=1)
    fill!(s.yi, 0)
    for ii = 1:7
         s.yi .+= s.ks[ii].*b[ii]
    end
    out =  @. s.y + s.dt.*s.yi
    s.prop!(out, s.t, ti)
    return out
end

"Make propagator for the case of constant linear operator"
function make_prop!(linop::AbstractArray, y0)
    prop! = let linop=linop
        function prop!(y, t1, t2, bwd=false)
            if bwd
                @. y *= exp(linop*(t1-t2))
            else
                @. y *= exp(linop*(t2-t1))
            end
        end
    end
end

"Make propagator for the case of non-constant linear operator"
function make_prop!(linop!, y0)
    linop_int = similar(y0)
    function prop!(y, t1, t2, bwd=false)
        # linop is always evaluated at later time, even for backward propagation
        linop!(linop_int, t2) 
        linop_int .*= bwd ? (t1-t2) : (t2-t1)
        @. y *= exp(linop_int)
    end
    return prop!
end

"Make closure for the pre-conditioned RHS function."
function make_fbar!(f!, prop!, y0)
    y = similar(y0)
    fbar! = let f! = f!, prop! = prop!, y=y
        function fbar!(out, ybar, t1, t2)
            y .= ybar
            prop!(y, t1, t2) # propagate to t2
            f!(out, y, t2) # evaluate RHS function
            prop!(out, t1, t2, true) # propagate back to t1
        end
    end
end

"Max-ish norm (from Dane Austin's code, no idea where he got it from)."
function maxnorm(yerr, y, yn, rtol, atol)
    maxerr = 0
    maxy = 0
    for ii in eachindex(yerr)
        maxerr = max(maxerr, abs(yerr[ii]))
        maxy = max(maxy, max(abs(y[ii]), abs(yn[ii])))
    end
    return maxerr/(atol + rtol*maxy)
end

"Alternative form of max-ish norm."
function maxnorm_ratio(yerr, y, yn, rtol, atol)
    m = 0
    for ii in eachindex(yerr)
        den = atol + rtol*max(abs(y[ii]), abs(yn[ii]))
        m = max(abs(yerr[ii])/den, m)
    end
    return m
end

"Semi-norm as used in DifferentialEquations.jl, see Hairer, Solving Ordinary Differential
Equations: Nonstiff Problems, eq. (4.11) (p.168 of the second revised edition)."
function normnorm(yerr, y, yn, rtol, atol)
    s = 0
    for ii in eachindex(yerr)
        s += abs2(yerr[ii]/(atol + rtol*max(abs(y[ii]), abs(yn[ii]))))
    end
    sqrt(s/length(yerr))
end

"'Weak' norm as used in fnfep."
function weaknorm(yerr, y, yn, rtol, atol)
    sy = 0
    syn = 0
    syerr = 0
    for ii in eachindex(yerr)
        sy += abs2(y[ii])
        syn += abs2(yn[ii])
        syerr += abs2(yerr[ii])
    end
    errwt = max(max(sqrt(sy), sqrt(syn)), atol)
    return sqrt(syerr)/rtol/errwt
end

"Simple proportional error controller, see e.g. Hairer eq. (4.13)."
function stepcontrolP!(s)
    if s.ok
        s.dtn = s.dt * min(5, s.safety*(s.err)^(-1/5))
    else
        if !isfinite(s.err) # check for NaN or Inf
            s.dtn = s.dt/2  # if we have one then we're in bug trouble so halve the step size
        else
            s.dtn = s.dt * max(0.1, s.safety*(s.err)^(-1/5))
        end
    end
    steplims!(s)
end

"Proportional-integral error controller, aka Lund stabilisation.
See G. Söderlind and L. Wang, J. Comput. Appl. Math. 185, 225 (2006).
"
function stepcontrolPI!(s)
    β1 = 3/5 / 5
    β2 = -1/5 / 5
    ε = 0.8
    if s.ok
        s.errlast == 0 && (s.errlast = 1)
        fac = s.safety * (ε/s.err)^β1 * (ε/s.errlast)^β2
        # (0.99 <= fac <= 1.01) && (fac = 1.0)
        s.dtn = fac * s.dt
        s.errlast = s.err
    else
        if !isfinite(s.err) # check for NaN or Inf
            s.dtn = s.dt/2  # if we have one then we're in bug trouble so halve the step size
        else
            s.dtn = s.dt * max(0.1, s.safety*(s.err)^(-1/5))
        end
    end
    steplims!(s)
end

"Apply user-defined limits on step size."
function steplims!(s)
    if s.dtn > s.max_dt
        s.dtn = s.max_dt
    elseif s.dtn < s.min_dt
        s.dtn = s.min_dt
        s.ok = true
    end
end

function donothing!(y, z, dz, interpolant)
end

end