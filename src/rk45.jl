module RK45
import HCubature
import Dates
import Logging
import Printf: @sprintf

#Get Butcher tableau etc from separate file (for convenience of changing if wanted)
include("dopri.jl")


mutable struct Stepper{T<:AbstractArray}
    f!::Function  # RHS function
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::T  # Interpolant array (see interpolate())
    yerr::T  # solution error estimate (from embedded RK)
    errfunc::Function  # function to calculate err from yerr, y, and yn
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    tol::Float64  # relative tolerance on error
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful
    err::Float64  # error metric to be compared to tol
end

function Stepper(f!::Function, y0, t, dt; tol=1e-6, max_dt=Inf, min_dt=0, locextrap=true)
    k1 = similar(y0)
    f!(k1, y0, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    yerr = similar(y0)
    errbuf = Array{Float64, ndims(y0)}(undef, size(y0))

    # Create errfunc with buffers for speed
    # This is necessary for complex-valued y since Julia can't take the maximum of a
    # complex-valued array without allocating a real-valued array
    errfunc = let abserr=similar(errbuf), absy=similar(errbuf), absyn=similar(errbuf)
        function maxnorm(yerr, y, yn)
            abserr .= abs.(yerr)
            absy .= abs.(y)
            absyn .= abs.(yn)
            absyn .= max.(absy, absyn)
            return maximum(abserr)/maximum(absyn)
        end
    end
    return Stepper(f!, copy(y0), copy(y0), similar(y0), yerr, errfunc, ks,
        float(t), float(t), float(dt), float(dt), tol, float(max_dt), float(min_dt), locextrap, false, 0.0)
end

function step!(s::Stepper)
    # Set new time and stepsize values -- this happens at the beginning because
    # the interpolant still requires the old values after the step has finished
    s.dt = s.dtn
    s.t = s.tn
    s.y .= s.yn
    for ii = 1:6
        s.yn .= s.y
        for jj = 1:ii
            s.yn .+= s.dt*B[ii][jj].*s.ks[jj]
        end
        s.f!(s.ks[ii+1], s.yn, s.t+nodes[ii]*s.dt)
    end

    if s.locextrap
        s.yn .= s.y
        for jj = 1:7
                s.yn .+= s.dt*b5[jj].*s.ks[jj]
        end
    end
    
    fill!(s.yerr, 0)
    for ii = 1:7
        @. s.yerr += s.dt*s.ks[ii]*errest[ii]
    end
    s.err = s.errfunc(s.yerr, s.y, s.yn)
    s.ok = s.err <= s.tol
    if s.ok
        temp = 1.25*(s.err/s.tol)^(1/5)
        s.dtn = s.dt*min(5, 1/temp)
    else
        s.dtn = s.dt * max(0.1, 0.9*(s.tol/s.err)^(1/5))
    end

    if s.dtn > s.max_dt
        s.dtn = s.max_dt
    elseif s.dtn < s.min_dt
        s.dtn = s.min_dt
        s.ok = true
    end
    if s.ok
        s.tn = s.t + s.dt
        s.ks[1] .= s.ks[end]
    else
        s.yn .= s.y
    end
    return s.ok
end

function interpolate(s::Stepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
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


mutable struct PreconStepper{T<:AbstractArray}
    fbar!::Function  # RHS function
    prop!::Function # linear propagator function
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::T  # Interpolant array (see interpolate())
    yerr::T  # solution error estimate (from embedded RK)
    errfunc::Function  # function to calculate err from yerr, y, and yn
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    tol::Float64  # relative tolerance on error
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful
    err::Float64  # error metric to be compared to tol
end

function PreconStepper(f!::Function, linop, y0, t, dt;
                       tol=1e-6, max_dt=Inf, min_dt=0, locextrap=true)
    prop! = make_prop!(linop, y0)
    fbar! = make_fbar!(f!, prop!, y0)
    k1 = similar(y0)
    fbar!(k1, y0, t, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    yerr = similar(y0)
    errbuf = Array{Float64, ndims(y0)}(undef, size(y0))

    # Create errfunc with buffers for speed
    # This is necessary for complex-valued y since Julia can't take the maximum of a
    # complex-valued array without allocating a real-valued array
    errfunc = let abserr=similar(errbuf), absy=similar(errbuf), absyn=similar(errbuf)
        function maxnorm(yerr, y, yn)
            abserr .= abs.(yerr)
            absy .= abs.(y)
            absyn .= abs.(yn)
            absyn .= max.(absy, absyn)
            return maximum(abserr)/maximum(absyn)
        end
    end
    return PreconStepper(fbar!, prop!, copy(y0), copy(y0), similar(y0), yerr, errfunc, ks,
        float(t), float(t), float(dt), float(dt), tol, float(max_dt), float(min_dt), locextrap, false, 0.0)
end

function make_prop!(linop::AbstractArray, y0)
    prop! = let linop=linop
        function prop!(y, t1, t2)
            @. y *= exp(linop*(t2-t1))
        end
    end
end

function make_prop!(linop::Function, y0)
    linop_int = similar(y0)
    prop! = let linop=linop, linop_int=linop_int
        function prop!(y, t1, t2)
            # linop_int .= HCubature.hquadrature(linop, t1, t2)[1]
            linop_int .= linop(t1).*(t2-t1)
            @. y *= exp(linop_int)
        end
    end
end

function make_fbar!(f!, prop!, y0)
    y = similar(y0)
    fbar! = let f! = f!, prop! = prop!, y=y
        function fbar!(out, ybar, t1, t2)
            y .= ybar
            prop!(y, t1, t2)
            f!(out, y, t2)
            prop!(out, t2, t1)
        end
    end
end

function step!(s::PreconStepper)
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
            s.yn .+= s.dt*B[ii][jj].*s.ks[jj]
        end
        s.fbar!(s.ks[ii+1], s.yn, s.t, s.t+nodes[ii]*s.dt)
    end

    if s.locextrap
        s.yn .= s.y
        for jj = 1:7
                s.yn .+= s.dt*b5[jj].*s.ks[jj]
        end
    end
    
    fill!(s.yerr, 0)
    for ii = 1:7
        @. s.yerr += s.dt*s.ks[ii]*errest[ii]
    end
    s.err = s.errfunc(s.yerr, s.y, s.yn)
    s.ok = s.err <= s.tol
    if s.ok
        temp = 1.25*(s.err/s.tol)^(1/5)
        s.dtn = s.dt*min(5, 1/temp)
    else
        s.dtn = s.dt * max(0.1, 0.9*(s.tol/s.err)^(1/5))
    end

    if s.dtn > s.max_dt
        s.dtn = s.max_dt
    elseif s.dtn < s.min_dt
        s.dtn = s.min_dt
        s.ok = true
    end
    if s.ok
        s.tn = s.t + s.dt
        s.ks[1] .= s.ks[end]
    else
        s.yn .= s.y
    end
    return s.ok
end

function interpolate(s::PreconStepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
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

function solve_precon(f!, linop, y0, t, dt, tmax, saveN;
                    tol=1e-6, max_dt=Inf, min_dt=0, locextrap=true,
                    kwargs...)
    stepper = PreconStepper(f!, linop, y0, t, dt,
                      tol=tol, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap)
    return solve(stepper, tmax, saveN; kwargs...)
end

function solve(f!, y0, t, dt, tmax, saveN;
               tol=1e-6, max_dt=Inf, min_dt=0, locextrap=true,
               kwargs...)
    stepper = Stepper(f!, y0, t, dt,
                      tol=tol, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap)
    return solve(stepper, tmax, saveN; kwargs...)
end

function solve(s, tmax, saveN; stepfun=donothing!, status_period=1, repeat_limit=10)
    yout = Array{eltype(s.y)}(undef, (size(s.y)..., saveN))
    tout = range(s.t, stop=tmax, length=saveN)

    yout[fill(:, ndims(s.y))..., 1] = s.y

    steps = 0
    saved = 1
    repeated = 0

    Logging.@info "Starting propagation"
    start = Dates.now()
    tic = Dates.now()
    while s.tn <= tmax
        # println(steps)
        ok = step!(s)
        steps += 1
        if Dates.value(Dates.now()-tic) > 1000*status_period
                Logging.@info @sprintf("%.2f %%, stepsize %.2e, err %.2e",
                    s.tn/tmax*100, s.dt, s.err)
                tic = Dates.now()
        end
        if ok
            while (saved<saveN) && tout[saved+1] < s.tn
                ti = tout[saved+1]
                yout[fill(:, ndims(s.y))..., saved+1] .= interpolate(s, ti)
                saved += 1
            end
            stepfun(s.yn)
            repeated = 0
        else
            repeated += 1
            if repeated > repeat_limit
                error("Reached limit for step repetition ($repeat_limit)")
            end
        end
    end
    Logging.@info @sprintf("Propagation finished in %.3f seconds, %d steps",
                           Dates.value(Dates.now()-start)/1000, steps)

    return collect(tout), yout, steps
end

function donothing!(x)
end

end