module RK45
import HCubature
import Dates
import Logging
import Printf: @sprintf

#Get Butcher tableau etc from separate file (for convenience of changing if wanted)
include("dopri.jl")

# TODO: Maybe change f() from returning to altering output array in place?

mutable struct Stepper{T<:AbstractArray}
    f::Function  # RHS function
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

function Stepper(f::Function, y0, t, dt; tol=1e-6, max_dt=Inf, min_dt=0, locextrap=true)
    k1 = similar(y0)
    k1 .= f(y0, t)
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
    return Stepper(f, y0, copy(y0), similar(y0), yerr, errfunc, ks,
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
        s.ks[ii+1] .= s.f(s.yn, s.t+nodes[ii]*s.dt)
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
        s.dtn = s.dt * max(0.1, 0.8*(s.tol/s.err)^(1/5))
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

function make_fbar(f, linop::AbstractArray, y0)
    fbar_buf = similar(y0)
    fbar = let f=f, linop=linop, fbar_buf=fbar_buf
        function fbar(ybar, t)
            fbar_buf .= exp.(.-linop.*t).*f(exp.(linop.*t).*ybar, t)
            return fbar_buf
        end
    end
end

function make_prop(linop::AbstractArray)
    prop = let linop=linop
        function prop(t)
            return linop.*t
        end
    end
end

function make_fbar(f, linop::Function, y0)
    fbar_buf = similar(y0)
    linop_int = similar(y0)
    fbar = let f=f, linop=linop, linop_int=linop_int, fbar_buf=fbar_buf
        function fbar(ybar, t)
            linop_int .= HCubature.hquadrature(linop, 0, t)[1]
            fbar_buf .= exp.(.-linop_int).*f(exp.(linop_int).*ybar, t)
            return fbar_buf
        end
    end
end

function make_prop(linop::Function)
    prop = let linop=linop
        function prop(t)
            linop_int = HCubature.hquadrature(linop, 0, t)[1]
            return linop_int
        end
    end
end

function solve_precon(f, linop, y0, t, dt, tmax, saveN;
                      kwargs...)

    fbar = make_fbar(f, linop, y0)
    prop = make_prop(linop)

    y0bar = exp.(.-prop(t).*t).*y0

    tout, yout, steps = solve(
        fbar, y0bar, t, dt, tmax, saveN;
        kwargs...)

    for ii in 1:saveN
        yout[fill(:, ndims(y0))..., ii] .*= exp.(prop(tout[ii]))
    end
    return tout, yout, steps
end

function solve(f, y0, t, dt, tmax, saveN;
               tol=1e-6, max_dt=Inf, min_dt=0, locextrap=true,
               status_period=1, repeat_limit=10)
    stepper = Stepper(f, y0, t, dt,
                      tol=tol, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap)

    yout = Array{eltype(y0)}(undef, (size(y0)..., saveN))
    tout = range(t, stop=tmax, length=saveN)

    yout[fill(:, ndims(y0))..., 1] = y0

    steps = 0
    saved = 1
    repeated = 0

    Logging.@info "Starting propagation"
    start = Dates.now()
    tic = Dates.now()
    while stepper.tn <= tmax
        println(steps)
        ok = step!(stepper)
        steps += 1
        if Dates.value(Dates.now()-tic) > 1000*status_period
                Logging.@info @sprintf("%.2f %%, stepsize %.2e, err %.2e",
                    stepper.tn/tmax*100, stepper.dt, stepper.err)
                tic = Dates.now()
        end
        if ok
            while (saved<saveN) && tout[saved+1] < stepper.tn
                ti = tout[saved+1]
                yout[fill(:, ndims(y0))..., saved+1] .= interpolate(stepper, ti)
                saved += 1
            end
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

    return tout, yout, steps
end

end