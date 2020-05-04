module Maths
import FiniteDifferences
import LinearAlgebra: Tridiagonal, mul!, ldiv!
import SpecialFunctions: erf, erfc
import StaticArrays: SVector
import Random: AbstractRNG, randn, GLOBAL_RNG
import FFTW
import Luna
import Luna.Utils: saveFFTwisdom, loadFFTwisdom
import Roots: fzero
import Dierckx

#= Pre-created finite difference methods for speed.
    Above order=7, this would create overflow errors in central_fwm() =#
FDMs = [FiniteDifferences.central_fdm(order+6, order) for order=1:7]

"Calculate derivative of function f(x) at value x using finite differences"
function derivative(f, x, order::Integer)
    if order == 0
        return f(x)
    else
        # use 5th order central finite differences with 4 adaptive steps
        scale = abs(x) > 0 ? x : 1.0
        FiniteDifferences.fdm(FDMs[order], y->f(y*scale), x/scale, adapt=2)/scale^order
    end
end

"Gaussian or hypergaussian function (with std dev σ as input)"
function gauss(x, σ; x0 = 0, power = 2)
    return exp(-1/2 * ((x-x0)/σ)^power)
end

"Gaussian or hypergaussian function (with FWHM as input)"
function gauss(x; x0 = 0, power = 2, fwhm)
    σ = fwhm / (2 * (2 * log(2))^(1 / power))
    return gauss(x, σ, x0 = x0, power=power)
end

"""
    randgauss([rng=GLOBAL_RNG], μ, σ, args...)

Generate random numbers with normal distribution centred on `μ` with standard deviation `σ`.

Any additional `args` are passed onto `randn` and can be used to specify the output dimensions.
"""
randgauss(μ, σ, args...) = randgauss(GLOBAL_RNG, μ, σ, args...)
randgauss(rng::AbstractRNG, μ, σ, args...) = σ*randn(rng, args...) .+ μ

"nth moment of the vector y"
function moment(x::Vector, y::Vector, n = 1)
    if length(x) ≠ length(y)
        throw(DomainError(x, "x and y must have same length"))
    end
    return sum(x.^n .* y) / sum(y)
end

"nth moment of multi-dimensional array y along dimension dim"
function moment(x::Vector, y, n = 1; dim = 1)
    if size(y, dim) ≠ length(x)
        throw(DomainError(y, "y must be of same length as x along dim"))
    end
    xshape = ones(Integer, ndims(y))
    xshape[dim] = length(x)
    return sum(reshape(x, Tuple(xshape)).^n .* y, dims=dim) ./ sum(y, dims=dim)
end

"RMS width of distribution y on axis x"
function rms_width(x::Vector, y::Vector; dim = 1)
    return sqrt(moment(x, y, 2) - moment(x, y, 1)^2)
end

function rms_width(x::Vector, y; dim = 1)
    return sqrt.(moment(x, y, 2, dim = dim) - moment(x, y, 1, dim = dim).^2)
end

"""
    fwhm(x, y; method=:linear, baseline=false, minmax=:min)

Calculate the full width at half maximum (FWHM) of `y` on the axis `x`

`method` can be `:spline` or `:nearest`. `:spline` uses a [`CSpline`](@ref), whereas
`:nearest` finds the closest values either side of the crossing point and interpolates linearly.

If `baseline` is true, the width is not taken at
half the global maximum, but at half of the span of `y`.

`minmax` determines whether the FWHM is taken at the narrowest (`:min`) or the widest (`:max`)
point of y.
"""
function fwhm(x, y; method=:linear, baseline=false, minmax=:min)
    minmax in (:min, :max) || error("minmax has to be :min or :max")
    if baseline
        val = minimum(y) + 0.5*(maximum(y) - minimum(y))
    else
        val = 0.5*maximum(y)
    end
    if !(method in (:spline, :linear, :nearest))
        error("Unknown FWHM method $method")
    end
    maxidx = argmax(y)
    xmax = x[maxidx]
    if method in (:nearest, :linear)
        try
            if minmax == :min
                lefti = findlast((x .< xmax) .& (y .< val))
                righti = findfirst((x .> xmax) .& (y .< val))
                (method == :nearest) && return abs(x[lefti] - x[righti])
                left = linterpx(x[lefti], x[lefti+1], y[lefti], y[lefti+1], val)
                right = linterpx(x[righti-1], x[righti], y[righti-1], y[righti], val)
            else
                lefti = findfirst((x .< xmax) .& (y .> val))
                righti = findlast((x .> xmax) .& (y .> val))
                (method == :nearest) && return abs(x[lefti] - x[righti])
                left = linterpx(x[lefti-1], x[lefti], y[lefti-1], y[lefti], val)
                right = linterpx(x[righti], x[righti+1], y[righti], y[righti+1], val)
            end
            return abs(right - left)
        catch
            return NaN
        end
    elseif method == :spline
        #spline method
        try
            spl = BSpline(x, y .- val)
            r = roots(spl; maxn=10000)
            rleft = r[r .< xmax]
            rright = r[r .> xmax]
            if minmax == :min
                return abs(minimum(rright) - maximum(rleft))
            else
                return abs(maximum(rright) - minimum(rleft))
            end
        catch e
            return NaN
        end
    else
        error("Unknown FWHM method $method")
    end
end

"""
    linterpx(x1, x2, y1, y2, val)

Given two points on a straight line, `(x1, y1)` and `(x2, y2)`, calculate the value of `x` 
at which this straight line intercepts with `val`.

# Examples
```jldoctest
julia> x1 = 0; x2 = 1; y1 = 0; y2 = 2; # y = 2x
julia> linterpx(x1, x2, y1, y2, 0.5)
0.25
```
"""
function linterpx(x1, x2, y1, y2, val)
    slope = (y2-y1)/(x2-x1)
    icpt = y1 - slope*x1
    (val-icpt)/slope
end

"""
    hwhm(f, x0=0; direction=:fwd)

Find the value `x` where the function `f(x)` drops to half of its maximum, which is located
at `x0`. For `direction==:fwd`, search in the region `x > x0`, for :bwd, search in `x < x0`.
"""
function hwhm(f, x0=0; direction=:fwd)
    m = f(x0)
    fhalf(x) = f(x) - 0.5*m
    if direction == :fwd
        xhw = fzero(fhalf, x0, Inf)
    elseif direction == :bwd
        xhw = fzero(fhalf, -Inf, x0)
    end
    return abs(xhw - x0)
end

"""
    cumtrapz!([out, ] y, x; dim=1)

Trapezoidal integration for multi-dimensional arrays or vectors, in-place or with output array.

If `out` is omitted, `y` is integrated in place. Otherwise the result is placed into `out`.

`x` can be an array (the x axis) or a number (the x axis spacing).
"""
function cumtrapz! end

function cumtrapz!(y, x; dim=1)
    idxlo = CartesianIndices(size(y)[1:dim-1])
    idxhi = CartesianIndices(size(y)[dim+1:end])
    _cumtrapz!(y, x, idxlo, idxhi)
end

"""
    _cumtrapz!([out, ] y, x, idxlo, idxhi)

Inner function for multi-dimensional `cumtrapz!` - uses 1-D routine internally
"""
function _cumtrapz!(y, x, idxlo, idxhi)
    for lo in idxlo
        for hi in idxhi
            cumtrapz!(view(y, lo, :, hi), x)
        end
    end
end

function cumtrapz!(y::T, x) where T <: Union{SubArray, Vector}
    tmp = y[1]
    y[1] = 0
    for i in 2:length(y)
        tmp2 = y[i]
        y[i] = y[i-1] + 1//2 * (tmp + tmp2) * _dx(x, i)
        tmp = tmp2
    end
end

function cumtrapz!(out, y, x; dim=1)
    idxlo = CartesianIndices(size(y)[1:dim-1])
    idxhi = CartesianIndices(size(y)[dim+1:end])
    _cumtrapz!(out, y, x, idxlo, idxhi)
end

function _cumtrapz!(out, y, x, idxlo, idxhi)
    for lo in idxlo
        for hi in idxhi
            cumtrapz!(view(out, lo, :, hi), view(y, lo, :, hi), x)
        end
    end
end

function cumtrapz!(out, y::Union{SubArray, Vector}, x)
    out[1] = 0
    for i in 2:length(y)
        out[i] = out[i-1]+ 1//2*(y[i-1] + y[i])*_dx(x, i)
    end
end

"""
    _dx(x, i)

Calculate the axis spacing at index `i` given an axis `x`. If `x` is a number, interpret this
as `δx` directly
"""
_dx(x, i) = x[i] - x[i-1]
_dx(δx::Number, i) = δx


"""
    cumtrapz(y, x; dim=1)

Calculate the cumulative trapezoidal integral of `y`.

`x` can be an array (the x axis) or a number (the x axis spacing).
"""
function cumtrapz(y, x; dim=1)
    out = similar(y)
    cumtrapz!(out, y, x; dim=dim) 
    return out
end

"Normalise an array by its maximum value"
function normbymax(x, dims)
    return x ./ maximum(x; dims = dims)
end

function normbymax(x)
    return x ./ maximum(x)
end

"Normalised log10 i.e. maximum of output is 0"
function log10_norm(x)
    return log10.(normbymax(x))
end

function log10_norm(x, dims)
    return log10.(normbymax(x, dims = dims))
end

"Window based on the error function"
function errfun_window(x, xmin, xmax, width)
    return @. 0.5 * (erf((x - xmin) / width) + erfc((x - xmax) / width) - 1)
end

"Error function window but with different widths on each side"
function errfun_window(x, xmin, xmax, width_left, width_right)
    return @. 0.5 * (erf((x - xmin) / width_left) + erfc((x - xmax) / width_right) - 1)
end

"""
    planck_taper(x, xmin, xmax, ε)

Planck taper window as defined in the paper (https://arxiv.org/pdf/1003.2939.pdf eq(7)).

#Arguments
-`xmin` : lower limit (window is 0 here)
-`xmax` : upper limit (window is 0 here)
-`ε` : fraction of window width over which to increase from 0 to 1
"""
function planck_taper(x::AbstractArray, xmin, xmax, ε)
    x0 = (xmax + xmin) / 2
    xc = x .- x0
    X = (xmax - xmin)
    x1  = -X / 2
    x2 = -X / 2 * (1 - 2ε)
    x3 = X / 2 * (1 - 2ε)
    x4 = X / 2
    return _taper(xc, x1, x2, x3, x4)
end

"""
    planck_taper(x, left0, left1, right1, right0)

Planck taper window, but finding the taper width by defining 4 points:
The window increases from 0 to 1 between `left0` and `left1`, and then drops again
to 0 between `right1` and `right0`.
"""
function planck_taper(x::AbstractArray, left0, left1, right1, right0)
    x0 = (right0 + left0) / 2
    xc = x .- x0
    X = right0 - left0
    εleft = abs(left1 - left0) / X
    εright = abs(right0 - right1) / X
    x1  = -X / 2
    x2 = -X / 2 * (1 - 2εleft)
    x3 = X / 2 * (1 - 2εright)
    x4 = X / 2
    return _taper(xc, x1, x2, x3, x4)
end

"""
Planck taper helper function, common to both versions of planck_taper
"""
function _taper(xc, x1, x2, x3, x4)
    idcs12 = x1 .< xc .< x2
    idcs23 = x2 .<= xc .<= x3
    idcs34 = x3 .< xc .< x4
    z12 = @. (x2 - x1) / (xc[idcs12] - x1) + (x2 - x1) / (xc[idcs12] - x2)
    z34 = @. (x3 - x4) / (xc[idcs34] - x3) + (x3 - x4) / (xc[idcs34] - x4)
    out = zero(xc)
    @. out[idcs12] = 1 / (1 + exp(z12))
    @. out[idcs23] = 1
    @. out[idcs34] = 1 / (1 + exp(z34))
    return out
end

"""
Hypergaussian window
"""
function hypergauss_window(x, xmin, xmax, power = 10)
    fw = xmax - xmin
    x0 = (xmax + xmin) / 2
    return gauss(x, x0 = x0, fwhm = fw, power = power)
end

"""
    gabor(t, A, ts, fw)

Compute the Gabor transform (aka spectrogram or time-gated Fourier transform) of the vector
`A`, sampled on axis `t`, with windows centred at `ts` and a window FWHM of `fw`.
"""
function gabor(t::Vector, A::Vector, ts, fw)
    tmp = Array{eltype(A), 2}(undef, (length(t), length(ts)));
    for (ii, ti) in enumerate(ts)
        tmp[:, ii] = A .* Maths.gauss.(t; x0=ti, fwhm=fw)
    end
    _gaborFT(tmp)
end

_gaborFT(x::Array{T, 2}) where T <: Real = FFTW.rfft(x, 1)
_gaborFT(x::Array{T, 2}) where T <: Complex = FFTW.fft(x, 1)

"""
    hilbert(x; dim=1)

Compute the Hilbert transform, i.e. find the analytic signal from a real signal.
"""
function hilbert(x::Array{T,N}; dim = 1) where T <: Real where N
    xf = FFTW.fft(x, dim)
    n1 = size(xf, dim)÷2
    n2 = size(xf, dim)
    idxlo = CartesianIndices(size(xf)[1:dim - 1])
    idxhi = CartesianIndices(size(xf)[dim + 1:end])
    xf[idxlo, 2:n1, idxhi] .*= 2
    xf[idxlo, (n1+1):n2, idxhi] .= 0
    return FFTW.ifft(xf, dim)
end

"""
    plan_hilbert!(x; dim=1)

Pre-plan a Hilbert transform.

Returns a closure `hilbert!(out, x)` which places the Hilbert transform of `x` in `out`.
"""
function plan_hilbert!(x; dim=1)
    loadFFTwisdom()
    FT = FFTW.plan_fft(x, dim, flags=Luna.settings["fftw_flag"])
    saveFFTwisdom()
    xf = Array{ComplexF64}(undef, size(FT))
    idxlo = CartesianIndices(size(xf)[1:dim - 1])
    idxhi = CartesianIndices(size(xf)[dim + 1:end])
    n1 = size(xf, dim)÷2
    n2 = size(xf, dim)
    xc = complex(x)
    function hilbert!(out, x)
        copyto!(xc, x)
        mul!(xf, FT, xc)
        xf[idxlo, 2:n1, idxhi] .*= 2
        xf[idxlo, (n1+1):n2, idxhi] .= 0
        ldiv!(out, FT, xf)
    end
    return hilbert!
end

"""
    plan_hilbert(x; dim=1)

Pre-plan a Hilbert transform.

Returns a closure `hilbert(x)` which returns the Hilbert transform of `x` without allocation.

!!! warning
    The closure returned always returns a reference to the same array buffer, which could lead
    to unexpected results if it is called from more than one location. To avoid this the array
    should either: (i) only be used in the same code segment; (ii) only be used transiently
    as part of a larger computation; (iii) copied.
"""
function plan_hilbert(x; dim=1)
    out = complex(x)
    hilbert! = plan_hilbert!(x, dim=dim)
    function hilbert(x)
        hilbert!(out, x)
    end
    return hilbert
end

"""
Oversample (smooth) an array by 0-padding in the frequency domain
"""
function oversample(t, x::Array{T,N}; factor::Integer = 4, dim = 1) where T <: Real where N
    if factor == 1
        return t, x
    end
    xf = FFTW.rfft(x, dim)

    len = size(xf, dim)
    newlen_t = factor * length(t)
    if iseven(newlen_t)
        newlen_ω = Int(newlen_t / 2 + 1)
    else
        newlen_ω = Int((newlen_t + 1) / 2)
    end
    δt = t[2] - t[1]
    δto = δt / factor
    Nto = collect(0:newlen_t - 1)
    to = t[1] .+ Nto .* δto

    shape = collect(size(xf))
    shape[dim] = newlen_ω
    xfo = zeros(eltype(xf), Tuple(shape))
    idxlo = CartesianIndices(size(xfo)[1:dim - 1])
    idxhi = CartesianIndices(size(xfo)[dim + 1:end])
    xfo[idxlo, 1:len, idxhi] .= factor .* xf
    return to, FFTW.irfft(xfo, newlen_t, dim)
end

"""
Oversampling for complex-valued arryas (e.g. envelope fields)
"""
function oversample(t, x::Array{T,N}; factor::Integer = 4, dim = 1) where T <: Complex where N
    if factor == 1
        return t, x
    end
    xf = FFTW.fftshift(FFTW.fft(x, dim), dim)

    len = size(xf, dim)
    newlen = factor * length(t)
    δt = t[2] - t[1]
    δto = δt / factor
    Nto = collect(0:newlen - 1)
    to = t[1] .+ Nto .* δto

    sidx  = (newlen - len)//2 + 1
    iseven(newlen) || (sidx -= 1//2)
    iseven(len) || (sidx += 1//2)
    startidx = Int(sidx)
    endidx = startidx+len-1

    shape = collect(size(xf))
    shape[dim] = newlen
    xfo = zeros(eltype(xf), Tuple(shape))
    idxlo = CartesianIndices(size(xfo)[1:dim - 1])
    idxhi = CartesianIndices(size(xfo)[dim + 1:end])
    xfo[idxlo, startidx:endidx, idxhi] .= factor .* xf
    return to, FFTW.ifft(FFTW.ifftshift(xfo, dim), dim)
end


"""
Find limit of a series by Aitken acceleration
"""
function aitken_accelerate(f, x0; n0 = 0, rtol = 1e-6, maxiter = 10000)
    n = n0
    x0 = f(x0, n)
    x1 = f(x0, n + 1)
    x2 = f(x1, n + 2)
    Ax = aitken(x0, x1, x2)
    success = false
    while ~success && n < maxiter
        n += 1
        Axprev = Ax
        x0 = x1
        x1 = x2
        x2 = f(x2, n + 2)
        Ax = aitken(x0, x1, x2)

        if 2 * abs(Ax - Axprev) / abs(Ax + Axprev) < rtol
            success = true
        end
    end
    return Ax, success, n
end

function aitken(x0, x1, x2)
    den = (x0 - x1) - (x1 - x2)
    return x0 - (x1 - x0)^2 / den
end

"""
Find limit of series by brute force
"""
function converge_series(f, x0; n0 = 0, rtol = 1e-6, maxiter = 10000)
    n = n0
    x1 = x0
    success = false
    while ~success && n < maxiter
        x1 = f(x0, n)

        if 2 * abs(x1 - x0) / abs(x1 + x0) < rtol
            success = true
        end

        n += 1
        x0 = x1
    end
    return x1, success, n
end

"""
    check_spline_args(x, y)

Ensure that the x array contains unique and sorted values (while preserving the
relaionship between the x and y values).
"""
function check_spline_args(x, y)
    if any(diff(x) .== 0)
        error("entries in x must be unique")
    end
    if !issorted(x)
        idcs = sortperm(x)
        x = x[idcs]
        y = y[idcs]
    end
    x, y
end

"""
     make_spline_ifun(x, ifun)

If `ifun != nothing` then `ifun(x0)` should return the index of the first element in x
which is bigger than x0. Otherwise, it defaults two one of two options:
1. If `x` is uniformly spaced, the index is calculated based on the spacing of `x`
2. If `x` is not uniformly spaced, a `FastFinder` is used instead.
"""
function make_spline_ifun(x, ifun)
    if ifun === nothing
        δx = x[2] - x[1]
        if all(diff(x) .≈ δx)
            # x is uniformly spaced - use fast lookup
            xmax = maximum(x)
            xmin = minimum(x)
            N = length(x)
            ffast(x0) = x0 <= xmin ? 2 :
                        x0 >= xmax ? N : 
                        ceil(Int, (x0-xmin)/(xmax-xmin)*(N-1))+1
            ifun = ffast
        else
            # x is not uniformly spaced - use brute-force lookup
            ifun = FastFinder(x)
        end
    end
    ifun
end

"""
    CSpline

Simple cubic spline, see e.g.:
http://mathworld.wolfram.com/CubicSpline.html Boundary        
conditions extrapolate with initially constant gradient
"""
struct CSpline{Tx,Ty,Vx<:AbstractVector{Tx},Vy<:AbstractVector{Ty}, fT}
    x::Vx
    y::Vy
    D::Vy
    ifun::fT
end

# make  broadcast like a scalar
Broadcast.broadcastable(c::CSpline) = Ref(c)

"""
    CSpline(x, y [, ifun])

Construct a `CSpline` to interpolate the values `y` on axis `x`.

If given, `ifun(x0)` should return the index of the first element in x which is bigger
than x0. Otherwise, it defaults two one of two options:
1. If `x` is uniformly spaced, the index is calculated based on the spacing of `x`
2. If `x` is not uniformly spaced, a `FastFinder` is used instead.
"""
function CSpline(x, y, ifun=nothing)
    x, y = check_spline_args(x, y)
    R = similar(y)
    R[1] = y[2] - y[1]
    for i in 2:(length(y)-1)
        R[i] = y[i+1] - y[i-1]
    end
    R[end] = y[end] - y[end - 1]
    @. R *= 3
    d = fill(4.0, size(y))
    d[1] = 2.0
    d[end] = 2.0
    dl = fill(1.0, length(y) - 1)
    M = Tridiagonal(dl, d, dl)
    D = M \ R
    CSpline(x, y, D, make_spline_ifun(x, ifun))
end

"""
    (c::CSpline)(x0)

Evaluate the `CSpline` at coordinate `x0`
"""
function (c::CSpline)(x0)
    i = c.ifun(x0)
    x0 == c.x[i] && return c.y[i]
    x0 == c.x[i-1] && return c.y[i-1]
    t = (x0 - c.x[i - 1])/(c.x[i] - c.x[i - 1])
    (c.y[i - 1] 
        + c.D[i - 1]*t 
        + (3*(c.y[i] - c.y[i - 1]) - 2*c.D[i - 1] - c.D[i])*t^2 
        + (2*(c.y[i - 1] - c.y[i]) + c.D[i - 1] + c.D[i])*t^3)
end

"Calculate frequency vector k from samples x for FFT"
function fftfreq(x)
    Dx = abs(x[2] - x[1])
    all(diff(x) .≈ Dx) || error("x must be spaced uniformly")
    N = length(x)
    n = collect(range(0, length=N))
    (n .- N/2) .* 2π/(N*Dx)
end

"Calculate frequency vector k from samples x for rFFT"
function rfftfreq(x)
    Dx = abs(x[2] - x[1])
    all(diff(x) .≈ Dx) || error("x must be spaced uniformly")
    Nx = length(x)
    Nf = length(x)÷2 + 1
    n = collect(range(0, length=Nf))
    n .* 2π/(Nx*Dx)
end

"""
    FastFinder

Callable type which accelerates index finding for the case where inputs are usually in order.
"""
mutable struct FastFinder{xT, xeT}
    x::xT
    mi::xeT
    ma::xeT
    N::Int
    ilast::Int
    xlast::xeT
end

"""
    FastFinder(x)

Construct a `FastFinder` to find indices in the array `x`.

!!! warning
    `x` must be sorted in ascending order for `FastFinder` to work.
"""
function FastFinder(x::AbstractArray)
    issorted(x) || error("Input array for FastFinder must be sorted in ascending order.")
    if any(diff(x) .== 0)
        error("Entries in input array for FastFinder must be unique.")
    end
    FastFinder(x, x[1], x[end], length(x), 0, typemin(x[1]))
end

"""
    (f::FastFinder)(x0)

Find the first index in `f.x` which is larger than `x0`.

This is similar to [`findfirst`](@ref), but it starts at the index which was last used.
If the new value `x0` is close to the previous `x0`, this is much faster than `findfirst`.
"""
function (f::FastFinder)(x0::Number)
    # Default cases if we're out of bounds
    if x0 <= f.x[1]
        f.xlast = x0
        f.ilast = 1
        return 2
    elseif x0 >= f.x[end]
        f.xlast = x0
        f.ilast = f.N
        return f.N
    end
    if f.ilast == 0 # first call -- f.xlast is not set properly so comparisons won't work
        # return using brute-force method instead
        f.ilast = findfirst(x -> x>x0, f.x)
        return f.ilast
    end
    if x0 == f.xlast # same value as before - no work to be done
        return f.ilast
    elseif x0 < f.xlast # smaller than previous value - go through array backwards
        f.xlast = x0
        for i = f.ilast:-1:1
            if f.x[i] < x0
                f.ilast = i+1 # found last idx where x < x0 -> at i+1, x > x0
                return i+1
            end
        end
        # we only get to this point if we haven't found x0 - return the lower bound (2)
        f.ilast = 1
        return 2
    else # larger than previous value - just pick up where we left off
        f.xlast = x0
        for i = f.ilast:f.N
            if f.x[i] > x0
                f.ilast = i
                return i
            end
        end
        # we only get to this point if we haven't found x0 - return the upper bound (N)
        f.ilast = f.N
        return f.N
    end
end

struct RealBSpline{sT,hT,fT}
    rspl::sT
    h::hT
    hh::hT
    ifun::fT
end

struct CmplxBSpline{sT}
    rspl::sT
    ispl::sT
end

Broadcast.broadcastable(rs::RealBSpline) = Ref(rs)
Broadcast.broadcastable(cs::CmplxBSpline) = Ref(cs)

"""
    BSpline(x, y)

Construct a `RealBSpline` or `CmplxSpline` of given `order` (1 to 5, default=3)
to interpolate the values `y` on axis `x`.

If given, `ifun(x0)` should return the index of the first element in x which is bigger
than x0. Otherwise, it defaults two one of two options:
1. If `x` is uniformly spaced, the index is calculated based on the spacing of `x`
2. If `x` is not uniformly spaced, a `FastFinder` is used instead.

# Note
For accurate derivatives use 5th order splines
"""
function BSpline(x::AbstractVector{Tx}, y::AbstractVector{T}; ifun = nothing, order=3) where {Tx <: Real, T <: Real}
    check_spline_args(x, y)
    rspl = Dierckx.Spline1D(x, y, bc="extrapolate", k=order, s=0.0)
    h = zeros(eltype(y), order + 1)
    hh = similar(h)
    RealBSpline(rspl, h, hh, make_spline_ifun(Dierckx.get_knots(rspl), ifun))
end

function BSpline(x::AbstractVector{Tx}, y::AbstractVector{T}; kwargs...) where {Tx <: Real, T <: Complex}
    CmplxBSpline(BSpline(x, real(y); kwargs...),
                 BSpline(x, imag(y); kwargs...))
end

"""
    (cs::RealBSpline)(x)

Evaluate the `RealBSpline` at coordinate(s) `x`
"""
function (rs::RealBSpline)(x)
    splev!(rs.h, rs.hh, rs.rspl.t, rs.rspl.c, rs.rspl.k, x, rs.ifun)
end

"""
    (cs::CmplxSpline)(x)

Evaluate the `CmplxBSpline` at coordinate(s) `x`
"""
function (cs::CmplxBSpline)(x)
    complex(cs.rspl(x), cs.ispl(x))
end

"""
    derivative(rs::RealBSpline, x, order::Integer)

Calculate derivative of the spline `rs`. For `1 <= order < k` (where `k` is the spline order)
this uses an optimised routine.
For `order >= k` this falls back to the generic finite difference based method.
"""
function derivative(rs::RealBSpline, x, order::Integer)
    if order == 0
        return rs(x)
    elseif order < rs.rspl.k
        return Dierckx.derivative(rs.rspl, x, order)
    else
        invoke(derivative, Tuple{Any,Any,Integer}, rs, x, order)
    end
end

"""
    derivative(cs::CmplxBSpline, x, order::Integer)

Calculate derivative of the spline `cs`. For `1 <= order < k` (where `k` is the spline order)
this uses an optimised routine.
For `order >= k` this falls back to the generic finite difference based method.
"""
function derivative(cs::CmplxBSpline, x, order::Integer)
    complex(derivative(cs.rspl, x, order), derivative(cs.ispl, x, order))
end

"""
    differentiate_spline(rs::RealBSpline; order::Integer=1)

Return a new spline which is the derivative of `rs` and has the same support.
Higher orders are obtained by repeated spline differentiation.
"""
function differentiate_spline(rs::RealBSpline, order::Integer)
    x = Dierckx.get_knots(rs.rspl)
    dspl = rs
    for i = 1:order
        dy = derivative.(dspl, x, 1)
        dspl = BSpline(x, dy, order=rs.rspl.k)
    end
    dspl
end

"""
    roots(rs::RealBSpline)

Find the roots of the spline `rs`.
"""
function roots(rs::RealBSpline; maxn::Int=128)
    Dierckx.roots(rs.rspl; maxn=maxn)
end


"""
    splev!(h, hh, t, c, k, x, ifun)

Evaluate a spline s(x) of degree k, given in its b-spline representation.

# Arguments
- `h::ArrayPT<:Real,1}`: work space
- `hh::ArrayPT<:Real,1}`: work space
- `t::Array{T<:Real,1}`: the positions of the knots
- `c::Array{T<:Real,1}`: the b-spline coefficients
- `k::Integer`: the degree of s(x)
- `x::Real`: the point to evaluate at
- `ifun::Function`: a function to find the index `i` s.t. `t[i] > x`

Adapted from splev.f from Dierckx
http://www.netlib.org/dierckx/index.html
"""
function splev!(h, hh, t, c, k, x, ifun)
    k1 = k + 1
    k2 = k1 + 1
    nk1 = length(t) - k1
    tb = t[k1]
    te = t[nk1 + 1]
    l = k1
    l1 = l + 1
    # search for knot interval t(l) <= arg < t(l+1)
    l = ifun(x) - 1 + k
    # evaluate the non-zero b-splines at x.
    fpbspl!(h, hh, t, k, x, l)
    # find the value of s(x) at x.
    sp = 0.0
    ll = l - k1
    for j = 1:k1
        ll = ll + 1
        sp += c[ll]*h[j]
    end
    sp
end

"""
    fpbspl!(h, hh, t, k, x, l)

Evaluate the (k+1) non-zero b-splines of
degree k at t[l] <= x < t[l+1] using the stable recurrence
relation of de boor and cox.

# Arguments
- `h::ArrayPT<:Real,1}`: work space
- `hh::ArrayPT<:Real,1}`: work space
- `t::Array{T<:Real,1}`: the positions of the knots
- `k::Integer`: the degree of s(x)
- `x::Real`: the point to evaluate at
- `l::Integer`: the active knot location: `t[l] <= x < t[l+1]`

Adapted from fpbspl.f from Dierckx
http://www.netlib.org/dierckx/index.html
"""
function fpbspl!(h, hh, t, k, x, l)
    h[1] = one(x)
    for j = 1:k
        for i = 1:j
            hh[i] = h[i]
        end
        h[1] = 0.0
        for i = 1:j
            li = l+i
            lj = li-j
            if t[li] != t[lj]
                f = hh[i]/(t[li] - t[lj]) 
                h[i] = h[i] + f*(t[li] - x)
                h[i+1] = f*(x - t[lj])
            else
                h[i+1] = 0.0 
            end
        end
    end
    return h
end

end
