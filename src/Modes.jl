module Modes
import Roots: fzero
import Cubature: hcubature
import LinearAlgebra: dot, norm
import NumericalIntegration: integrate, Trapezoidal
import Luna: Maths
import Luna.PhysData: c, ε_0, μ_0

export dimlimits, neff, β, α, losslength, transmission, dB_per_m, dispersion, zdw, field, Exy, Aeff, @delegated, @arbitrary, chkzkwarg

abstract type AbstractMode end

# make modes broadcast like a scalar
Broadcast.broadcastable(m::AbstractMode) = Ref(m)

"Maximum dimensional limits of validity for this mode"
function dimlimits(m::AbstractMode)
    error("abstract method called")
end

"Create function of coords that returns (xs) -> (Ex, Ey)"
function field(m::AbstractMode)
    error("abstract method called")
end

"Get mode normalization constant"
function N(m::AbstractMode)
    f = field(m)
    dl = dimlimits(m)
    function Nfunc(xs)
        E = f(xs)
        ret = sqrt(ε_0/μ_0)*dot(E, E)
        dl[1] == :polar ? xs[1]*ret : ret
    end
    val, err = hcubature(Nfunc, dl[2], dl[3])
    0.5*abs(val)
end

"Create function that returns normalised (xs) -> |E|"
function absE(m::AbstractMode)
    func = let sN = sqrt(N(m)), f = field(m)
        function func(xs)
            norm(f(xs) ./ sN)
        end
    end
end

"Create function that returns normalised (xs) -> (Ex, Ey)"
function Exy(m)
    func = let sN = sqrt(N(m)), f = field(m)
        function func(xs)
            f(xs) ./ sN
        end
    end
end

"Get effective area of mode"
function Aeff(m::AbstractMode)
    em = absE(m)
    dl = dimlimits(m)
    # Numerator
    function Aeff_num(xs)
        e = em(xs)
        dl[1] == :polar ? xs[1]*e^2 : e^2
    end
    val, err = hcubature(Aeff_num, dl[2], dl[3])
    num = val^2
    # Denominator
    function Aeff_den(xs)
        e = em(xs)
        dl[1] == :polar ? xs[1]*e^4 : e^4
    end
    den, err = hcubature(Aeff_den, dl[2], dl[3])
    return num / den
end

"full complex refractive index of a mode"
function neff(m::AbstractMode, ω; z=0)
    error("abstract method called")
end

function β(m::AbstractMode, ω; z=0)
    return ω/c*real(neff(m, ω, z=z))
end

function α(m::AbstractMode, ω; z=0)
    return 2*ω/c*imag(neff(m, ω, z=z))
end

function losslength(m::AbstractMode, ω; z=0)
    return 1/α(m, ω, z=z)
end

function transmission(m::AbstractMode, ω, L; z=0)
    return exp(-α(m, ω)*L)
end

function dB_per_m(m::AbstractMode, ω; z=0)
    return 10/log(10).*α(m, ω)
end

function dispersion_func(m::AbstractMode, order; z=0)
    βn(ω) = Maths.derivative(ω -> β(m, ω, z=z), ω, order)
    return βn
end

function dispersion(m::AbstractMode, order, ω; z=0)
    return dispersion_func(m, order, z=z).(ω)
end

function zdw(m::AbstractMode; ub=200e-9, lb=3000e-9)
    ubω = 2π*c/ub
    lbω = 2π*c/lb
    ω0 = missing
    try
        ω0 = fzero(dispersion_func(m, 2), lbω, ubω)
    catch
    end
    return 2π*c/ω0
end

"Check that function accepts z keyword argument and add it if necessary"
function chkzkwarg(func)
    try
        func(2.5e15, z=0.0)
        return func
    catch e
        if isa(e, ErrorException)
            f = (ω; z) -> func(ω)
            return f
        else
            throw(e)
        end
    end
end

"""
Macro to create a delegated mode, which takes its methods from an existing mode except
for those which are overwritten
    Arguments:
        mex: Expression which evaluates to a valid Mode(<:AbstractMode)
        exprs: keyword-argument-like tuple of expressions α=..., β=... etc
    (Note: technically, exprs is a tuple of assignment expressions, which we are turning
    into key-value pairs by only considering the two arguments to the = operator)
"""
macro delegated(mex, exprs...)
    Tname = Symbol(:DelegatedMode, gensym())
    @eval struct $Tname{mT}<:AbstractMode
        m::mT # wrapped mode
    end
    funs = [kw.args[1] for kw in exprs]
    for mfun in (:α, :β, :field, :dimlimits)
        if mfun in funs
            dfun = exprs[findfirst(mfun.==funs)].args[2]
            @eval ($mfun)(dm::$Tname, args...; kwargs...) = $dfun(args...; kwargs...)
        else
            @eval ($mfun)(dm::$Tname, args...; kwargs...) = ($mfun)(dm.m, args...; kwargs...)
        end
    end
    quote
        $Tname($(esc(mex))) # create mode from expression (evaluted in the caller by esc)
    end
end

"""
Macro to create a "fully delegated" or arbitrary mode from the four required functions,
α, β, field and dimlimits.
    Arguments:
        exprs: keyword-argument-like tuple of expressions α=..., β=... etc
"""
macro arbitrary(exprs...)
    Tname = Symbol(:ArbitraryMode, gensym())
    @eval struct $Tname<:AbstractMode end
    funs = [kw.args[1] for kw in exprs]
    for mfun in (:α, :β, :field, :dimlimits)
        if mfun in funs
            dfun = exprs[findfirst(mfun.==funs)].args[2]
            @eval ($mfun)(dm::$Tname, args...; kwargs...) = $dfun(args...; kwargs...)
        else
            error("Must define $mfun for arbitrary mode!")
        end
    end
    quote
        $Tname()
    end
end

"""
    overlap(m::AbstractMode, r, E; dim)

Calculate mode overlap between radially symmetric field and radially symmetric mode.

# Examples
```jldoctest
julia> a = 100e-6;
julia> m = Capillary.MarcatilliMode(a, :He, 1.0);
julia> unm = besselj_zero(0, 1);
julia> r = collect(range(0, a, length=512));
julia> Er = besselj.(0, unm*r/a);

julia> η = Modes.overlap(m, r, Er; dim=1);
julia> η[1] ≈ 1
true
```
"""
function overlap(m::AbstractMode, r, E; dim)
    f = field(m) # f((r, θ)) returns field [Ex(r, θ), Ey(r, θ)] of the mode
    dl = dimlimits(m) # integration limits
    # sample the modal field at the same coords as E - select y polarisation component 
    Er = [f((ri, 0))[2] for ri in r] 
    Er[r .> dl[3][1]] .= 0 
    normEr = sqrt(2π*integrate(r, r.*abs2.(Er), Trapezoidal())) # normalisation factor

    # Generate output array: same shape as input, except length in space is 1
    shape = collect(size(E))
    shape[dim] = 1
    integral = ones(Tuple(shape)) # make output array

    # Indices to iterate over all other dimensions (e.g. polarisation, frequency)
    idxlo = CartesianIndices(size(E)[1:dim-1])
    idxhi = CartesianIndices(size(E)[dim+1:end])
    for hi in idxhi
        for lo in idxlo
                # normalisation factor for the other field
                normE = sqrt(2π*integrate(r, r.*abs2.(E[lo, :, hi]), Trapezoidal()))
                # E[lo, :, hi] is a vector
                integrand = 2π .* E[lo, :, hi] .* Er.*r./(normE*normEr)
                integral[lo, 1, hi] = abs2.(integrate(r, integrand, Trapezoidal()))
        end
    end
    return integral
end

end