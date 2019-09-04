module Modes
import Roots: fzero
import Cubature: hcubature
import LinearAlgebra: dot, norm
import Luna: Maths
import Luna.PhysData: c, ε_0, μ_0

export dimlimits, neff, β, α, losslength, transmission, dB_per_m, dispersion, zdw, field, Exy, Aeff, @delegated, @arbitrary

abstract type AbstractMode end

# make modes broadcast like a scalar
Broadcast.broadcastable(m::AbstractMode) = Ref(m)

"Maximum dimensional limits of validity for this mode"
function dimlimits(m::M) where {M <: AbstractMode}
    error("abstract method called")
end

"full complex refractive index of a mode"
function neff(m::M, ω) where {M <: AbstractMode}
    error("abstract method called")
end

function β(m::M, ω) where {M <: AbstractMode}
    return ω/c*real(neff(m, ω))
end

function β(m::M; λ) where {M <: AbstractMode}
    return β(m, 2π*c./λ)
end

function α(m::M, ω) where {M <: AbstractMode}
    return 2*ω/c*imag(neff(m, ω))
end

function α(m::M; λ) where {M <: AbstractMode}
    return α(m, 2π*c./λ)
end

function losslength(m::M, ω) where {M <: AbstractMode}
    return 1 ./ α(m, ω)
end

function losslength(m::M; λ) where {M <: AbstractMode}
    return losslength(m::M, 2π*c./λ) 
end

function transmission(m::M, L; λ) where {M <: AbstractMode}
    return @. exp(-α(m, λ=λ)*L)
end

function dB_per_m(m::M, ω) where {M <: AbstractMode}
    return 10/log(10).*α(m, ω)
end

function dB_per_m(m::M; λ) where {M <: AbstractMode}
    return return 10/log(10) .* α(m, λ=λ)
end

function dispersion_func(m::M, order) where {M <: AbstractMode}
    βn(ω) = Maths.derivative(ω -> β(m, ω), ω, order)
    return βn
end

function dispersion(m::M, order, ω) where {M <: AbstractMode}
    return dispersion_func(m, order).(ω)
end

function dispersion(m::M, order; λ) where {M <: AbstractMode}
    return dispersion(m, order, 2π*c./λ)
end

function zdw(m::M; ub=200e-9, lb=3000e-9) where {M <: AbstractMode}
    ubω = 2π*c/ub
    lbω = 2π*c/lb
    ω0 = fzero(dispersion_func(m, 2), lbω, ubω)
    return 2π*c/ω0
end

"Create function of coords that returns (xs) -> (Ex, Ey)"
function field(m::M) where {M <: AbstractMode}
    error("abstract method called")
end

"Get mode normalization constant"
function N(m::M) where {M <: AbstractMode}
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
function absE(m::M) where {M <: AbstractMode}
    func = let sN = sqrt(N(m)), f = field(m)
        function func(xs)
            norm(f(xs) ./ sN)
        end
    end
end

"Create function that returns normalised (xs) -> (Ex, Ey)"
function Exy(m) where {M <: AbstractMode}
    func = let sN = sqrt(N(m)), f = field(m)
        function func(xs)
            f(xs) ./ sN
        end
    end
end

"Get effective area of mode"
function Aeff(m) where {M <: AbstractMode}
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

"""
Macro to create a delegated mode, which takes its methods from an existing mode except
for those which are overwritten
"""
macro delegated(mex, kwargs...)
    Tname = Symbol(:DelegatedMode, gensym())
    @eval struct $Tname{mT}<:AbstractMode
        m::mT # wrapped mode
    end
    funs = [kw.args[1] for kw in kwargs]
    for mfun in (:α, :β, :field, :dimlimits)
        if mfun in funs
            dfun = kwargs[findfirst(mfun.==funs)].args[2]
            @eval ($mfun)(dm::$Tname, args...) = $dfun(args...)
        else
            @eval ($mfun)(dm::$Tname, args...) = ($mfun)(dm.m, args...)
        end
    end
    # get value of the mode expression that was passed in - needs to be done in caller scope
    mode = __module__.eval(mex)
    quote
        $Tname($mode)
    end
end

"""
Macro to create a "fully delegated" or arbitrary mode from the four required functions,
α, β, field and dimlimits.
"""
macro arbitrary(kwargs...)
    Tname = Symbol(:ArbitraryMode, gensym())
    @eval struct $Tname<:AbstractMode end
    funs = [kw.args[1] for kw in kwargs]
    for mfun in (:α, :β, :field, :dimlimits)
        if mfun in funs
            dfun = kwargs[findfirst(mfun.==funs)].args[2]
            @eval ($mfun)(dm::$Tname, args...) = $dfun(args...)
        else
            error("Must define $mfun for arbitrary mode!")
        end
    end
    quote
        $Tname()
    end
end
end