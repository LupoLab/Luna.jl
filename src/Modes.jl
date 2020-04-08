module Modes
import Roots: fzero
import Cubature: hcubature
import LinearAlgebra: dot, norm
import Luna: Maths
import Luna.PhysData: c, ε_0, μ_0
import Memoize: @memoize
import LinearAlgebra: mul!

export dimlimits, neff, β, α, losslength, transmission, dB_per_m, dispersion, zdw, field, Exy, Aeff, @delegated, @arbitrary, chkzkwarg

abstract type AbstractMode end

# make modes broadcast like a scalar
Broadcast.broadcastable(m::AbstractMode) = Ref(m)

"Maximum dimensional limits of validity for this mode"
function dimlimits(m::AbstractMode; z=0.0)
    error("abstract method called")
end

"Get the field components `(Ex, Ey)`` at position `xs`, `z`"
function field(m::AbstractMode, xs; z=0.0)
    error("abstract method called")
end

"Create function of coords that returns (xs) -> (Ex, Ey)"
field(m::AbstractMode; z=0) =  (xs) -> field(m, xs, z=z)

"Get mode normalization constant"
# we memoize this so it is only called once for each mode and z position
@memoize function N(m::AbstractMode; z=0.0)
    f = field(m, z=z)
    dl = dimlimits(m, z=z)
    function Nfunc(xs)
        E = f(xs)
        ret = sqrt(ε_0/μ_0)*dot(E, E)
        dl[1] == :polar ? xs[1]*ret : ret
    end
    val, err = hcubature(Nfunc, dl[2], dl[3])
    0.5*abs(val)
end

"Get the normalised field components at position `xs`, `z`"
Exy(m::AbstractMode, xs; z=0.0) = field(m, xs, z=z) ./ sqrt(N(m, z=z))

"Create function that returns normalised (xs) -> (Ex, Ey)"
Exy(m::AbstractMode; z=0.0) = (xs) -> Exy(m, xs, z=z)

"Get the field norm `|E|`` at position `xs`, `z`"
absE(m::AbstractMode, xs; z=0.0) = norm(Exy(m, xs, z=z))

"Create function that returns normalised (xs) -> |E|"
absE(m::AbstractMode; z=0.0) = (xs) -> absE(m, xs, z=z)

"Get effective area of mode"
# we memoize this so it is only called once for each mode and z position
@memoize function Aeff(m::AbstractMode; z=0.0)
    em = absE(m, z=z)
    dl = dimlimits(m, z=z)
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
function neff(m::AbstractMode, ω; z=0.0)
    error("abstract method called")
end

function β(m::AbstractMode, ω; z=0.0)
    return ω/c*real(neff(m, ω, z=z))
end

function α(m::AbstractMode, ω; z=0.0)
    return 2*ω/c*imag(neff(m, ω, z=z))
end

function losslength(m::AbstractMode, ω; z=0.0)
    return 1/α(m, ω, z=z)
end

function transmission(m::AbstractMode, ω, L; z=0.0)
    return exp(-α(m, ω)*L)
end

function dB_per_m(m::AbstractMode, ω; z=0.0)
    return 10/log(10).*α(m, ω)
end

function dispersion_func(m::AbstractMode, order; z=0.0)
    βn(ω) = Maths.derivative(ω -> β(m, ω, z=z), ω, order)
    return βn
end

function dispersion(m::AbstractMode, order, ω; z=0.0)
    return dispersion_func(m, order, z=z).(ω)
end

function zdw(m::AbstractMode; ub=200e-9, lb=3000e-9, z=0.0)
    ubω = 2π*c/ub
    lbω = 2π*c/lb
    ω0 = missing
    try
        ω0 = fzero(dispersion_func(m, 2, z=z), lbω, ubω)
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

struct ToSpace{mT,iT}
    ms::mT
    indices::iT
    nmodes::Int
    npol::Int
    Ems::Array{Float64,2}
end

"""
    ToSpace(ms; components=:xy)

Construct a `ToSpace` for high performance conversion between modal fields and real space.

# Arguments
- `ms::Tuple`: a tuple of modes
- `components::Symbol`: which polarisation components to return: :x, :y, :xy
"""
function ToSpace(ms; components=:xy)
    if components == :xy
        indices = 1:2
    elseif components == :x
        indices = 1
    elseif components == :y
        indices = 2
    else
        error("components $components not recognised")
    end
    nmodes = length(ms)
    npol = length(indices)
    Ems = Array{Float64,2}(undef, nmodes, npol)
    ToSpace(ms, indices, nmodes, npol, Ems)
end

"""
    to_space!(Erω, Emω, xs, ts::ToSpace; z=0.0)

Convert from modal fields to real space using provided `ToSpace` struct.

# Arguments
- `Erω::Array{ComplexF64}`: a dimension nω x npol array where the real space frequency domain
                            field will be written to
- `Emω::Array{ComplexF64}`: a dimension nω x nmodes array containing the frequency domain
                            modal fields
- `xs:Tuple`: the transverse coordinates, `x,y` for cartesian, `r,θ` for polar
- `ts::ToSpace`: the corresponding `ToSpace` struct
- `z::Real`: the axial position
"""
function to_space!(Erω, Emω, xs, ts::ToSpace; z=0.0)
    if ts.nmodes != size(Emω,2)
        error("the number of modes must match the number of modal fields")
    end
    if ts.npol != size(Erω,2)
        error("the number of output fields must match the number of polarisation components")
    end
    # we assume all dimlimits are the same
    dimlims = dimlimits(ts.ms[1], z=z)
    # handle limits
    if dimlims[1] == :cartesian
        # for the cartesian case
        # if either coordinate is outside dimlimits we return 0
        if xs[1] <= dimlimits[2][1] || xs[1] >= dimlimits[3][1]
            fill!(Erω, 0.0)
            return
        elseif xs[2] <= dimlimits[2][2] || xs[2] >= dimlimits[3][2]
            fill!(Erω, 0.0)
            return
        end
    elseif dimlimits[1] == :polar
        # for the polar case
        # if the r coordinate is negative we error
        if xs[1] < 0.0
            error("polar coordinate r cannot be smaller than 0")
        # if r is greater or equal to the boundary we return 0
        elseif xs[1] >= dimlimits[3][1]
            fill!(Erω, 0.0)
            return
        end
    end
    # get the field at x1, x2
    for i = 1:ts.nmodes
        ts.Ems[i,:] .= Exy(ts.ms[i], xs, z=z)[ts.indices] # field matrix (nmodes x npol)
    end
    mul!(Erω, Emω, ts.Ems) # matrix product (nω x nmodes) * (nmodes x npol) -> (nω x npol)
end

"""
    to_space(Emω, xs, ts::ToSpace; z=0.0)

Convert from modal fields to real space using provided `ToSpace` struct.

# Arguments
- `Emω::Array{ComplexF64}`: a dimension nω x nmodes array containing the frequency domain
                            modal fields
- `xs:Tuple`: the transverse coordinates, `x,y` for cartesian, `r,θ` for polar
- `ts::ToSpace`: the corresponding `ToSpace` struct
- `z::Real`: the axial position
"""
function to_space(Emω, xs, ts::ToSpace; z=0.0)
    Erω = Array{ComplexF64,2}(undef, size(Emω,1), ts.npol)
    to_space!(Erω, Emω, xs, ts, z=z)
    Erω
end

"""
    to_space(Emω, xs, ms; components=:xy, z=0.0)

Convert from modal fields to real space.

# Arguments
- `Emω::Array{ComplexF64}`: a dimension nω x nmodes array containing the frequency domain
                            modal fields
- `xs:Tuple`: the transverse coordinates, `x,y` for cartesian, `r,θ` for polar
- `ms::Tuple`: a tuple of modes
- `components::Symbol`: which polarisation components to return: :x, :y, :xy
- `z::Real`: the axial position
"""
function to_space(Emω, xs, ms; components=:xy, z=0.0)
    ts = ToSpace(ms, components=components)
    Erω = Array{ComplexF64,2}(undef, size(Emω,1), ts.npol)
    to_space!(Erω, Emω, xs, ts, z=z)
    Erω
end

end