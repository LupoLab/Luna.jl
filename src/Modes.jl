module Modes
import Roots: find_zero, Order2
import Cubature: hcubature
import LinearAlgebra: dot, norm
import NumericalIntegration: integrate, Trapezoidal
import Luna: Maths, Grid
import Luna.PhysData: c, ε_0, μ_0
import Memoize: @memoize
import LinearAlgebra: mul!
import DSP: unwrap

export dimlimits, neff, β, α, losslength, transmission, dB_per_m, dispersion, zdw, field, Exy, Aeff, @delegated, @arbitrary, chkzkwarg

abstract type AbstractMode end

ModeCollection = Union{Tuple{Vararg{T} where T <: Modes.AbstractMode},
                       AbstractArray{T} where T <: Modes.AbstractMode}

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

"""
    transmission(m::AbstractMode, ω, L; z=0.0)

Calculate the power transmission after propagation through length `L` in the mode `m` for
radiation at the frequency `ω`.
"""
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

"""
    zdw(m::AbstractMode; ub=200e-9, lb=3000e-9, z=0.0)

Calculate the zero-dispersion wavelength (ZDW) of mode `m` within the range `ub` to `lb`.
"""
function zdw(m::AbstractMode; ub=200e-9, lb=3000e-9, z=0.0)
    ubω = 2π*c/ub
    lbω = 2π*c/lb
    ω0 = missing
    try
        ω0 = find_zero(dispersion_func(m, 2, z=z), (lbω, ubω))
    catch
    end
    return 2π*c/ω0
end

"""
    zdw(m::AbstractMode, λ0; z=0.0, rtol=1e-4)

Calculate the zero-dispersion wavelength (ZDW) of mode `m` with an initial guess of `λ0`.

This method is faster than the bounded version if the ZDW is known to be close to `λ0`.
"""
function zdw(m::AbstractMode, λ0; z=0.0, rtol=1e-4)
    ωguess = 2π*c/λ0 * 1e-15 # convert everything find_zero sees to units close to 1.0
    dfun = dispersion_func(m, 2, z=z)
    β2(ω) = 1e30 * dfun(ω*1e15)
    ω0 = missing
    try
        ω0 = 1e15*find_zero(β2, ωguess, Order2(), rtol=rtol)
    catch
    end
    return 2π*c/ω0
end

function β_ret(m::AbstractMode, ω; z=0, λ0)
    ω0 = 2π*c/λ0
    β.(m, ω; z=z) .- dispersion(m, 1, ω0; z=z)*(ω.-ω0) .- β(m, ω0; z=z)
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
julia> abs2(η[1]) ≈ 1
true
```
"""
function overlap(m::AbstractMode, r, E; dim, norm=true)
    dl = dimlimits(m) # integration limits
    # sample the modal field at the same coords as E - select y polarisation component 
    Er = [Exy(m, (ri, 0))[2] for ri in r]  #field [Ex(r, θ), Ey(r, θ)] of the mode
    Er[r .> dl[3][1]] .= 0 
    #= normalisation factor - we want the integral of the modal intensity over the waveguide
        to be 1, but the  fields Exy(...) are normalised to produce modal power,
        so they include the factor of cε₀/2 =#
    normEr = 1/sqrt(c*ε_0/2) 

    # Generate output array: same shape as input, except length in space is 1
    shape = collect(size(E))
    shape[dim] = 1
    integral = zeros(eltype(E), Tuple(shape)) # make output array

    # Indices to iterate over all other dimensions (e.g. polarisation, frequency)
    idxlo = CartesianIndices(size(E)[1:dim-1])
    idxhi = CartesianIndices(size(E)[dim+1:end])
    for hi in idxhi
        for lo in idxlo
                # normalisation factor for the other field
                normE = norm ? sqrt(2π*integrate(r, r.*abs2.(E[lo, :, hi]), Trapezoidal())) :
                               1/sqrt(c*ε_0/2)
                # E[lo, :, hi] is a vector
                integrand = 2π .* E[lo, :, hi] .* Er.*r./(normE*normEr)
                integral[lo, 1, hi] = integrate(r, integrand, Trapezoidal())
        end
    end
    return integral
end

"""
    overlap(modes::ModeCollection, newgrid, oldgrid, r, Eωr)

Decompose the spatio-spectral field `Eωr`, sampled on radial coordinate `r` and time-grid
`oldgrid`, into the given `modes` and resample onto `newgrid` via cubic interpolation.
"""
function overlap(modes::ModeCollection, newgrid::Grid.RealGrid, oldgrid::Grid.RealGrid, r, Eωr)
    Egm = zeros(ComplexF64, (length(newgrid.ω), length(modes))) # output array
    # If old and new grids have different number of samples, we need to scale by δt1/δt2
    scale = (oldgrid.t[2] - oldgrid.t[1]) / (newgrid.t[2] - newgrid.t[1])
    #= Pulses centred on t=0 have large linear spectral phase components which can confuse
        phase unwrapping and lead to oscillations in the spectral phase after interpolation.
        Here we shift the pulse to t=-t_max of the old grid to remove this, then do the
        interpolation, then shift back to t=0 on the new grid. Note that this preserves any
        actual time shifts on the original pulse =#
    τold = length(oldgrid.t) * (oldgrid.t[2] - oldgrid.t[1])/2
    τnew = length(newgrid.t) * (newgrid.t[2] - newgrid.t[1])/2
    for (midx, mode) in enumerate(modes)
        Eωm = overlap(mode, r, Eωr; norm=false, dim=2)[:, 1]
        Eωm .*= exp.(1im.*oldgrid.ω.*τold) # shift to -t_max before unwrapping
        Aω = abs.(Eωm) # spectral amplitude
        ϕω = unwrap(angle.(Eωm)) # spectral phase
        Ag = Maths.BSpline(oldgrid.ω, Aω).(newgrid.ω)
        ϕg = Maths.BSpline(oldgrid.ω, ϕω).(newgrid.ω) .- newgrid.ω*τnew # shift to t=0
        Egm[:, midx] = scale * Ag .* exp.(1im*ϕg)
    end
    Egm
end


function _overlap(mode1::AbstractMode, mode2::AbstractMode)
    dl = dimlimits(mode1)
    if dl != dimlimits(mode2)
        error("Integration limits for both modes must be the same")
    end
    function f(xs)
        ret = 1/2*sqrt(ε_0/μ_0)*dot(conj(Exy(mode1, xs)), Exy(mode2, xs))
        dl[1] == :polar ? xs[1]*ret : ret
    end
    val, err = hcubature(f, dl[2], dl[3]; maxevals=1000)
end

"""
    overlap(mode1, mode2)

Calculate the normalised overlap between two `AbstractMode`s.
"""
function overlap(mode1::AbstractMode, mode2::AbstractMode)
    val, err = _overlap(mode1, mode2)
    val
end

"""
    orthogonal(mode1, mode2)

Test whether the `AbstractMode`s `mode1` and `mode2` are orthogonal to each other.
"""
function orthogonal(mode1::AbstractMode, mode2::AbstractMode)
    val, err = _overlap(mode1, mode2)
    isapprox(val, 0; atol=err)
end

"""
    orthonormal(modes)

Test whether the `modes` form an orthonormal set.
"""
function orthonormal(modes)
    out = true
    for (ii1, ii2) in Iterators.product(eachindex(modes), eachindex(modes))
        val, err = _overlap(modes[ii1], modes[ii2])
        this = ii1 == ii2 ? isapprox(val, 1; atol=err) : isapprox(val, 0; atol=err)
        out = out && this
    end
    out
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
@noinline function to_space!(Erω, Emω, xs, ts::ToSpace; z=0.0)
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
        if xs[1] <= dimlims[2][1] || xs[1] >= dimlims[3][1]
            fill!(Erω, 0.0)
            return
        elseif xs[2] <= dimlims[2][2] || xs[2] >= dimlims[3][2]
            fill!(Erω, 0.0)
            return
        end
    elseif dimlims[1] == :polar
        # for the polar case
        # if the r coordinate is negative we error
        if xs[1] < 0.0
            error("polar coordinate r cannot be smaller than 0")
        # if r is greater or equal to the boundary we return 0
        elseif xs[1] >= dimlims[3][1]
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

struct DelegatedMode{mT, idT} <: AbstractMode
    mode::mT # wrapper mode
    id::idT # unique identifier type to distinguish different DelegatedModes
end

function DelegatedMode(mode)
    DelegatedMode(mode, Val(gensym()))
end

"""
    delegated(mode, kwargs...)

Create a delegated mode, which takes its methods from an existing mode except
for those which are overwritten
# Arguments:
- `mode::AbstractMode`: The wrapped mode to which non-specified methods are delegated
- `kwargs`: functions that override: `neff`, `field`, `dimlimits`, `Aeff`, or `N`.

The functions given should have the **same signature** as the mode methods, i.e. take
an `AbstractMode` as their first argument, **even if** they do not do anything with it. This
is to ensure that the delegated functions can access the data of the wrapped mode if necessary.

To override `neff` with functions for `α` and `β`, create the `neff` function using
[`neff_from_αβ`](@ref).
"""
function delegated(mode; kwargs...)
    dmode = DelegatedMode(mode)
    mT = typeof(dmode) # this is unique for each instance by virtue of the id field
    kw = Dict(kwargs)
    for mfun in (:neff, :field, :dimlimits)
        if haskey(kw, mfun)
            dfun = kw[mfun]
            @eval $mfun(dm::$mT, args...; kwargs...) = $dfun(dm.mode, args...; kwargs...)
        else
            @eval $mfun(dm::$mT, args...; kwargs...) = $mfun(dm.mode, args...; kwargs...)
        end
    end
    for mfun in (:Aeff, :N)
        if haskey(kw, mfun)
            dfun = kw[mfun]
            @eval $mfun(dm::$mT, args...; kwargs...) = $dfun(dm.mode, args...; kwargs...)
        elseif (!haskey(kw, :dimlimits)) && (!haskey(kw, :field))
            #= if dimlimits and field have not been changed, Aeff and N are the same too,
                so we can safely delegate them to the wrapped mode (likely faster) =#
            @eval $mfun(dm::$mT, args...; kwargs...) = $mfun(dm.mode, args...; kwargs...)
        end
    end
    dmode
end

"""
    arbitrary(kwargs...)

Create an arbitrary mode, which takes its methods from given functions. 

The functions given should have the same signature as defined `Luna.Modes`, **except** that
the first argument (the `AbstractMode`) is omitted, e.g. for `neff` the function should be
of the form `n(ω; z) = ...`

To define `neff` with functions for `α` and `β`, create the `neff` function using
[`neff_from_αβ`](@ref).
"""
function arbitrary(;kwargs...)
    dmode = DelegatedMode(nothing)
    mT = typeof(dmode)
    kw = Dict(kwargs)
    for mfun in (:neff, :field, :dimlimits)
        if haskey(kw, mfun)
            dfun = kw[mfun]
            @eval $mfun(dm::$mT, args...; kwargs...) = $dfun(args...; kwargs...)
        else
            err = "method $mfun not defined for this mode"
            @eval $mfun(dm::$mT, args...; kwargs...) = error($err)
        end
    end
    for mfun in (:Aeff, :N)
        if haskey(kw, mfun)
            dfun = kw[mfun]
            @eval $mfun(dm::$mT, args...; kwargs...) = $dfun(args...; kwargs...)
        end
    end
    dmode
end

"""
    neff_from_αβ(α, β)

Create a closure converting the functions `α(ω; z)` and `β(ω; z)` into an effective index.
"""
neff_from_αβ(α, β) = (ω; z=0) -> c/ω * (β(ω; z=z) + 0.5im*α(ω; z=z))

end