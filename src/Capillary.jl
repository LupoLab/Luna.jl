module Capillary
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import StaticArrays: SVector
import Cubature: hquadrature
using Reexport
@reexport using Luna.Modes
import Luna: Maths, Grid
import Luna.PhysData: c, ε_0, μ_0, ref_index_fun, roomtemp, densityspline, sellmeier_gas
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N, modeinfo
import Luna.LinearOps: make_linop, conj_clamp, neff_grid, neff_β_grid
import Luna.PhysData: wlfreq, roomtemp
import Luna.Utils: subscript
import Base: show

export MarcatiliMode, dimlimits, neff, field, N, Aeff

"""
    MarcatiliMode

Type representing a mode of a hollow capillary as presented in:

Marcatili, E. & Schmeltzer, R.
"Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation)."
Bell System Technical Journal 43, 1783–1809 (1964).
"""
struct MarcatiliMode{Ta, Tcore, Tclad, LT} <: AbstractMode
    a::Ta # core radius callable as function of z only, or fixed core radius if a Number
    n::Int # azimuthal mode index
    m::Int # radial mode index
    kind::Symbol # kind of mode (transverse magnetic/electric or hybrid)
    unm::Float64 # mth zero of the nth Bessel function of the first kind
    ϕ::Float64 # overall rotation angle of the mode
    coren::Tcore # callable, returns (possibly complex) core ref index as function of ω
    cladn::Tclad # callable, returns (possibly complex) cladding ref index as function of ω
    model::Symbol # if :full, includes complete influence of complex cladding ref index
    loss::LT # Val{true}() or Val{false}() - whether to include the loss
    aeff_intg::Float64 # Pre-calculated integral fraction for effective area
end

function show(io::IO, m::MarcatiliMode)
    a = radius_string(m)
    loss = "loss=" * (m.loss == Val(true) ? "true" : "false")
    model = "model="*string(m.model)
    angle = "ϕ=$(m.ϕ/π)π"
    out = "MarcatiliMode{"*join([mode_string(m), a, loss, model, angle], ", ")*"}"
    print(io, out)
end

mode_string(m::MarcatiliMode) = string(m.kind)*subscript(m.n)*subscript(m.m)
radius_string(m::MarcatiliMode{<:Number, Tco, Tcl, LT}) where {Tco, Tcl, LT} = "a=$(m.a)"
radius_string(m::MarcatiliMode) = "a(z=0)=$(radius(m, 0))"

modeinfo(m::MarcatiliMode) = Dict(:kind => m.kind, :n => m.n, :m => m.m,
                                  :radius => radius(m, 0), :ϕ => m.ϕ, :model => m.model,
                                  :loss => m.loss == Val(true))

"""
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)

Create a MarcatiliMode.

# Arguments
- `a` : Either a `Number` for constant core radius, or a function `a(z)` for variable radius.
- `n::Int` : Azimuthal mode index (number of nodes in the field along azimuthal angle).
- `m::Int` : Radial mode index (number of nodes in the field along radial coordinate).
- `kind::Symbol` : `:TE` for transverse electric, `:TM` for transverse magnetic,
                   `:HE` for hybrid mode.
- `ϕ::Float` : Azimuthal offset angle (for linearly polarised modes, this is the angle
                between the mode polarisation and the `:y` axis)
- `coren` : Callable `coren(ω; z)` which returns the refractive index of the core
- `cladn` : Callable `cladn(ω; z)` which returns the refractive index of the cladding
- `model::Symbol=:full` : If `:full`, use the complete Marcatili model which takes into
                          account the dispersive influence of the cladding refractive index.
                          If `:reduced`, use the simplified model common in the literature
- `loss::Bool=true` : Whether to include loss.

"""
function MarcatiliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)
    # chkzkwarg makes sure that coren and cladn take z as a keyword argument
    aeff_intg = Aeff_Jintg(n, get_unm(n, m, kind), kind)
    MarcatiliMode(a, n, m, kind, get_unm(n, m, kind), ϕ,
                   chkzkwarg(coren), chkzkwarg(cladn),
                   model, Val(loss), aeff_intg)
end

"""
    MarcatiliMode(a, gas, P; kwargs...)

Create a MarcatiliMode for a capillary with radius `a` which is filled with `gas` to
pressure `P`.
"""
function MarcatiliMode(a, gas, P;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full,
                        clad=:SiO2, loss=true)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatiliMode(a, gas, P, cladn; kwargs...)

Create a MarcatiliMode for a capillary made of a cladding material defined by the refractive
index `cladn(ω; z)` with a core radius `a` which is filled with `gas` to pressure `P`.
"""
function MarcatiliMode(a, gas, P, cladn;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, loss=true)
    rfg = ref_index_fun(gas, P, T)
    coren = (ω; z) -> rfg(wlfreq(ω))
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatiliMode(a, coren; kwargs...)

Create a MarcatiliMode for a capillary with radius `a` with `z`-dependent gas fill determined
by `coren(ω; z)`.
"""
function MarcatiliMode(a, coren;
                        n=1, m=1, kind=:HE, ϕ=0.0, model=:full, clad=:SiO2, loss=true)
    rfs = ref_index_fun(clad)
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end


"""
    MarcatiliMode(a; kwargs...)

Create a `MarcatiliMode` for a capillary with radius `a` and no gas fill.
"""
MarcatiliMode(a; kwargs...) = MarcatiliMode(a, (ω; z) -> 1; kwargs...)

"""
    neff(m::MarcatiliMode, ω; z=0)

Calculate the complex effective index of Marcatili mode with dielectric core and arbitrary
(metal or dielectric) cladding.

Adapted from:

Marcatili, E. & Schmeltzer, R.
"Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation)."
Bell System Technical Journal 43, 1783–1809 (1964).
"""
function neff(m::MarcatiliMode, ω; z=0)
    εcl = m.cladn(ω, z=z)^2
    εco = m.coren(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    neff(m, ω, εco, vn, radius(m, z))
end

# Dispatch on loss to make neff type stable
# m.loss = Val{true}() (returns ComplexF64)
function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
    if m.model == :full
        k = ω/c
        n = sqrt(complex(εco - (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2))
        return (real(n) < 1e-3) ? (1e-3 + im*clamp(imag(n), 0, Inf)) : n
    elseif m.model == :reduced
        return ((1 + (εco - 1)/2 - c^2*m.unm^2/(2*ω^2*a^2))
                    + im*(c^3*m.unm^2)/(a^3*ω^3)*vn)
    else
        error("model must be :full or :reduced")
    end
end

# m.loss = Val{false}() (returns Float64)
function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
    if m.model == :full
        k = ω/c
        n = real(sqrt(εco - (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2))
        return (n < 1e-3) ? 1e-3 : n
    elseif m.model == :reduced
        return real(1 + (εco - 1)/2 - c^2*m.unm^2/(2*ω^2*a^2))
    else
        error("model must be :full or :reduced")
    end
end

function neff_wg(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, ω; z=0) where {Ta, Tcl, Tco}
    εcl = m.cladn(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    a = radius(m, z)
    if m.model == :full
        k = ω/c
        return (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2
    elseif m.model == :reduced
        return c^2*m.unm^2/(2*ω^2*a^2) + im*(c^3*m.unm^2)/(a^3*ω^3)*vn
    else
        error("model must be :full or :reduced")
    end
end

function neff_wg(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, ω; z=0) where {Ta, Tcl, Tco}
    εcl = m.cladn(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    a = radius(m, z)
    if m.model == :full
        k = ω/c
        return (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2
    elseif m.model == :reduced
        return c^2*m.unm^2/(2*ω^2*a^2)
    else
        error("model must be :full or :reduced")
    end
end

function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        return sqrt(complex(εco - nwg))
    elseif m.model == :reduced
        return complex((1 + (εco - 1)/2 - nwg))
    else
        error("model must be :full or :reduced")
    end
end

function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        return real(sqrt(complex(εco - nwg)))
    elseif m.model == :reduced
        return real((1 + (εco - 1)/2 - nwg))
    else
        error("model must be :full or :reduced")
    end
end

function get_vn(εcl, kind)
    if kind == :HE
        (εcl + 1)/(2*sqrt(complex(εcl - 1)))
    elseif kind == :TE
        1/sqrt(complex(εcl - 1))
    elseif kind == :TM
        εcl/sqrt(complex(εcl - 1))
    else
        error("kind must be :TE, :TM or :HE")
    end
end

function get_unm(n, m, kind)
    if (kind == :TE) || (kind == :TM)
        if (n != 0)
            error("n=0 for TE or TM modes")
        end
        besselj_zero(1, m)
    elseif kind == :HE
        if n == 0
            error("n ≠ 0 for HE modes")
        end
        besselj_zero(n-1, m)
    else
        error("kind must be :TE, :TM or :HE")
    end
end

radius(m::MarcatiliMode{<:Number, Tco, Tcl, LT}, z) where {Tcl, Tco, LT} = m.a
radius(m::MarcatiliMode, z) = m.a(z)

dimlimits(m::MarcatiliMode; z=0) = (:polar, (0.0, 0.0), (radius(m, z), 2π))

# we use polar coords, so xs = (r, θ)
function field(m::MarcatiliMode, xs; z=0)
    r, θ = xs
    if m.kind == :HE
        return (besselj(m.n-1, r*m.unm/radius(m, z)) .* SVector(
            cos(θ)*sin(m.n*(θ + m.ϕ)) - sin(θ)*cos(m.n*(θ + m.ϕ)),
            sin(θ)*sin(m.n*(θ + m.ϕ)) + cos(θ)*cos(m.n*(θ + m.ϕ))
            ))
    elseif m.kind == :TE
        return besselj(1, r*m.unm/radius(m, z)) .* SVector(-sin(θ), cos(θ))
    elseif m.kind == :TM
        return besselj(1, r*m.unm/radius(m, z)) .* SVector(cos(θ), sin(θ))
    end
end

function N(m::MarcatiliMode; z=0)
    np1 = (m.kind == :HE) ? m.n : 2
    π/2 * radius(m, z)^2 * besselj(np1, m.unm)^2 * sqrt(ε_0/μ_0)
end

function Aeff_Jintg(n, unm, kind)
    den, err = hquadrature(r -> r*besselj(n-1, unm*r)^4, 0, 1)
    np1 = (kind == :HE) ? n : 2
    num = 1/4 * besselj(np1, unm)^4
    return 2π*num/den
end

Aeff(m::MarcatiliMode; z=0) = radius(m, z)^2 * m.aeff_intg


"""
    gradient(gas, L, p0, p1; T=roomtemp)

Convenience function to create density and core index profiles for
simple two-point gradient fills defined by the waveguide length `L` and the pressures at
`z=0` and `z=L`.
"""
function gradient(gas, L, p0, p1; T=roomtemp)
    γ = sellmeier_gas(gas)
    dspl = densityspline(gas, Pmin=p0==p1 ? 0 : min(p0, p1), Pmax=max(p0, p1); T)
    p(z) =  z > L ? p1 :
            z <= 0 ? p0 :
            sqrt(p0^2 + z/L*(p1^2 - p0^2))
    dens(z) = dspl(p(z))
    coren(ω; z) = sqrt(1 + γ(wlfreq(ω)*1e6)*dens(z))
    return coren, dens
end

"""
    gradient(gas, Z, P; T=roomtemp)

Convenience function to create density and core index profiles for
multi-point gradient fills defined by positions `Z` and pressures `P`.
"""
function gradient(gas, Z, P; T=roomtemp)
    γ = sellmeier_gas(gas)
    ex = extrema(P)
    dspl = densityspline(gas, Pmin=ex[1]==ex[2] ? 0 : ex[1], Pmax=ex[2]; T)
    function p(z)
        if z <= Z[1]
            return P[1]
        elseif z >= Z[end]
            return P[end]
        else
            i = findlast(x -> x < z, Z)
            return sqrt(P[i]^2 + (z - Z[i])/(Z[i+1] - Z[i])*(P[i+1]^2 - P[i]^2))
        end
    end
    dens(z) = dspl(p(z))
    coren(ω; z) = sqrt(1 + γ(wlfreq(ω)*1e6)*dens(z))
    return coren, dens
end

#= Avoid repeated calculation of the waveguide part of the effective index for modes with
    constant core radius.
    This is used by LinearOps.make_linop =#
function neff_β_grid(grid,
                   mode::MarcatiliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT},
                   λ0)
    nwg = complex(zero(grid.ω))
    sidcs = (1:length(grid.ω))[grid.sidx]
    for iω in sidcs
        nwg[iω] = neff_wg(mode, grid.ω[iω]; z=0)
    end
    _neff = let nwg=nwg, ω=grid.ω, mode=mode
        _neff(iω; z) = neff(mode, mode.coren(ω[iω], z=z)^2, nwg[iω])
    end
    _β = let nwg=nwg, ω=grid.ω, _neff=_neff
        _β(iω; z) = ω[iω]/c*real(_neff(iω; z=z))
    end
    _neff, _β
end

# Collection of modes with fixed core radius
FixedCoreCollection = Union{
    Tuple{Vararg{MarcatiliMode{<:Number, Tco, Tcl, LT}} where {Tco, Tcl, LT}},
    AbstractArray{MarcatiliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT}}
    }

function neff_grid(grid, modes::FixedCoreCollection, λ0; ref_mode=1)
    nwg = Array{ComplexF64, 2}(undef, (length(grid.ω), length(modes)))
    sidcs = (1:length(grid.ω))[grid.sidx]
    for (i, mi) in enumerate(modes)
        for iω in sidcs
            nwg[iω, i] = neff_wg(mi, grid.ω[iω]; z=0)
        end
    end
    _neff = let nwg=nwg, ω=grid.ω, modes=modes
        _neff(iω, iim; z) = neff(modes[iim], modes[iim].coren(ω[iω], z=z)^2, nwg[iω, iim])
    end
    _neff
end

"""
    transmission(a, λ, L; kind=:HE, n=1, m=1)

Calculate the transmission through a capillary with core radius `a` and length `L` at the
wavelength `λ` when propagating the `MarcatiliMode` defined by `kind`, `n` and `m`.
"""
function transmission(a, λ, L; kind=:HE, n=1, m=1)
    # TODO hardcoded fill needs to be updated if using absorbing materials
    mode = MarcatiliMode(a, :He, 0; n=n, m=m, kind=kind)
    Modes.transmission(mode, wlfreq(λ), L)
end

end
