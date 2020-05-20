module Capillary
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import StaticArrays: SVector
import Cubature: hquadrature
using Reexport
@reexport using Luna.Modes
import Luna: Maths, Grid
import Luna.PhysData: c, ε_0, μ_0, ref_index_fun, roomtemp, densityspline, sellmeier_gas
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N
import Luna.LinearOps: make_linop, conj_clamp
import Luna.PhysData: wlfreq
import Luna.Utils: subscript
import Base: show

export MarcatilliMode, dimlimits, neff, field, N, Aeff

struct MarcatilliMode{Ta, Tcore, Tclad, LT} <: AbstractMode
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

function show(io::IO, m::MarcatilliMode)
    a = radius_string(m)
    loss = "loss=" * (m.loss == Val(true) ? "true" : "false")
    model = "model="*string(m.model)
    out = "MarcatilliMode{"*join([mode_string(m), a, loss, model], ", ")*"}"
    print(io, out)
end

mode_string(m::MarcatilliMode) = string(m.kind)*subscript(m.n)*subscript(m.m)
radius_string(m::MarcatilliMode{<:Number, Tco, Tcl, LT}) where {Tco, Tcl, LT} = "a=$(m.a)"
radius_string(m::MarcatilliMode) = "a(z=0)=$(radius(m, 0))"

"""
    MarcatilliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)

Create a MarcatilliMode.

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
- `model::Symbol=:full` : If `:full`, use the complete Marcatilli model which takes into
                          account the dispersive influence of the cladding refractive index.
                          If `:reduced`, use the simplified model common in the literature
- `loss::Bool=true` : Whether to include loss.

"""
function MarcatilliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)
    # chkzkwarg makes sure that coren and cladn take z as a keyword argument
    aeff_intg = Aeff_Jintg(n, get_unm(n, m, kind), kind)
    MarcatilliMode(a, n, m, kind, get_unm(n, m, kind), ϕ,
                   chkzkwarg(coren), chkzkwarg(cladn),
                   model, Val(loss), aeff_intg)
end

"""
    MarcatilliMode(a, gas, P; kwargs...)

Create a MarcatilliMode for a capillary with radius `a` which is filled with `gas` to
pressure `P`.
"""
function MarcatilliMode(a, gas, P;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full,
                        clad=:SiO2, loss=true)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatilliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatilliMode(a, gas, P cladn; kwargs...)

Create a MarcatilliMode for a capillary made of a cladding material defined by the refractive
index `cladn(ω; z)` with a core radius `a` which is filled with `gas` to pressure `P`.
"""
function MarcatilliMode(a, gas, P, cladn;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, loss=true)
    rfg = ref_index_fun(gas, P, T)
    coren = (ω; z) -> rfg(wlfreq(ω))
    MarcatilliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatilliMode(a, coren; kwargs...)

Create a MarcatilliMode for a capillary with radius `a` with `z`-dependent gas fill determined
by `coren(ω; z)`.
"""
function MarcatilliMode(a, coren;
                        n=1, m=1, kind=:HE, ϕ=0.0, model=:full, clad=:SiO2, loss=true)
    rfs = ref_index_fun(clad)
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatilliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    neff(m::MarcatilliMode, ω; z=0)
    
Calculate the complex effective index of Marcatilli mode with dielectric core and arbitrary
(metal or dielectric) cladding.

Adapted from:

Marcatili, E. & Schmeltzer, R.
"Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation)."
Bell System Technical Journal 43, 1783–1809 (1964).
"""
function neff(m::MarcatilliMode, ω; z=0)
    εcl = m.cladn(ω, z=z)^2
    εco = m.coren(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    neff(m, ω, εco, vn, radius(m, z))
end

# Dispatch on loss to make neff type stable
# m.loss = Val{true}() (returns ComplexF64)
function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{true}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
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
function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{false}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
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

function neff_wg(m::MarcatilliMode{Ta, Tco, Tcl, Val{true}}, ω; z=0) where {Ta, Tcl, Tco}
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

function neff_wg(m::MarcatilliMode{Ta, Tco, Tcl, Val{false}}, ω; z=0) where {Ta, Tcl, Tco}
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

function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{true}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        n = sqrt(complex(εco - nwg))
        return (real(n) < 1e-3) ? (1e-3 + im*clamp(imag(n), 0, Inf)) : n
    elseif m.model == :reduced
        return ((1 + (εco - 1)/2 - nwg))
    else
        error("model must be :full or :reduced")
    end 
end

function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{false}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        n = real(sqrt(complex(εco - nwg)))
        return (real(n) < 1e-3) ? (1e-3 + im*clamp(imag(n), 0, Inf)) : n
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
        if (n != 0) || (m != 1)
            error("n=0, m=1 for TE or TM modes")
        end
        besselj_zero(1, 1)
    elseif kind == :HE
        besselj_zero(n-1, m)
    else
        error("kind must be :TE, :TM or :HE")
    end
end

radius(m::MarcatilliMode{<:Number, Tco, Tcl, LT}, z) where {Tcl, Tco, LT} = m.a
radius(m::MarcatilliMode, z) = m.a(z)

dimlimits(m::MarcatilliMode; z=0) = (:polar, (0.0, 0.0), (radius(m, z), 2π))

# we use polar coords, so xs = (r, θ)
function field(m::MarcatilliMode, xs; z=0)
    if m.kind == :HE
        return (besselj(m.n-1, xs[1]*m.unm/radius(m, z)) .* SVector(
            cos(xs[2])*sin(m.n*(xs[2] + m.ϕ)) - sin(xs[2])*cos(m.n*(xs[2] + m.ϕ)),
            sin(xs[2])*sin(m.n*(xs[2] + m.ϕ)) + cos(xs[2])*cos(m.n*(xs[2] + m.ϕ))
            ))
    elseif m.kind == :TE
        return besselj(1, xs[1]*m.unm/radius(m, z)) .* SVector(-sin(xs[2]), cos(xs[2]))
    elseif m.kind == :TM
        return besselj(1, xs[1]*m.unm/radius(m, z)) .* SVector(cos(xs[2]), sin(xs[2]))
    end
end

function N(m::MarcatilliMode; z=0)
    np1 = (m.kind == :HE) ? m.n : 2
    π/2 * radius(m, z)^2 * besselj(np1, m.unm)^2 * sqrt(ε_0/μ_0)
end

function Aeff_Jintg(n, unm, kind)
    den, err = hquadrature(r -> r*besselj(n-1, unm*r)^4, 0, 1)
    np1 = (kind == :HE) ? n : 2
    num = 1/4 * besselj(np1, unm)^4
    return 2π*num/den
end

Aeff(m::MarcatilliMode; z=0) = radius(m, z)^2 * m.aeff_intg


"Convenience function to create density and core index profiles for
simple two-point gradient fills."
function gradient(gas, L, p0, p1)
    γ = sellmeier_gas(gas)
    dspl = densityspline(gas, Pmin=p0==p1 ? 0 : min(p0, p1), Pmax=max(p0, p1))
    p(z) =  z > L ? p1 :
            z <= 0 ? p0 : 
            sqrt(p0^2 + z/L*(p1^2 - p0^2))
    dens(z) = dspl(p(z))
    coren(ω; z) = sqrt(1 + (γ(wlfreq(ω)*1e6)*dens(z)))
    return coren, dens
end

"Convenience function to create density and core index profiles for
multi-point gradient fills."
function gradient(gas, Z, P)
    γ = sellmeier_gas(gas)
    ex = extrema(P)
    dspl = densityspline(gas, Pmin=ex[1]==ex[2] ? 0 : ex[1], Pmax=ex[2])
    function p(z)
        if z <= Z[1]
            return P[1]
        elseif z >= Z[end]
            return P[end]
        else
            i = findfirst(x -> x < z, Z)
            return sqrt(P[i]^2 + (z - Z[i])/(Z[i+1] - Z[i])*(P[i+1]^2 - P[i]^2))
        end
    end
    dens(z) = dspl(p(z))
    coren(ω; z) = sqrt(1 + (γ(wlfreq(ω)*1e6)*dens(z)))
    return coren, dens
end

function make_linop(grid::Grid.RealGrid,
                    mode::MarcatilliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT},
                    λ0)
    nwg = complex(zero(grid.ω))
    nwg[2:end] = neff_wg.(mode, grid.ω[2:end]; z=0)
    function linop!(out, z)
        β1 = Modes.dispersion(mode, 1, wlfreq(λ0), z=z)::Float64
        for iω = 2:length(grid.ω)
            εco = mode.coren(grid.ω[iω], z=z)^2
            nc = conj_clamp(neff(mode, εco, nwg[iω]), grid.ω[iω])
            out[iω] = -im*(grid.ω[iω]/c*nc - grid.ω[iω]*β1)
        end
        out[1] = 0
    end
    function βfun!(out, z)
        for iω = 2:length(grid.ω)
            εco = mode.coren(grid.ω[iω], z=z)^2
            n = neff(mode, εco, nwg[iω])
            out[iω] = grid.ω[iω]/c*real(n)
        end
        out[1] = 1.0
    end
    return linop!, βfun!
end

function make_linop(grid::Grid.EnvGrid,
                    mode::MarcatilliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT},
                    λ0; thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    nwg = complex(zero(grid.ω))
    nwg[grid.sidx] = neff_wg.(mode, grid.ω[grid.sidx]; z=0)
    function linop!(out, z)
        fill!(out, 0.0)
        β1 = Modes.dispersion(mode, 1, wlfreq(λ0), z=z)::Float64
        if !thg
            βref = Modes.β(mode, wlfreq(λ0), z=z)
        end
        for iω in sidcs
            εco = mode.coren(grid.ω[iω], z=z)^2
            nc = conj_clamp(neff(mode, εco, nwg[iω]), grid.ω[iω])
            out[iω] = -im*(grid.ω[iω]/c*nc - (grid.ω[iω] - grid.ω0)*β1)
            if !thg
                out[iω] -= -im*βref
            end
        end
    end
    function βfun!(out, z)
        fill!(out, 1.0)
        for iω in sidcs
            εco = mode.coren(grid.ω[iω], z=z)^2
            n = neff(mode, εco, nwg[iω])
            out[iω] = grid.ω[iω]/c*real(n)
        end
    end
    return linop!, βfun!
end

FixedCoreCollection = Union{
    Tuple{Vararg{MarcatilliMode{<:Number, Tco, Tcl, LT}} where {Tco, Tcl, LT}},
    AbstractArray{MarcatilliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT}}
    }

function make_linop(grid::Grid.RealGrid, modes::FixedCoreCollection, λ0; ref_mode=1)
    nwg = Array{ComplexF64, 2}(undef, (length(grid.ω), length(modes)))
    for (i, mi) in enumerate(modes)
        nwg[2:end, i] = neff_wg.(mi, grid.ω[2:end]; z=0)
    end
    εco = complex(zero(grid.ω))
    function linop!(out, z)
        β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0), z=z)::Float64
        fill!(out, 0.0)
        # NOTE here we assume that all modes have the same gas fill
        εco[2:end] .= modes[ref_mode].coren.(grid.ω[2:end], z=z).^2
        for i in eachindex(modes)
            for iω = 2:length(grid.ω)
                nc = conj_clamp(neff(modes[i], εco[iω], nwg[iω, i]), grid.ω[iω])
                out[iω, i] = -im*(grid.ω[iω]/c*nc - grid.ω[iω]*β1)
            end
        end
    end
end

function make_linop(grid::Grid.EnvGrid, modes::FixedCoreCollection, λ0;
                    ref_mode=1, thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    nwg = Array{ComplexF64, 2}(undef, (length(grid.ω), length(modes)))
    for (i, mi) in enumerate(modes)
        nwg[grid.sidx, i] = neff_wg.(mi, grid.ω[grid.sidx]; z=0)
    end
    εco = complex(zero(grid.ω))
    function linop!(out, z)
        β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0), z=z)::Float64
        fill!(out, 0.0)
        if !thg
            βref = Modes.β(modes[ref_mode], wlfreq(λ0), z=z)
        end
        # NOTE here we assume that all modes have the same gas fill
        εco[grid.sidx] .= modes[ref_mode].coren.(grid.ω[grid.sidx], z=z).^2
        for i in eachindex(modes)
            for iω in sidcs
                nc = conj_clamp(neff(modes[i], εco[iω], nwg[iω, i]), grid.ω[iω])
                out[iω, i] = -im*(grid.ω[iω]/c*nc - (grid.ω[iω] - grid.ω0)*β1)
                if !thg
                    out[iω, i] -= -im*βref
                end
            end
        end
    end
end

end
