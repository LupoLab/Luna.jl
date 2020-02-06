module Capillary
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import StaticArrays: SVector
using Reexport
@reexport using Luna.Modes
import Luna: Maths
import Luna.PhysData: c, ref_index_fun, roomtemp, densityspline, sellmeier_gas
import Luna.Modes: AbstractMode, dimlimits, neff, field
import Luna.PhysData: wlfreq

export MarcatilliMode, dimlimits, neff, field

"Marcatili mode"
struct MarcatilliMode{Ta, Tcore, Tclad, LT} <: AbstractMode
    a::Ta # core radius callable as function of z only
    n::Int # radial mode index
    m::Int # azimuthal mode index
    kind::Symbol # kind of mode (transverse magnetic/electric or hybrid)
    unm::Float64 # mth zero of the nth Bessel function of the first kind
    ϕ::Float64 # overall rotation angle of the mode
    coren::Tcore # callable, returns (possibly complex) core ref index as function of ω
    cladn::Tclad # callable, returns (possibly complex) cladding ref index as function of ω
    model::Symbol # if :full, includes complete influence of complex cladding ref index
    loss::LT # Val{true}() or Val{false}() - whether to include the loss
end

function MarcatilliMode(afun, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)
    # chkzkwarg makes sure that coren and cladn take z as a keyword argument
    MarcatilliMode(afun, n, m, kind, get_unm(n, m, kind), ϕ,
                   chkzkwarg(coren), chkzkwarg(cladn),
                   model, Val(loss))
end

"convenience constructor assuming single gas filling"
function MarcatilliMode(a, gas, P; n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, clad=:SiO2, loss=true)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> rfs(wlfreq(ω))
    afun(z) = a
    MarcatilliMode(afun, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"convenience constructor for non-constant core index"
function MarcatilliMode(a::Number, coren; n=1, m=1, kind=:HE, ϕ=0.0, model=:full, clad=:SiO2, loss=true)
    rfs = ref_index_fun(clad)
    cladn = (ω; z) -> rfs(wlfreq(ω))
    afun(z) = a
    MarcatilliMode(afun, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"convenience constructor for non-constant core index and core radius"
function MarcatilliMode(afun, coren; n=1, m=1, kind=:HE, ϕ=0.0, model=:full, clad=:SiO2, loss=true)
    rfs = ref_index_fun(clad)
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatilliMode(afun, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"complex effective index of Marcatilli mode with dielectric core and arbitrary
 (metal or dielectric) cladding.

Adapted from
Marcatili, E. & Schmeltzer, R.
Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation).
Bell System Technical Journal 43, 1783–1809 (1964).
"
function neff(m::MarcatilliMode, ω; z=0)
    εcl = m.cladn(ω, z=z)^2
    εco = m.coren(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    neff(m, ω, εcl, εco, vn, m.a(z))
end

# Dispatch on loss to make neff type stable
# m.loss = Val{true}() (returns ComplexF64)
function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{true}}, ω, εcl, εco, vn, a) where Ta where Tcl where Tco
    if m.model == :full
        k = ω/c
        return sqrt(complex(εco - (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2))
    elseif m.model == :reduced
        return ((1 + (εco - 1)/2 - c^2*m.unm^2/(2*ω^2*a^2))
                    + im*(c^3*m.unm^2)/(a^3*ω^3)*vn)
    else
        error("model must be :full or :reduced")
    end 
end

# m.loss = Val{false}() (returns Float64)
function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{false}}, ω, εcl, εco, vn, a) where Ta where Tcl where Tco
    if m.model == :full
        k = ω/c
        return real(sqrt(εco - (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2))
    elseif m.model == :reduced
        return real(1 + (εco - 1)/2 - c^2*m.unm^2/(2*ω^2*a^2))
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

dimlimits(m::MarcatilliMode; z=0) = (:polar, (0.0, 0.0), (m.a(z), 2π))

# we use polar coords, so xs = (r, θ)
function field(m::MarcatilliMode; z=0)
    if m.kind == :HE
        return (xs) -> besselj(m.n-1, xs[1]*m.unm/m.a(z)) .* SVector(
            cos(xs[2])*sin(m.n*(xs[2] + m.ϕ)) - sin(xs[2])*cos(m.n*(xs[2] + m.ϕ)),
            sin(xs[2])*sin(m.n*(xs[2] + m.ϕ)) + cos(xs[2])*cos(m.n*(xs[2] + m.ϕ))
            )
    elseif m.kind == :TE
        return (xs) -> besselj(1, xs[1]*m.unm/m.a(z)) .* SVector(-sin(xs[2]), cos(xs[2]))
    elseif m.kind == :TM
        return (xs) -> besselj(1, xs[1]*m.unm/m.a(z)) .* SVector(cos(xs[2]), sin(xs[2]))
    end
end

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
end
