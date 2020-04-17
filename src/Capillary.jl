module Capillary
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import StaticArrays: SVector
import Cubature: hquadrature
using Reexport
@reexport using Luna.Modes
import Luna: Maths
import Luna.PhysData: c, ε_0, μ_0, ref_index_fun, roomtemp, densityspline, sellmeier_gas
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N
import Luna.PhysData: wlfreq
import Base: show

export MarcatilliMode, dimlimits, neff, field, N, Aeff

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
    aeff_intg::Float64 # Pre-calculated integral fraction for effective area
end

function show(io::IO, m::MarcatilliMode)
    a = "a(z=0)=$(m.a(0))"
    loss = "loss=" * (m.loss == Val(true) ? "true" : "false")
    model = "model="*string(m.model)
    out = "MarcatilliMode{"*join([mode_string(m), a, loss, model], ", ")*"}"
    print(io, out)
end

mode_string(m::MarcatilliMode) = string(m.kind)*string(m.n)*string(m.m)

function MarcatilliMode(a::Number, args...; kwargs...)
    afun(z) = a
    MarcatilliMode(afun, args...; kwargs...)
end

function MarcatilliMode(afun, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)
    # chkzkwarg makes sure that coren and cladn take z as a keyword argument
    aeff_intg = Aeff_Jintg(n, get_unm(n, m, kind), kind)
    MarcatilliMode(afun, n, m, kind, get_unm(n, m, kind), ϕ,
                   chkzkwarg(coren), chkzkwarg(cladn),
                   model, Val(loss), aeff_intg)
end

"convenience constructor assuming single gas filling"
function MarcatilliMode(afun, gas, P;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full,
                        clad=:SiO2, loss=true)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatilliMode(afun, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"convenience constructor assuming single gas filling but custom cladding index"
function MarcatilliMode(afun, gas, P, cladn;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, loss=true)
    rfg = ref_index_fun(gas, P, T)
    coren = (ω; z) -> rfg(wlfreq(ω))
    MarcatilliMode(afun, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"convenience constructor for non-constant core index"
function MarcatilliMode(afun, coren;
                        n=1, m=1, kind=:HE, ϕ=0.0, model=:full, clad=:SiO2, loss=true)
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
function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{true}}, ω, εcl, εco, vn, a) where {Ta, Tcl, Tco}
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
function neff(m::MarcatilliMode{Ta, Tco, Tcl, Val{false}}, ω, εcl, εco, vn, a) where {Ta, Tcl, Tco}
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
function field(m::MarcatilliMode, xs; z=0)
    if m.kind == :HE
        return (besselj(m.n-1, xs[1]*m.unm/m.a(z)) .* SVector(
            cos(xs[2])*sin(m.n*(xs[2] + m.ϕ)) - sin(xs[2])*cos(m.n*(xs[2] + m.ϕ)),
            sin(xs[2])*sin(m.n*(xs[2] + m.ϕ)) + cos(xs[2])*cos(m.n*(xs[2] + m.ϕ))
            ))
    elseif m.kind == :TE
        return besselj(1, xs[1]*m.unm/m.a(z)) .* SVector(-sin(xs[2]), cos(xs[2]))
    elseif m.kind == :TM
        return besselj(1, xs[1]*m.unm/m.a(z)) .* SVector(cos(xs[2]), sin(xs[2]))
    end
end

function N(m::MarcatilliMode; z=0)
    np1 = (m.kind == :HE) ? m.n : 2
    π/2 * m.a(z)^2 * besselj(np1, m.unm)^2 * sqrt(ε_0/μ_0)
end

function Aeff_Jintg(n, unm, kind)
    den, err = hquadrature(r -> r*besselj(n-1, unm*r)^4, 0, 1)
    np1 = (kind == :HE) ? n : 2
    num = 1/4 * besselj(np1, unm)^4
    return 2π*num/den
end

Aeff(m::MarcatilliMode; z=0) = m.a(z)^2 * m.aeff_intg


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

end
