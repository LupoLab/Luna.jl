module RectModes
import StaticArrays: SVector
using Reexport
@reexport using Luna.Modes
import Luna: Maths
import Luna.PhysData: c, ε_0, μ_0, ref_index_fun, roomtemp
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N

export RectMode, dimlimits, neff, field, N, Aeff

# core and clad are function-like objects which return the
# (possibly complex) refractive index as a function of freq
# pol is either :x or :y
# a and b are the half widths of the waveguide in each dimension.
struct RectMode{Ta, Tb, Tcore, Tclad} <: AbstractMode
    a::Ta
    b::Tb
    n::Int
    m::Int
    pol::Symbol
    coren::Tcore
    cladn::Tclad
end

RectMode(a::Number, args...; kwargs...) = RectMode(z->a, args...; kwargs...)
RectMode(afun, b::Number, args...; kwargs...) = RectMode(afun, z->b, args...; kwargs...)
RectMode(a::Number, b::Number, args...; kwargs...) = RectMode(z->a, z->b, args...; kwargs...)

"convenience constructor assunming single gas filling and specified cladding"
function RectMode(afun, bfun, gas, P, clad; n=1, m=1, pol=:x, T=roomtemp)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = (ω; z) -> rfg(2π*c./ω)
    cladn = (ω; z) -> rfs(2π*c./ω)
    RectMode(afun, bfun, n, m, pol, coren, cladn)
end

"convenience constructor for non-constant core index"
function RectMode(afun, bfun, coren, clad; n=1, m=1, pol=:x)
    rfs = ref_index_fun(clad)
    cladn = (ω; z) -> rfs(2π*c./ω)
    RectMode(afun, bfun, n, m, pol, coren, cladn)
end

dimlimits(m::RectMode; z=0) = (:cartesian, (-m.a(z), -m.b(z)), (m.a(z), m.b(z)))

"effective index of rectangular mode with dielectric core and arbitrary
 (metal or dielectric) cladding.

Adapted from
Laakmann, K. D. & Steier, W. H.
Waveguides: characteristic modes of hollow rectangular dielectric waveguides.
Appl. Opt., AO 15, 1334–1340 (1976).

I had to re-derive the result in order to get the complex cladding index contribution
to the real part of neff.
"
function neff(m::RectMode, ω; z=0)
    εcl = m.cladn(ω, z=z)^2
    εco = m.coren(ω, z=z)^2
    λ = 2π*c./ω
    if m.pol == :x
        ac = εcl/sqrt(complex(εcl - 1))
        bc = 1/sqrt(complex(εcl - 1))
    elseif m.pol == :y
        bc = εcl/sqrt(complex(εcl - 1))
        ac = 1/sqrt(complex(εcl - 1))
    else
        error("RectMode pol must be either :x or :y")
    end
    sqrt(complex(εco - (m.m*λ/(4*m.a(z)))^2*(1 - im*λ/(2π*m.a(z))*ac)^2
                     - (m.n*λ/(4*m.b(z)))^2*(1 - im*λ/(2π*m.b(z))*bc)^2))
end

# here we use cartesian coords, so xs = (x, y)
function field(m::RectMode, xs; z=0)
    if isodd(m.m)
        Ea = cos(m.m*π*xs[1]/(2*m.a(z)))
    else
        Ea = sin(m.m*π*xs[1]/(2*m.a(z)))
    end
    if isodd(m.n)
        Eb = cos(m.n*π*xs[2]/(2*m.b(z)))
    else
        Eb = sin(m.n*π*xs[2]/(2*m.b(z)))
    end
    E = Ea*Eb
    if m.pol == :x
        return SVector(E, 0.0)
    else
        return SVector(0.0, E)
    end
end

N(m::RectMode; z=0) = 0.5*sqrt(ε_0/μ_0)*m.a(z)*m.b(z)

Aeff(m::RectMode; z=0) = 16/9*m.a(z)*m.b(z)

end
