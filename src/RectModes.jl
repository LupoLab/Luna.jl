module RectModes
import StaticArrays: SVector
using Reexport
@reexport using Luna.Modes
import Luna: Maths
import Luna.PhysData: c, ref_index_fun, roomtemp
import Luna.Modes: FreeMode, dimlimits, neff, field

export RectMode, dimlimits, neff, field

# core and clad are function-like objects which return the
# (possibly complex) refractive index as a function of freq
# pol is either :x or :y
# a and b are the half widths of the waveguide in each dimension.
struct RectMode{Tcore, Tclad} <: FreeMode
    a::Float64
    b::Float64
    n::Int
    m::Int
    pol::Symbol
    coren::Tcore
    cladn::Tclad
end

"convenience constructor assunming single gas filling and specified cladding"
function RectMode(a, b, gas, P, clad; n=1, m=1, pol=:x, T=roomtemp)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = ω -> rfg(2π*c./ω)
    cladn = ω -> rfs(2π*c./ω)
    RectMode(a, b, n, m, pol, coren, cladn)
end

dimlimits(m::RectMode) = (:cartesian, (-m.a, -m.b), (m.a, m.b))

"effective index of rectabgular mode with dielectric core and arbitrary
 (metal or dielectric) cladding.

Adapted from
Laakmann, K. D. & Steier, W. H.
Waveguides: characteristic modes of hollow rectangular dielectric waveguides.
Appl. Opt., AO 15, 1334–1340 (1976).

I had to re-derive the result in order to get the complex cladding index contribution
to the real part of neff.
"
function neff(m::RectMode, ω)
    εcl = m.cladn(ω)^2
    εco = m.coren(ω)^2
    λ = 2π*c./ω
    if m.pol == :x
        ac = εcl/sqrt(Complex(εcl - 1))
        bc = 1/sqrt(Complex(εcl - 1))
    elseif m.pol == :y
        bc = εcl/sqrt(Complex(εcl - 1))
        ac = 1/sqrt(Complex(εcl - 1))
    else
        error("RectMode pol must be either :x or :y")
    end
    sqrt(Complex(εco - (m.m*λ/(4*m.a))^2*(1 - im*λ/(2π*m.a)*ac)^2
                     - (m.n*λ/(4*m.b))^2*(1 - im*λ/(2π*m.b)*bc)^2))
end

# here we use cartesian coords, so xs = (x, y)
function field(m::RectMode)
    if isodd(m.m)
        Ea = (x) -> cos(m.m*π*x/(2*m.a))
    else
        Ea = (x) -> sin(m.m*π*x/(2*m.a))
    end
    if isodd(m.n)
        Eb = (x) -> cos(m.n*π*x/(2*m.b))
    else
        Eb = (x) -> sin(m.n*π*x/(2*m.b))
    end
    if m.pol == :x
        return (xs) -> SVector(Ea(xs[1])*Eb(xs[2]), 0.0)
    else
        return (xs) -> SVector(0.0, Ea(xs[1])*Eb(xs[2]))
    end
end

end
