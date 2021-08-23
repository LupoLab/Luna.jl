module StepIndexFibre
import StaticArrays: SVector
using Reexport
@reexport using Luna.Modes
import Luna.PhysData: c, ε_0, μ_0, ref_index_fun, wlfreq
import Luna.Modes: AbstractMode, dimlimits, neff, field
export StepIndexMode, dimlimits, neff, field
import SpecialFunctions: besselj, besselk
import Roots: find_zeros
import Memoize: @memoize
import Luna.Utils: subscript
import Base: show
import Luna.Maths: CSpline

"""
    StepIndexMode(a, n, m, kind, coren, cladn; parity=:even, pts=100)

Create a StepIndexMode.

# Arguments
- `a` : Either a `Number` for constant core radius, or a function `a(z)` for variable radius.
- `n::Int` : Azimuthal mode index (number of nodes in the field along azimuthal angle).
- `m::Int` : Radial mode index (number of nodes in the field along radial coordinate).
- `kind::Symbol` : `:TE` for transverse electric, `:TM` for transverse magnetic,
                   `:HE` or `:EH`, following Snyder and Love convention.
- `coren` : Callable `coren(ω; z)` which returns the refractive index of the core
- `cladn` : Callable `cladn(ω; z)` which returns the refractive index of the cladding
- `parity::Symbol` : `:even` or `:odd`, following Snyder and Love convention.
- `pts::Int` : number of grid points to use in zero search.

"""
struct StepIndexMode{Ta, Tcore, Tclad, AT, NT} <: AbstractMode
    a::Ta # core radius callable as function of z only, or fixed core radius if a Number
    n::Int # azimuthal mode index
    m::Int # radial mode index
    kind::Symbol # kind of mode (transverse magnetic/electric or hybrid)
    parity::Symbol # :odd or :even
    coren::Tcore # callable, returns (possibly complex) core ref index as function of ω
    cladn::Tclad # callable, returns (possibly complex) cladding ref index as function of ω
    pts::Int # number of grid points to use in zero search
    accel::AT # Val{true}() or Val{false}() - whether to accelerate neff search
    neff::NT # neff accelerator (a spline)
end

function StepIndexMode(a, n, m, kind, parity, coren, cladn, pts, accellims=nothing)
    if isnothing(accellims)
        return StepIndexMode(a, n, m, kind, parity, coren, cladn, pts, Val(false), nothing)
    else
        λmin, λmax, npts = accellims
        ωs = collect(range(wlfreq(λmax), wlfreq(λmin), length=npts))
        neffs = findneff.(a, ωs ./ c, coren.(ωs, z=0), cladn.(ωs, z=0), n, m, kind, pts)
        neff = CSpline(ωs, neffs)
        return StepIndexMode(a, n, m, kind, parity, coren, cladn, pts, Val(true), neff)
    end
end

"""
    StepIndexMode(a; n=1, m=1, kind=:HE, core=:SiO2, clad=:Air, parity=:even,
                  pts=100, accellims=nothing)

Create a StepIndexMode. Defaults to a silica strand in air.

# Arguments
- `a` : Either a `Number` for constant core radius, or a function `a(z)` for variable radius.
- `n::Int=1` : Azimuthal mode index (number of nodes in the field along azimuthal angle).
- `m::Int=1` : Radial mode index (number of nodes in the field along radial coordinate).
- `kind::Symbol=:HE` : `:TE` for transverse electric, `:TM` for transverse magnetic,
                   `:HE` or `:EH`, following Snyder and Love convention.
- `core=:Air` : The core material.
- `clad=:SiO2` : The clad material.
- `parity::Symbol=:even` : `:even` or `:odd`, following Snyder and Love convention.
- `pts::Int=100` : number of grid points to use in zero search.
- `accellims::Tuple=nothing` : can be set to (λmin, λmax, npts) to build a spline to
   accelerate neff lookup.

"""
function StepIndexMode(a; n=1, m=1, kind=:HE, core=:SiO2, clad=:Air, parity=:even,
                       pts=100, accellims=nothing)
    rfco = ref_index_fun(core)
    rfcl = ref_index_fun(clad)
    coren = (ω; z) -> real(rfco(wlfreq(ω)))
    cladn = (ω; z) -> real(rfcl(wlfreq(ω)))
    StepIndexMode(a, n, m, kind, parity, coren, cladn, pts, accellims)
end

"""
    StepIndexMode(a, NA; n=1, m=1, kind=:HE, clad=:SiO2, parity=:even,
                  pts=100, accellims=nothing)

Create a StepIndexMode based on the NA and specified cladding material.

# Arguments
- `a` : Either a `Number` for constant core radius, or a function `a(z)` for variable radius.
- `NA` : The numerical aperture of the mode.
- `n::Int=1` : Azimuthal mode index (number of nodes in the field along azimuthal angle).
- `m::Int=1` : Radial mode index (number of nodes in the field along radial coordinate).
- `kind::Symbol=:HE` : `:TE` for transverse electric, `:TM` for transverse magnetic,
                   `:HE` or `:EH`, following Snyder and Love convention.
- `clad=:SiO2` : The clad material.
- `parity::Symbol=:even` : `:even` or `:odd`, following Snyder and Love convention.
- `pts::Int=100` : number of grid points to use in zero search.
- `accellims::Tuple=nothing` : can be set to (λmin, λmax, npts) to build a spline to
   accelerate neff lookup.

"""
function StepIndexMode(a, NA; n=1, m=1, kind=:HE, clad=:SiO2,  parity=:even, pts=100,
                       accellims=nothing)
    rfcl = ref_index_fun(clad)
    function rfco(λ)
        ncl = real(rfcl(λ))
        ncl + NA^2/(2*ncl)
    end
    coren = (ω; z) -> rfco(wlfreq(ω))
    cladn = (ω; z) -> real(rfcl(wlfreq(ω)))
    StepIndexMode(a, n, m, kind, parity, coren, cladn, pts, accellims)
end

function show(io::IO, m::StepIndexMode)
    a = radius_string(m)
    out = "StepIndexMode{"*join([mode_string(m), a], ", ")*"}"
    print(io, out)
end

mode_string(m::StepIndexMode) = string(m.kind)*subscript(m.n)*subscript(m.m)
radius_string(m::StepIndexMode{<:Number, Tco, Tcl, AT, NT}) where {Tco, Tcl, AT, NT} = "a=$(m.a)"
radius_string(m::StepIndexMode) = "a(z=0)=$(radius(m, 0))"

besseljp(n, z) = 0.5*(besselj(n - 1, z) - besselj(n + 1, z))

besselkp(n, z) = -0.5*(besselk(n - 1, z) + besselk(n + 1, z))

function uwv(radius, k0, ncore, nclad, neff)
    # From Snyder & Love, 1983, Eqns. 11-49, 11-50, Pages 227, 228
    rk0 = radius*k0
    u = rk0*sqrt(ncore^2 - neff^2)
    w = rk0*sqrt(neff^2 - nclad^2)
    v = sqrt(u^2 + w^2)
    u, w, v
end

function R(ncore, nclad, n, u, w, neff)
    kpart = ((ncore^2 - nclad^2)/(2*ncore^2))^2*(besselkp(n, w)/(w*besselk(n, w)))^2
    rightpart = (n*neff/ncore)^2*(1/u^2 + 1/w^2)^2
    sqrt(kpart + rightpart)
end

char_LHS(n, u) = besseljp(n, u)/(u*besselj(n, u))

char_RHS(ncore, nclad, n, w) = -(ncore^2 + nclad^2)/(2*ncore^2)*besselkp(n, w)/(w*besselk(n, w))

function char_EH(radius, k0, ncore, nclad, n, neff)
    u, w, v = uwv(radius, k0, ncore, nclad, neff)
    char_LHS(n, u) - char_RHS(ncore, nclad, n, w) - R(ncore, nclad, n, u, w, neff)
end

function char_HE(radius, k0, ncore, nclad, n, neff)
    u, w, v = uwv(radius, k0, ncore, nclad, neff)
    char_LHS(n, u) - char_RHS(ncore, nclad, n, w) + R(ncore, nclad, n, u, w, neff)
end

function char_TE(radius, k0, ncore, nclad, neff)
    char_HE(radius, k0, ncore, nclad, 0, neff)
end

function char_TM(radius, k0, ncore, nclad, neff)
    char_EH(radius, k0, ncore, nclad, 0, neff)
end

function make_char(radius, k0, ncore, nclad, n, mode)
    if mode == :HE
        return let radius=radius, k0=k0, ncore=ncore, nclad=nclad, n=n
            char(neff) = char_HE(radius, k0, ncore, nclad, n, neff)
        end
    elseif mode == :EH
        return let radius=radius, k0=k0, ncore=ncore, nclad=nclad, n=n
            char(neff) = char_EH(radius, k0, ncore, nclad, n, neff)
        end
    elseif mode == :TE
        if n != 0
            error("n must be 0 for TE modes")
        end
        return let radius=radius, k0=k0, ncore=ncore, nclad=nclad
            char(neff) = char_TE(radius, k0, ncore, nclad, neff)
        end
    elseif mode == :TM
        if n != 0
            error("n must be 0 for TM modes")
        end
        return let radius=radius, k0=k0, ncore=ncore, nclad=nclad
            char(neff) = char_TM(radius, k0, ncore, nclad, neff)
        end
    else
        error("unkown mode: $mode")
    end
end

@memoize function findneff(radius, k0, ncore, nclad, n, m, mode=:HE, pts=100)
    char = make_char(radius, k0, ncore, nclad, n, mode)
    roots = find_zeros(char, nclad, ncore, no_pts=pts)
    roots[end - (m - 1)]
end

radius(m::StepIndexMode{<:Number, Tco, Tcl, AT, NT}, z) where {Tcl, Tco, AT, NT} = m.a
radius(m::StepIndexMode, z) = m.a(z)

dimlimits(m::StepIndexMode; z=0) = (:polar, (0.0, 0.0), (10*radius(m, z), 2π))

"""
    neff(m::StepIndexMode, ω; z=0)
    
Calculate the effective index of a StepIndexMode.

"""
function neff(m::StepIndexMode, ω; z=0)
    findneff(radius(m, z), ω/c, m.coren(ω, z=z), m.cladn(ω, z=z), m.n, m.m, m.kind, m.pts)
end

function neff(m::StepIndexMode{<:Number, Tco, Tcl, Val{true}, NT}, ω; z=0) where {Tcl, Tco, NT}
    m.neff(ω)
end

function f1f2(radius, k0, ncore, nclad, neff, n)
    # From Snyder & Love, 1983, Table 12-3, Page 250
    u, w, v = uwv(radius, k0, ncore, nclad, neff)
    b1 =  1/(2*u)*(besselj(n - 1, u) - besselj(n + 1, u))/besselj(n, u)
    b2 = -1/(2*w)*(besselk(n - 1, w) + besselk(n + 1, w))/besselk(n, w)
    delta = (ncore^2 - nclad^2)/(2*ncore^2)
    f1 = (u*w/v)^2*(b1 + (1 - 2*delta)*b2)/n
    f2 = (v/(u*w))^2*n/(b1 + b2)
    f1, f2, u, w, v
end

function corefields(n, u, R, a1, a2, fv, gv)
    jm1_ur = besselj(n - 1, u*R)
    jp1_ur = besselj(n + 1, u*R)
    j_u = besselj(n, u)
    Er = -(a1*jm1_ur + a2*jp1_ur)/j_u*fv
    Et = -(a1*jm1_ur - a2*jp1_ur)/j_u*gv
    Er, Et
end

function cladfields(n, u, w, R, a1, a2, fv, gv)
    km1_wr = besselk(n - 1, w*R)
    kp1_wr = besselk(n + 1, w*R)
    k_w = besselk(n, w)
    Er = -u/w*(a1*km1_wr - a2*kp1_wr)/k_w*fv
    Et = -u/w*(a1*km1_wr + a2*kp1_wr)/k_w*gv
    Er, Et
end

# we use polar coords, so xs = (r, θ)
# TODO: how do we handle wavelength dependence?
# TODO: how do we handle non-negligible z component?
function field(m::StepIndexMode, xs; z=0, ω=wlfreq(1030e-9))
    # From Snyder & Love, 1983, Table 12-3, Page 250
    r, θ = xs[1], xs[2]
    f1, f2, u, w, v = f1f2(radius(m, z), ω/c, m.coren(ω, z=z), m.cladn(ω, z=z), neff(m, ω; z=z), m.n)
    a1 = (f2 - 1)/2
    a2 = (f2 + 1)/2
    fv = m.parity == :even ? cos(m.n*θ) : sin(m.n*theta)
    gv = m.parity == :even ? -sin(m.n*θ) : cos(m.n*θ)
    R = r/radius(m, z)
    Er, Et = abs(R) <= 1.0 ? corefields(m.n, u, R, a1, a2, fv, gv) : cladfields(m.n, u, w, R, a1, a2, fv, gv)
    SVector(Er*cos(θ) - Et*sin(θ), Er*sin(θ) + Et*cos(θ))
end

end
