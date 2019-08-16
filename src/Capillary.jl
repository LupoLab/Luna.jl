module Capillary
import PhysicalConstants
import Unitful
import FunctionZeros: besselj_zero
import Roots: fzero
import Cubature: hquadrature, hcubature
import SpecialFunctions: besselj
import StaticArrays: SVector
import LinearAlgebra: dot, norm
import Luna: Maths
import Luna.PhysData: c, ε_0, χ1, ref_index, μ_0

function getunm(n, m, kind)
    if (kind == :TE) || (kind == :TM)
        if (n != 0) || (m != 1)
            error("n=0, m=1 for TE or TM modes")
        end
        return besselj_zero(1, 1)
    elseif kind == :HE
        return besselj_zero(n-1, m)
    else
        error("kind must be :TE, :TM or :HE")
    end
end

function β(a, ω; gas::Symbol=:He, pressure=0, n=1, m=1)
    χ = pressure .* χ1(gas, 2π*c./ω)
    unm = getunm(n, m, :HE)
    return @. ω/c*(1 + χ/2 - c^2*unm^2/(2*ω^2*a^2))
end

function β(a; λ, gas::Symbol=:He, pressure=0, n=1, m=1)
    return β(a, 2π*c./λ, gas=gas, pressure=pressure, n=n, m=m)
end

function α(a, ω; n=1, m=1, kind=:HE)
    unm = getunm(n, m, kind)
    ν = ref_index(:SiO2, 2π*c./ω)
    if kind == :He
        vp = @. (ν^2 + 1)/(2*real(sqrt(Complex(ν^2-1))))
    elseif kind == :TE
        vp = @. 1/(real(sqrt(Complex(ν^2-1))))
    elseif kind == :TM
        vp = @. ν^2/(real(sqrt(Complex(ν^2-1))))
    else
        error("kind must be :TE, :TM or :HE")
    end
    return @. 2*(c^2 * unm^2)/(a^3 * ω^2) * vp
end

# TODO: why do we repeat this?
function α(a, ω::AbstractArray; n=1, m=1, kind=:HE)
    unm = getunm(n, m, kind)
    ν = ref_index(:SiO2, 2π*c./ω)
    ν[ω .< 3e14] .= 1.4
    if kind == :He
        vp = @. (ν^2 + 1)/(2*real(sqrt(Complex(ν^2-1))))
    elseif kind == :TE
        vp = @. 1/(real(sqrt(Complex(ν^2-1))))
    elseif kind == :TM
        vp = @. ν^2/(real(sqrt(Complex(ν^2-1))))
    else
        error("kind must be :TE, :TM or :HE")
    end
    ret = @. 2*(c^2 * unm^2)/(a^3 * ω^2) * vp
    ret[isinf.(ret)] .= 0 # TODO FIND OUT WHY THIS IS NEEDED
    return ret
end

function α(a; λ, n=1, m=1, kind=:HE)
    return α(a, 2π*c./λ, n=n, m=m, kind=:He)
end

function losslength(a, ω; n=1, m=1, kind=:HE)
    return 1 ./ α(a, ω, n=n, m=m, kind=:HE)
end

function losslength(a; λ, n=1, m=1, kind=:HE)
    return 1 ./ α(a, n=n, m=m, λ=λ, kind=:HE)
end

function transmission(a, L; λ, n=1, m=1, kind=:HE)
    return @. exp(-α(a, λ=λ, n=n, m=m, kind=:HE)*L)
end

function dB_per_m(a, ω; n=1, m=1, kind=:HE)
    return 10/log(10).*α(a, ω, n=n, m=m, kind=:HE)
end

function dB_per_m(a; n=1, m=1, λ=λ, kind=:HE)
    return return 10/log(10) .* α(a, n=n, m=m, λ=λ, kind=:HE)
end

function dispersion_func(order, a; gas::Symbol=:He, pressure=0, n=1, m=1)
    βn(ω) = Maths.derivative(ω -> β(a, ω, gas=gas, pressure=pressure, n=n, m=m), ω, order)
    return βn
end

function dispersion(order, a, ω; gas::Symbol=:He, pressure=0, n=1, m=1)
    return dispersion_func(order, a, gas=gas, pressure=pressure, n=n, m=m).(ω)
end

function dispersion(order, a; λ, gas::Symbol=:He, pressure=0, n=1, m=1)
    return dispersion(order, a, 2π*c./λ, gas=gas, pressure=pressure, n=n, m=m)
end

function zdw(a; gas::Symbol, pressure, n=1, m=1)
    ω0 = fzero(dispersion_func(2, a, gas=gas, pressure=pressure, n=n, m=m), 1e14, 2e16)
    return 2π*c/ω0
end

"Create function that returns (r,θ) -> (Ex, Ey)"
function field(a, n, m, kind; ϕ=0.0)
    unm = getunm(n, m, kind)
    if kind == :HE
        return (r, θ) -> besselj(n-1, r*unm/a) .* SVector(cos(θ)*sin(n*(θ + ϕ)) - sin(θ)*cos(n*(θ + ϕ)),
                                                          sin(θ)*sin(n*(θ + ϕ)) + cos(θ)*cos(n*(θ + ϕ)))
    elseif kind == :TE
        return (r, θ) -> besselj(1, r*unm/a) .* SVector(-sin(θ), cos(θ))
    elseif kind == :TM
        return (r, θ) -> besselj(1, r*unm/a) .* SVector(cos(θ), sin(θ))
    end
end

"Get mode normalization constant"
function N(a, n, m, kind; ϕ=0.0)
    f = field(a, n, m, kind, ϕ=ϕ)
    function Nfunc(x)
        r = x[1]
        θ = x[2]
        E = f(r, θ)
        r*sqrt(ε_0/μ_0)*dot(E, E)
    end
    val, err = hcubature(Nfunc, (0.0, 0.0), (a, 2π))
    0.5*abs(val)
end

"Create function that returns normalised (r,θ) -> |E|"
function emagh(a, n, m, kind; ϕ=0.0)
    func = let sN = sqrt(N(a, n, m, kind, ϕ=ϕ)), f = field(a, n, m, kind, ϕ=ϕ)
        function func(r,θ)
            norm(f(r,θ) ./ sN)
        end
    end
end

"Create function that returns normalised (r,θ) -> (Ex, Ey)"
function getExy(a, n, m, kind; ϕ=0.0)
    func = let sN = sqrt(N(a, n, m, kind, ϕ=ϕ)), f = field(a, n, m, kind, ϕ=ϕ)
        function func(r,θ)
            f(r,θ) ./ sN
        end
    end
end

"Get effective area of mode"
function Aeff(a, n, m, kind; ϕ=0.0)
    em = emagh(a, n, m, kind, ϕ=ϕ)
    function Aefft(x)
        r = x[1]
        θ = x[2]
        e = em(r, θ)
        r*e^2
    end
    val, err = hcubature(Aefft, (0.0, 0.0), (a, 2π))
    ret = val^2
    function Aeffb(x)
        r = x[1]
        θ = x[2]
        e = em(r, θ)
        r*e^4
    end
    val, err = hcubature(Aeffb, (0.0, 0.0), (a, 2π))
    return ret / val
end

"Create function that returns (r,θ) -> ((Ex, Ey),(Ex,Ey),...) for each mode"
function fields(a, modes; components=:Ey)
    if components == :Ey
        indices = 2
    elseif components == :Ex
        indices = 1
    elseif components == :Exy
        indices = 1:2
    else
        error("components must be one of :Ex, :Ey or :Exy")
    end
    Ets = []
    for i = 1:length(modes)
        push!(Ets, Capillary.getExy(a, modes[i][1], modes[i][2], modes[i][3]))
    end
    Exy = Array{Float64,2}(undef,size(modes,1),length(indices))
    
    ret = let Exy=Exy
        function ret(r,θ)
            for i = 1:length(modes)
                Exy[i,:] .= Ets[i](r,θ)[indices]
            end
            Exy
        end
    end
    ret
end

end