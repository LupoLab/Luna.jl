module AbstractModes
import Roots: fzero
import Cubature: hcubature
import LinearAlgebra: dot, norm
import Luna: Maths
import Luna.PhysData: c, ε_0, μ_0

abstract type AbstractMode end

"Maximum dimensional limits of validity for this mode"
function dimlimits(m::M) where {M <: AbstractMode}
    error("abstract method called")
end

function β(m::M, ω) where {M <: AbstractMode}
    error("abstract method called")
end

function β(m::M; λ) where {M <: AbstractMode}
    return β(m, 2π*c./λ)
end

function α(m::M, ω) where {M <: AbstractMode}
    error("abstract method called")
end

function α(m::M; λ) where {M <: AbstractMode}
    return α(m, 2π*c./λ)
end

function losslength(m::M, ω) where {M <: AbstractMode}
    return 1 ./ α(m, ω)
end

function losslength(m::M; λ) where {M <: AbstractMode}
    return losslength(m::M, 2π*c./λ) 
end

function transmission(m::M, L; λ) where {M <: AbstractMode}
    return @. exp(-α(m, λ=λ)*L)
end

function dB_per_m(m::M, ω) where {M <: AbstractMode}
    return 10/log(10).*α(m, ω)
end

function dB_per_m(m::M; λ) where {M <: AbstractMode}
    return return 10/log(10) .* α(m, λ=λ)
end

function dispersion_func(m::M, order) where {M <: AbstractMode}
    βn(ω) = Maths.derivative(ω -> β(m, ω), ω, order)
    return βn
end

function dispersion(m::M, order, ω) where {M <: AbstractMode}
    return dispersion_func(m, order).(ω)
end

function dispersion(m::M, order; λ) where {M <: AbstractMode}
    return dispersion(m, order, 2π*c./λ)
end

function zdw(m::M) where {M <: AbstractMode}
    ω0 = fzero(dispersion_func(m, 2), 1e14, 2e16) # TODO magic numbers
    return 2π*c/ω0
end

"Create function that returns (r,θ) -> (Ex, Ey)"
function field(m::M) where {M <: AbstractMode}
    error("abstract method called")
end

"Get mode normalization constant"
function N(m::M) where {M <: AbstractMode}
    f = field(m)
    function Nfunc(x)
        r = x[1]
        θ = x[2]
        E = f(r, θ)
        r*sqrt(ε_0/μ_0)*dot(E, E)
    end
    val, err = hcubature(Nfunc, dimlimits(m)[1], dimlimits(m)[2])
    0.5*abs(val)
end

"Create function that returns normalised (r,θ) -> |E|"
function absE(m::M) where {M <: AbstractMode}
    func = let sN = sqrt(N(m)), f = field(m)
        function func(r,θ)
            norm(f(r,θ) ./ sN)
        end
    end
end

"Create function that returns normalised (r,θ) -> (Ex, Ey)"
function Exy(m) where {M <: AbstractMode}
    func = let sN = sqrt(N(m)), f = field(m)
        function func(r,θ)
            f(r,θ) ./ sN
        end
    end
end

"Get effective area of mode"
function Aeff(m) where {M <: AbstractMode}
    em = absE(m)
    function Aefft(x)
        r = x[1]
        θ = x[2]
        e = em(r, θ)
        r*e^2
    end
    val, err = hcubature(Aefft, dimlimits(m)[1], dimlimits(m)[2])
    ret = val^2
    function Aeffb(x)
        r = x[1]
        θ = x[2]
        e = em(r, θ)
        r*e^4
    end
    val, err = hcubature(Aeffb, dimlimits(m)[1], dimlimits(m)[2])
    return ret / val
end

end
