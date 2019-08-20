module AbstractModes
import Roots: fzero
import Luna: Maths
import Luna.PhysData: c, ε_0

abstract type AbstractMode end

"Maximum radius valid for this mode"
function maxR(m::M) where {M <: AbstractMode}
    error("abstract method called")
end

function β(m::M, ω) where {M <: AbstractMode}
    error("abstract method called")
end

function α(m::M, ω) where {M <: AbstractMode}
    error("abstract method called")
end

function dispersion_func(m::M, order) where {M <: AbstractMode}
    βn(ω) = Maths.derivative(ω -> β(m, ω), ω, order)
    return βn
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
    val, err = hcubature(Nfunc, (0.0, 0.0), (maxR(m), 2π))
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
    val, err = hcubature(Aefft, (0.0, 0.0), (maxR(m), 2π))
    ret = val^2
    function Aeffb(x)
        r = x[1]
        θ = x[2]
        e = em(r, θ)
        r*e^4
    end
    val, err = hcubature(Aeffb, (0.0, 0.0), (maxR(m), 2π))
    return ret / val
end

end
