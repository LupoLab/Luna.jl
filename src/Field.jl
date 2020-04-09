module Field
import Luna
import Luna: Grid, Maths, PhysData
import Parameters: @with_kw

abstract type AbstractField end

@with_kw struct GaussField <: AbstractField
    λ0::Float64         # central wavelength
    τfwhm::Float64      # fwhm duration
    energy::Float64     # pulse energy
    ϕ::Float64 = 0.0    # CEO phase
    τ0::Float64 = 0.0   # delay from grid time 0
end

function (p::GaussField)(grid::Grid.RealGrid)
    t = grid.t .- p.τ0
    ω0 = PhysData.wlfreq(p.λ0)
    @. sqrt(Maths.gauss(t, fwhm=p.τfwhm))*cos(ω0*t + p.ϕ)
end

function (p::GaussField)(grid::Grid.EnvGrid)
    t = grid.t .- p.τ0
    Δω = PhysData.wlfreq(p.λ0) - grid.ω0
    @. sqrt(Maths.gauss(t, fwhm=p.τfwhm))*exp(im*(p.ϕ + Δω*t))
end

end
    