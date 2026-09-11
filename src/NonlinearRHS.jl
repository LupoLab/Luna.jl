module NonlinearRHS
import FFTW
import Hankel
import Cubature
import Base: show
import LinearAlgebra: mul!, ldiv!
import NumericalIntegration: integrate, SimpsonEven
import Luna: PhysData, Modes, Maths, Grid
import Luna.PhysData: wlfreq

"""
    to_time!(Ato, Aω, Aωo, IFTplan)

Transform ``A(ω)`` on normal grid to ``A(t)`` on oversampled time grid.
"""
function to_time!(Ato::Array{<:Real, D}, Aω, Aωo, IFTplan) where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (No-1)/(N-1) # Scale factor makes up for difference in FFT array length
    fill!(Aωo, 0)
    copy_scale!(Aωo, Aω, N, scale)
    mul!(Ato, IFTplan, Aωo)
end

function to_time!(Ato::Array{<:Complex, D}, Aω, Aωo, IFTplan) where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = No/N # Scale factor makes up for difference in FFT array length
    fill!(Aωo, 0)
    copy_scale_both!(Aωo, Aω, N÷2, scale)
    mul!(Ato, IFTplan, Aωo)
end

"""
    to_freq!(Aω, Aωo, Ato, FTplan)

Transform oversampled A(t) to A(ω) on normal grid
"""
function to_freq!(Aω, Aωo, Ato::Array{<:Real, D}, FTplan) where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (N-1)/(No-1) # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale!(Aω, Aωo, N, scale)
end

function to_freq!(Aω, Aωo, Ato::Array{<:Complex, D}, FTplan) where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = N/No # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale_both!(Aω, Aωo, N÷2, scale)
end

"""
    copy_scale!(dest, source, N, scale)

Copy first N elements from source to dest and simultaneously multiply by scale factor.
For multi-dimensional `dest` and `source`, work along first axis.
"""
function copy_scale!(dest::Vector, source::Vector, N, scale)
    for i = 1:N
        dest[i] = scale * source[i]
    end
end

"""
    copy_scale_both!(dest::Vector, source::Vector, N, scale)

Copy first and last N elements from source to first and last N elements in dest
and simultaneously multiply by scale factor.
For multi-dimensional `dest` and `source`, work along first axis.
"""
function copy_scale_both!(dest::Vector, source::Vector, N, scale)
    for i = 1:N
        dest[i] = scale * source[i]
    end
    for i = 1:N
        dest[end-i+1] = scale * source[end-i+1]
    end
end

function copy_scale!(dest, source, N, scale)
    (size(dest)[2:end] == size(source)[2:end] 
     || error("dest and source must be same size except along first dimension"))
    idcs = CartesianIndices(size(dest)[2:end])
    _cpsc_core(dest, source, N, scale, idcs)
end

function _cpsc_core(dest, source, N, scale, idcs)
    for i in idcs
        for j = 1:N
            dest[j, i] = scale * source[j, i]
        end
    end
end

function copy_scale_both!(dest, source, N, scale)
    (size(dest)[2:end] == size(source)[2:end] 
     || error("dest and source must be same size except along first dimension"))
    idcs = CartesianIndices(size(dest)[2:end])
    _cpscb_core(dest, source, N, scale, idcs)
end

function _cpscb_core(dest, source, N, scale, idcs)
    for i in idcs
        for j = 1:N
            dest[j, i] = scale * source[j, i]
        end
        for j = 1:N
            dest[end-j+1, i] = scale * source[end-j+1, i]
        end
    end
end

# Note on noise and ionization/plasma: when the modified shot-noise model is active,
# Et_to_Pt! receives the combined field (field + noise). The noise amplitude is of order
# √(ħω·Δν) ≈ 5×10⁻⁴ √W per mode — roughly 10⁻¹⁴ of typical pulse peak power. This is
# completely negligible for the highly nonlinear ionization rate and plasma
# response, so including noise in the field passed to all response functions is physically
# reasonable. The noise meaningfully affects only Kerr and Raman processes, as intended.
"""
    Et_to_Pt!(Pt, Et, responses, density)

Accumulate responses induced by Et in Pt.
"""
function Et_to_Pt!(Pt, Et, responses, density::Number)
    fill!(Pt, 0)
    for resp! in responses
        resp!(Pt, Et, density)
    end
end

function Et_to_Pt!(Pt, Et, responses, density::AbstractVector)
    fill!(Pt, 0)
    for ii in eachindex(density)
        for resp! in responses[ii]
            resp!(Pt, Et, density[ii])
        end
    end
end

function Et_to_Pt!(Pt, Et, responses, density, idcs)
    for i in idcs
        Et_to_Pt!(view(Pt, :, i), view(Et, :, i), responses, density)
    end
end

"""
    ModifiedNoise(Eω_noise, phase, grid)

The noise field of the modified shot-noise model (Chen & Wise, arXiv:2410.20567) together
with the propagator which advances it in `z`.

`Eω_noise` is the one-photon-per-mode spectral draw from
[`Fields.generate_noise_field`](@ref Luna.Fields.generate_noise_field), made once before
propagation and never re-randomised. `phase` is an
[`AbstractUnitaryPhase`](@ref Luna.LinearOps.AbstractUnitaryPhase), created with
[`LinearOps.unitary_phase`](@ref Luna.LinearOps.unitary_phase), which supplies the
accumulated phase `Φ(ω, z) = ∫₀ᶻ imag(linop) dz'`.

The noise is advanced by the **unitary** part of the linear operator only:

- **dispersion — yes.** It is unitary, and it is the phase evolution of the medium's own
  vacuum modes. Without it the model is not related to the traditional (`:input`) model by
  the identity `A_s = A_s' + A_noise` on which it rests, and Kerr/FWM magnitudes are wrong by
  up to ~6× per frequency (much more in conservative propagation).
- **loss — no.** Fluctuation–dissipation: loss replenishes vacuum rather than removing it, so
  the local noise level stays at one photon per mode. This is also what allows the model to
  seed spontaneous generation in absorptive media.
- **gain — no.** The source term already injects fresh vacuum at every `z`, which the
  homogeneous propagator amplifies over the remaining length — that integral *is* the ASE.
  Amplifying the noise as well double-counts it.

Because the field is drawn once and only ever propagated deterministically, results remain
step-size independent, which is what the "constant noise field" requirement of the paper
actually protects.

The draw is multiplied by `grid.ωwin` on construction, matching the windowing the traditional
input shot noise receives from the propagation loop.
"""
struct ModifiedNoise{aT, pT, bT}
    Eω0::aT # z = 0 spectral draw (windowed)
    phase::pT # accumulated unitary phase of the linear operator
    φ::bT # buffer for Φ(ω, z)
    Eω::aT # buffer holding the propagated spectral noise
    lastz::Base.RefValue{Float64}
end

function ModifiedNoise(Eω_noise, phase, grid)
    Eω0 = Eω_noise .* grid.ωwin
    ModifiedNoise(Eω0, phase, zeros(Float64, size(Eω0)), similar(Eω0), Ref(NaN))
end

"""
    noise_at!(noise::ModifiedNoise, z)

Fill and return the propagated spectral noise field at `z`. The result aliases an internal
buffer, so copy it if it needs to outlive the next call.
"""
function noise_at!(n::ModifiedNoise, z)
    if z != n.lastz[]
        n.phase(n.φ, z)
        @. n.Eω = n.Eω0 * cis(n.φ)
        n.lastz[] = z
    end
    n.Eω
end

"""
    noise_at(noise::ModifiedNoise, z)

As [`noise_at!`](@ref) but returns a fresh array.
"""
noise_at(n::ModifiedNoise, z) = copy(noise_at!(n, z))

"""
    TransModal

Transform E(ω) -> Pₙₗ(ω) for multimode propagation via spatial integration.

# Fields
- `noise`: [`ModifiedNoise`](@ref) for the modified shot-noise model, or `nothing`. It is
  advanced to the current `z` once per call, in `reset!`, and then projected to real space at
  each integration point and combined with the field in a separate buffer (`Er_nl`) for
  nonlinear evaluation. The propagating field (`Er`) is never modified.
- `Er_noise`: preallocated buffer for the real-space time-domain noise, same shape as `Er`.
- `Er_nl`: preallocated buffer for the combined field + noise, passed to `Et_to_Pt!`.
"""
mutable struct TransModal{tsT, lT, TT, FTT, rT, gT, dT, ddT, nT, nsT, enT, enlT}
    ts::tsT
    full::Bool
    dimlimits::lT
    Emω::Array{ComplexF64,2}
    Erω::Array{ComplexF64,2}
    Erωo::Array{ComplexF64,2}
    Er::Array{TT,2}
    Pr::Array{TT,2}
    Prω::Array{ComplexF64,2}
    Prωo::Array{ComplexF64,2}
    Prmω::Array{ComplexF64,2}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    density::ddT
    norm!::nT
    ncalls::Int
    z::Float64
    rtol::Float64
    atol::Float64
    mfcn::Int
    err::Array{ComplexF64,2}
    noise::nsT # ModifiedNoise for the modified shot-noise model, or nothing
    Er_noise::enT # buffer for real-space time-domain noise, or nothing
    Er_nl::enlT # buffer for field+noise passed to Et_to_Pt!, or nothing
end

function show(io::IO, t::TransModal)
    grid = "grid type: $(typeof(t.grid))"
    modes = "modes: $(t.ts.nmodes)\n"*" "^4*join([string(mi) for mi in t.ts.ms], "\n    ")
    p = t.ts.indices == 1:2 ? "x,y" : t.ts.indices == 1 ? "x" : "y"
    pol = "polarisation: $p"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    full = "full: $(t.full)"
    out = join(["TransModal", modes, pol, grid, samples, full, resp], "\n  ")
    print(io, out)
end

"""
    TransModal(grid, ts, FT, resp, densityfun, norm!; rtol=1e-3, atol=0.0, mfcn=300, full=false, noise=nothing)

Construct a `TransModal`, transform E(ω) -> Pₙₗ(ω) for modal fields.

# Arguments
- `grid::AbstractGrid` : the grid used in the simulation
- `ts::Modes.ToSpace` : pre-created `ToSpace` for conversion from modal fields to space
- `FT::FFTW.Plan` : the time-frequency Fourier transform for the oversampled time grid
- `resp` : `Tuple` of response functions
- `densityfun` : callable which returns the gas density as a function of `z`
- `norm!` : normalisation function as fctn of `z`, can be created via [`norm_modal`](@ref)
- `rtol::Float=1e-3` : relative tolerance on the `HCubature` integration
- `atol::Float=0.0` : absolute tolerance on the `HCubature` integration
- `mfcn::Int=512` : maximum number of function evaluations for one modal integration
- `full::Bool=false` : if `true`, use full 2-D mode integral, if `false`, only do radial integral
- `noise=nothing` : optional [`ModifiedNoise`](@ref) for the modified shot-noise model, whose
  `(nω, nmodes)` draw holds independent one-photon-per-mode noise in each mode column.
"""
function TransModal(tT, grid, ts::Modes.ToSpace, FT, resp, densityfun, norm!;
                    rtol=1e-3, atol=0.0, mfcn=512, full=false, noise=nothing)
    Emω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    Erω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Erωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Er = Array{tT,2}(undef, length(grid.to), ts.npol)
    Pr = Array{tT,2}(undef, length(grid.to), ts.npol)
    Prω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Prωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Prmω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    IFT = inv(FT)
    # For the modified shot-noise model, allocate a buffer for the modal noise field at the
    # current z and one for the real-space time-domain noise. The noise is projected to space
    # at each integration point in Erω_to_Prω!, so we hold it in the modal domain.
    if !isnothing(noise)
        Er_noise = Array{tT,2}(undef, length(grid.to), ts.npol)
        Er_nl = Array{tT,2}(undef, length(grid.to), ts.npol)
    else
        Er_noise = nothing
        Er_nl = nothing
    end
    TransModal(ts, full, Modes.dimlimits(ts.ms[1]), Emω, Erω, Erωo, Er, Pr, Prω, Prωo, Prmω,
               FT, resp, grid, densityfun, densityfun(0.0), norm!, 0, 0.0, rtol, atol, mfcn,
               similar(Prmω), noise, Er_noise, Er_nl)
end

function TransModal(grid::Grid.RealGrid, args...; kwargs...)
    TransModal(Float64, grid, args...; kwargs...)
end

function TransModal(grid::Grid.EnvGrid, args...; kwargs...)
    TransModal(ComplexF64, grid, args...; kwargs...)
end

function reset!(t::TransModal, Emω::Array{ComplexF64,2}, z::Float64)
    t.Emω .= Emω
    t.ncalls = 0
    t.z = z
    t.dimlimits = Modes.dimlimits(t.ts.ms[1], z=z)
    t.density = t.densityfun(z)
    # Advance the modified shot-noise field to this z. Done once per RHS evaluation rather
    # than at every cubature point, which is why this path costs essentially nothing.
    isnothing(t.noise) || noise_at!(t.noise, z)
end

function pointcalc!(fval, xs, t::TransModal)
    # TODO: parallelize this in Julia 1.3
    for i in 1:size(xs, 2)
        x1 = xs[1, i]
        # on or outside boundaries are zero
        if x1 <= t.dimlimits[2][1] || x1 >= t.dimlimits[3][1]
            fval[:, i] .= 0.0
            continue
        end
        if size(xs, 1) > 1 # full 2-D mode integral
            x2 = xs[2, i]
            if t.dimlimits[1] == :polar
                pre = x1
            else
                if x2 <= t.dimlimits[2][2] || x1 >= t.dimlimits[3][2]
                    fval[:, i] .= 0.0
                    continue
                end
                pre = 1.0
            end
        else
            if t.dimlimits[1] == :polar
                x2 = 0.0
                pre = 2π*x1
            else
                x2 = 0.0
                pre = 1.0
            end
        end
        x = (x1,x2)
        Erω_to_Prω!(t, x)
        t.ncalls += 1
        # now project back to each mode
        # matrix product (nω x npol) * (npol x nmodes) -> (nω x nmodes)
        mul!(t.Prmω, t.Prω, transpose(t.ts.Ems))
        fval[:, i] .= pre.*reshape(reinterpret(Float64, t.Prmω), length(t.Emω)*2)
    end
end

function Erω_to_Prω!(t, x)
    Modes.to_space!(t.Erω, t.Emω, x, t.ts, z=t.z)
    to_time!(t.Er, t.Erω, t.Erωo, inv(t.FT))
    # Modified shot-noise model: project the noise modes (already advanced to t.z by reset!)
    # to real space at this spatial point, convert to the oversampled time domain, and
    # combine with the field in a separate buffer (Er_nl) so the propagating field (Er) is
    # never contaminated.
    if !isnothing(t.noise)
        Modes.to_space!(t.Erω, t.noise.Eω, x, t.ts, z=t.z)
        to_time!(t.Er_noise, t.Erω, t.Erωo, inv(t.FT))
        @. t.Er_nl = t.Er + t.Er_noise
        Et_to_Pt!(t.Pr, t.Er_nl, t.resp, t.density)
    else
        Et_to_Pt!(t.Pr, t.Er, t.resp, t.density)
    end
    @. t.Pr *= t.grid.towin
    to_freq!(t.Prω, t.Prωo, t.Pr, t.FT)
    @. t.Prω *= t.grid.ωwin
    t.norm!(t.Prω)
end

function (t::TransModal)(nl, Eω, z)
    reset!(t, Eω, z)
    _, ll, ul = t.dimlimits
    if t.full
        val, err = Cubature.hcubature_v(
            length(Eω)*2,
            (x, fval) -> pointcalc!(fval, x, t),
            ll, ul, 
            reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn, error_norm=Cubature.L2)
    else
        val, err = Cubature.pcubature_v(
            length(Eω)*2,
            (x, fval) -> pointcalc!(fval, x, t),
            (ll[1],), (ul[1],), 
            reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn, error_norm=Cubature.L2)
    end
    t.err .= reshape(reinterpret(ComplexF64, err), size(nl))
    nl .= reshape(reinterpret(ComplexF64, val), size(nl))
end

"""
    norm_modal(grid; shock=true)

Normalisation function for modal propagation. If `shock` is `false`, the intrinsic frequency
dependence of the nonlinear response is ignored, which turns off optical shock formation/
self-steepening.
"""
function norm_modal(grid; shock=true)
    ω0 = PhysData.wlfreq(grid.referenceλ)
    withshock!(nl) = @. nl *= (-im * grid.ω/4)
    withoutshock!(nl) = @. nl *= (-im * ω0/4)
    shock ? withshock! : withoutshock!
end

"""
    TransModeAvg

Transform E(ω) -> Pₙₗ(ω) for mode-averaged single-mode propagation.

# Fields
- `noise`: [`ModifiedNoise`](@ref) for the modified shot-noise model (Chen & Wise,
  arXiv:2410.20567), or `nothing` for the traditional model.
- `Eωo_noise`/`Et_noise`: buffers holding the noise field at the current `z`, on the
  oversampled frequency and time grids.
- `Et_nl`: preallocated buffer for the combined field + noise. When `noise` is present,
  `Et_nl = Eto + Et_noise` is computed at each step and passed to `Et_to_Pt!`. The
  propagating field (`Eto`) is never modified: the noise feels only the unitary part of the
  linear operator, and never the loss.
"""
struct TransModeAvg{TT, FTT, rT, gT, dT, nT, aT, nsT, eoT, eT, nlT}
    Pto::Vector{TT}
    Eto::Vector{TT}
    Eωo::Vector{ComplexF64}
    Pωo::Vector{ComplexF64}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    norm!::nT
    aeff::aT # function which returns effective area
    noise::nsT # ModifiedNoise for the modified shot-noise model, or nothing
    Eωo_noise::eoT # buffer for the oversampled spectral noise at the current z, or nothing
    Et_noise::eT # buffer for the time-domain noise at the current z, or nothing
    Et_nl::nlT # buffer for field+noise passed to Et_to_Pt!, or nothing
end

function show(io::IO, t::TransModeAvg)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    out = join(["TransModeAvg", grid, samples, resp], "\n  ")
    print(io, out)
end

"""
    TransModeAvg(TT, grid, FT, resp, densityfun, norm!, aeff; noise=nothing)

Construct a `TransModeAvg` transform for mode-averaged propagation.

# Keyword arguments
- `noise=nothing`: optional [`ModifiedNoise`](@ref) for the modified shot-noise model. When
  provided, the noise field is advanced to the current `z`, converted to the oversampled time
  grid, and injected into the nonlinear operator at every propagation step.
"""
function TransModeAvg(TT, grid, FT, resp, densityfun, norm!, aeff; noise=nothing)
    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(TT, length(grid.to))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    if !isnothing(noise)
        Eωo_noise = zeros(ComplexF64, length(grid.ωo))
        Et_noise = zeros(TT, length(grid.to))
        Et_nl = zeros(TT, length(grid.to))
    else
        Eωo_noise = nothing
        Et_noise = nothing
        Et_nl = nothing
    end
    TransModeAvg(Pto, Eto, Eωo, Pωo, FT, resp, grid, densityfun, norm!, aeff,
                 noise, Eωo_noise, Et_noise, Et_nl)
end

function TransModeAvg(grid::Grid.RealGrid, FT, resp, densityfun, norm!, aeff; kwargs...)
    TransModeAvg(Float64, grid, FT, resp, densityfun, norm!, aeff; kwargs...)
end

function TransModeAvg(grid::Grid.EnvGrid, FT, resp, densityfun, norm!, aeff; kwargs...)
    TransModeAvg(ComplexF64, grid, FT, resp, densityfun, norm!, aeff; kwargs...)
end

const nlscale = sqrt(PhysData.ε_0*PhysData.c/2)

function (t::TransModeAvg)(nl, Eω, z)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT))
    sc = nlscale*sqrt(t.aeff(z))
    @. t.Eto /= sc
    # Modified shot-noise model: advance the noise to this z, then compute field+noise in a
    # separate buffer (Et_nl) so that the propagating field (Eto) is never contaminated. The
    # noise is scaled by the same normalisation factor (nlscale × √Aeff) so it enters in
    # physical units.
    if !isnothing(t.noise)
        to_time!(t.Et_noise, noise_at!(t.noise, z), t.Eωo_noise, inv(t.FT))
        @. t.Et_nl = t.Eto + t.Et_noise / sc
        Et_to_Pt!(t.Pto, t.Et_nl, t.resp, t.densityfun(z))
    else
        Et_to_Pt!(t.Pto, t.Eto, t.resp, t.densityfun(z))
    end
    @. t.Pto *= t.grid.towin
    to_freq!(nl, t.Pωo, t.Pto, t.FT)
    t.norm!(nl, z)
    for i in eachindex(nl)
        !t.grid.sidx[i] && continue
        nl[i] *= t.grid.ωwin[i]
    end
end

function norm_mode_average(grid, βfun!, aeff; shock=true)
    β = zeros(Float64, length(grid.ω))
    shockterm = shock ? grid.ω.^2 : grid.ω .* PhysData.wlfreq(grid.referenceλ)
    pre = @. -im*shockterm/4 / nlscale / PhysData.c
    function norm!(nl, z)
        βfun!(β, z)
        sqrtaeff = sqrt(aeff(z))
        for i in eachindex(nl)
            !grid.sidx[i] && continue
            nl[i] *= pre[i]/β[i]*sqrtaeff
        end
    end
end

function norm_mode_average_gnlse(grid, aeff; shock=true)
    shockterm = shock ? grid.ω.^2 : grid.ω .* PhysData.wlfreq(grid.referenceλ)
    pre = @. -im*shockterm/(2*PhysData.c^(3/2)*sqrt(2*PhysData.ε_0))/(grid.ω/PhysData.c)
    function norm!(nl, z)
        sqrtaeff = sqrt(aeff(z))
        for i in eachindex(nl)
            !grid.sidx[i] && continue
            nl[i] *= pre[i]*sqrtaeff
        end
    end
end

"""
    TransRadial

Transform E(ω) -> Pₙₗ(ω) for radially symmetric free-space propagation.

# Fields
- `noise`: [`ModifiedNoise`](@ref) for the modified shot-noise model, or `nothing`.
- `Eωo_noise`/`Et_noise`: buffers holding the noise field at the current `z`, on the
  oversampled frequency/k-space and time/real-space grids `(nto, nr)`.
- `Et_nl`: preallocated buffer for the combined field + noise, passed to `Et_to_Pt!`. The
  propagating field (`Eto`) is never modified.
"""
struct TransRadial{TT, HTT, FTT, nT, rT, gT, dT, iT, nsT, eoT, eT, nlT}
    QDHT::HTT # Hankel transform (space to k-space)
    FT::FTT # Fourier transform (time to frequency)
    normfun::nT # Function which returns normalisation factor
    resp::rT # nonlinear responses (tuple of callables)
    grid::gT # time grid
    densityfun::dT # callable which returns density
    Pto::Array{TT,2} # Buffer array for NL polarisation on oversampled time grid
    Eto::Array{TT,2} # Buffer array for field on oversampled time grid
    Eωo::Array{ComplexF64,2} # Buffer array for field on oversampled frequency grid
    Pωo::Array{ComplexF64,2} # Buffer array for NL polarisation on oversampled frequency grid
    idcs::iT # CartesianIndices for Et_to_Pt! to iterate over
    noise::nsT # ModifiedNoise for the modified shot-noise model, or nothing
    Eωo_noise::eoT # buffer for the oversampled spectral noise at the current z, or nothing
    Et_noise::eT # buffer for the time-domain noise at the current z, or nothing
    Et_nl::nlT # buffer for field+noise passed to Et_to_Pt!, or nothing
end

function show(io::IO, t::TransRadial)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    nr = "radial points: $(t.QDHT.N)"
    R = "aperture: $(t.QDHT.R)"
    out = join(["TransRadial", grid, samples, nr, R, resp], "\n  ")
    print(io, out)
end

"""
    TransRadial(TT, grid, HT, FT, responses, densityfun, normfun; noise=nothing)

Construct a `TransRadial` to calculate the reciprocal-domain nonlinear polarisation.

# Keyword arguments
- `noise=nothing`: optional [`ModifiedNoise`](@ref) for the modified shot-noise model, whose
  draw is an `(nω, nk)` frequency/k-space noise field. At each step it is advanced to the
  current `z` and converted to the real-space time domain `(nto, nr)` via inverse FFT and
  inverse Hankel transform.
"""
function TransRadial(TT, grid, HT, FT, responses, densityfun, normfun; noise=nothing)
    Eωo = zeros(ComplexF64, (length(grid.ωo), HT.N))
    Eto = zeros(TT, (length(grid.to), HT.N))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    idcs = CartesianIndices(size(Pto)[2:end])
    if !isnothing(noise)
        Eωo_noise = zeros(ComplexF64, (length(grid.ωo), HT.N))
        Et_noise = zeros(TT, (length(grid.to), HT.N))
        Et_nl = zeros(TT, (length(grid.to), HT.N))
    else
        Eωo_noise = nothing
        Et_noise = nothing
        Et_nl = nothing
    end
    TransRadial(HT, FT, normfun, responses, grid, densityfun, Pto, Eto, Eωo, Pωo, idcs,
                noise, Eωo_noise, Et_noise, Et_nl)
end

function TransRadial(grid::Grid.RealGrid, args...; kwargs...)
    TransRadial(Float64, grid, args...; kwargs...)
end

function TransRadial(grid::Grid.EnvGrid, args...; kwargs...)
    TransRadial(ComplexF64, grid, args...; kwargs...)
end

"""
    (t::TransRadial)(nl, Eω, z)

Calculate the reciprocal-domain (ω-k-space) nonlinear response due to the field `Eω` and
place the result in `nl`
"""
function (t::TransRadial)(nl, Eω, z)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT)) # transform ω -> t
    ldiv!(t.Eto, t.QDHT, t.Eto) # transform k -> r
    # Modified shot-noise: advance the noise to this z (ω→t, then k→r) and compute
    # field+noise in a separate buffer (Et_nl) so the propagating field (Eto) is never
    # contaminated.
    if !isnothing(t.noise)
        to_time!(t.Et_noise, noise_at!(t.noise, z), t.Eωo_noise, inv(t.FT))
        ldiv!(t.Et_noise, t.QDHT, t.Et_noise)
        @. t.Et_nl = t.Eto + t.Et_noise
        Et_to_Pt!(t.Pto, t.Et_nl, t.resp, t.densityfun(z), t.idcs)
    else
        Et_to_Pt!(t.Pto, t.Eto, t.resp, t.densityfun(z), t.idcs)
    end
    @. t.Pto *= t.grid.towin # apodisation
    mul!(t.Pto, t.QDHT, t.Pto) # transform r -> k
    to_freq!(nl, t.Pωo, t.Pto, t.FT) # transform t -> ω
    nl .*= t.grid.ωwin .* (-im.*t.grid.ω)./(2 .* t.normfun(z))
end

"""
    const_norm_radial(ω, q, nfun)

Make function to return normalisation factor for radial symmetry without re-calculating at
every step. 
"""
function const_norm_radial(grid, q, nfun)
    nfunω = (ω; z) -> nfun(wlfreq(ω))
    normfun = norm_radial(grid, q, nfunω)
    out = copy(normfun(0.0))
    function norm(z)
        return out
    end
    return norm
end

"""
    norm_radial(ω, q, nfun)

Make function to return normalisation factor for radial symmetry. 

!!! note
    Here, `nfun(ω; z)` needs to take frequency `ω` and a keyword argument `z`.
"""
function norm_radial(grid, q, nfun)
    ω = grid.ω
    out = zeros(Float64, (length(ω), q.N))
    kr2 = q.k.^2
    k2 = zeros(Float64, length(ω))
    function norm(z)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z).*grid.ω[grid.sidx]./PhysData.c).^2
        for ir = 1:q.N
            for iω in eachindex(ω)
                if ω[iω] == 0
                    out[iω, ir] = 1.0
                    continue
                end
                βsq = k2[iω] - kr2[ir]
                if βsq <= 0
                    out[iω, ir] = 1.0
                    continue
                end
                out[iω, ir] = sqrt(βsq)/(PhysData.μ_0*ω[iω])
            end
        end
        return out
    end
    return norm
end

"""
    TransFree

Transform E(ω) -> Pₙₗ(ω) for 3D free-space propagation.

# Fields
- `noise`: [`ModifiedNoise`](@ref) for the modified shot-noise model, or `nothing`.
- `Eωo_noise`/`Et_noise`: buffers holding the noise field at the current `z`, on the
  oversampled frequency/k-space and time/real-space grids `(nto, ny, nx)`.
- `Et_nl`: preallocated buffer for the combined field + noise, passed to `Et_to_Pt!`. The
  propagating field (`Eto`) is never modified.
"""
mutable struct TransFree{TT, FTT, nT, rT, gT, xygT, dT, iT, nsT, eoT, eT, nlT}
    FT::FTT # 3D Fourier transform (space to k-space and time to frequency)
    normfun::nT # Function which returns normalisation factor
    resp::rT # nonlinear responses (tuple of callables)
    grid::gT # time grid
    xygrid::xygT
    densityfun::dT # callable which returns density
    Pto::Array{TT, 3} # buffer for oversampled time-domain NL polarisation
    Eto::Array{TT, 3} # buffer for oversampled time-domain field
    Eωo::Array{ComplexF64, 3} # buffer for oversampled frequency-domain field
    Pωo::Array{ComplexF64, 3} # buffer for oversampled frequency-domain NL polarisation
    scale::Float64 # scale factor to be applied during oversampling
    idcs::iT # iterating over these slices Eto/Pto into Vectors, one at each position
    noise::nsT # ModifiedNoise for the modified shot-noise model, or nothing
    Eωo_noise::eoT # buffer for the oversampled spectral noise at the current z, or nothing
    Et_noise::eT # buffer for the time-domain noise at the current z, or nothing
    Et_nl::nlT # buffer for field+noise passed to Et_to_Pt!, or nothing
end

function show(io::IO, t::TransFree)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    y = "y grid: $(minimum(t.xygrid.y)) to $(maximum(t.xygrid.y)), N=$(length(t.xygrid.y))"
    x = "x grid: $(minimum(t.xygrid.x)) to $(maximum(t.xygrid.x)), N=$(length(t.xygrid.x))"
    out = join(["TransFree", grid, samples, y, x, resp], "\n  ")
    print(io, out)
end

"""
    TransFree(TT, scale, grid, xygrid, FT, responses, densityfun, normfun; noise=nothing)

Construct a `TransFree` to calculate the reciprocal-domain nonlinear polarisation for 3D
free-space propagation.

# Keyword arguments
- `noise=nothing`: optional [`ModifiedNoise`](@ref) for the modified shot-noise model, whose
  draw is an `(nω, ny, nx)` frequency/k-space noise field. At each step it is advanced to the
  current `z` and converted to the real-space oversampled time domain `(nto, ny, nx)` via
  `copy_scale!` and a 3D inverse FFT.
"""
function TransFree(TT, scale, grid, xygrid, FT, responses, densityfun, normfun;
                   noise=nothing)
    Ny = length(xygrid.y)
    Nx = length(xygrid.x)
    Eωo = zeros(ComplexF64, (length(grid.ωo), Ny, Nx))
    Eto = zeros(TT, (length(grid.to), Ny, Nx))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    idcs = CartesianIndices((Ny, Nx))
    if !isnothing(noise)
        Eωo_noise = zeros(ComplexF64, (length(grid.ωo), Ny, Nx))
        Et_noise = zeros(TT, (length(grid.to), Ny, Nx))
        Et_nl = zeros(TT, (length(grid.to), Ny, Nx))
    else
        Eωo_noise = nothing
        Et_noise = nothing
        Et_nl = nothing
    end
    TransFree(FT, normfun, responses, grid, xygrid, densityfun,
              Pto, Eto, Eωo, Pωo, scale, idcs, noise, Eωo_noise, Et_noise, Et_nl)
end

function TransFree(grid::Grid.RealGrid, args...; kwargs...)
    N = length(grid.ω)
    No = length(grid.ωo)
    scale = (No-1)/(N-1)
    TransFree(Float64, scale, grid, args...; kwargs...)
end

function TransFree(grid::Grid.EnvGrid, args...; kwargs...)
    N = length(grid.ω)
    No = length(grid.ωo)
    scale = No/N
    TransFree(ComplexF64, scale, grid, args...; kwargs...)
end

"""
    (t::TransFree)(nl, Eω, z)

Calculate the reciprocal-domain (ω-kx-ky-space) nonlinear response due to the field `Eω`
and place the result in `nl`.
"""
function (t::TransFree)(nl, Eωk, z)
    fill!(t.Eωo, 0)
    copy_scale!(t.Eωo, Eωk, length(t.grid.ω), t.scale)
    ldiv!(t.Eto, t.FT, t.Eωo) # transform (ω, ky, kx) -> (t, y, x)
    # Modified shot-noise: advance the noise to this z ((ω,ky,kx) → (t,y,x)) and compute
    # field+noise in a separate buffer (Et_nl) so the propagating field (Eto) is never
    # contaminated.
    if !isnothing(t.noise)
        fill!(t.Eωo_noise, 0)
        copy_scale!(t.Eωo_noise, noise_at!(t.noise, z), length(t.grid.ω), t.scale)
        ldiv!(t.Et_noise, t.FT, t.Eωo_noise)
        @. t.Et_nl = t.Eto + t.Et_noise
        Et_to_Pt!(t.Pto, t.Et_nl, t.resp, t.densityfun(z), t.idcs)
    else
        Et_to_Pt!(t.Pto, t.Eto, t.resp, t.densityfun(z), t.idcs)
    end
    @. t.Pto *= t.grid.towin # apodisation
    mul!(t.Pωo, t.FT, t.Pto) # transform (t, y, x) -> (ω, ky, kx)
    copy_scale!(nl, t.Pωo, length(t.grid.ω), 1/t.scale)
    nl .*= t.grid.ωwin .* (-im.*t.grid.ω)./(2 .* t.normfun(z))
end

"""
    const_norm_free(grid, xygrid, nfun)

Make function to return normalisation factor for 3D propagation without re-calculating at
every step.
"""
function const_norm_free(grid, xygrid, nfun)
    nfunω = (ω; z) -> nfun(wlfreq(ω))
    normfun = norm_free(grid, xygrid, nfunω)
    out = copy(normfun(0.0))
    function norm(z)
        return out
    end
    return norm
end

"""
    norm_free(grid, xygrid, nfun)

Make function to return normalisation factor for 3D propagation.

!!! note
    Here, `nfun(ω; z)` needs to take frequency `ω` and a keyword argument `z`.
"""
function norm_free(grid, xygrid, nfun)
    ω = grid.ω
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    out = zeros(Float64, (length(grid.ω), length(xygrid.ky), length(xygrid.kx)))
    function norm(z)
        k2[grid.sidx] = (nfun.(grid.ω[grid.sidx]; z=z).*grid.ω[grid.sidx]./PhysData.c).^2
        for ii in idcs
            for iω in eachindex(ω)
                if ω[iω] == 0
                    out[iω, ii] = 1.0
                    continue
                end
                βsq = k2[iω] - kperp2[ii]
                if βsq <= 0
                    out[iω, ii] = 1.0
                    continue
                end
                out[iω, ii] = sqrt(βsq)/(PhysData.μ_0*ω[iω])
            end
        end
        return out
    end
end

end