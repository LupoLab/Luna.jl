module NonlinearRHS
import FFTW
import Hankel
import Cubature
import Base: show
import LinearAlgebra: mul!, ldiv!
import NumericalIntegration: integrate, SimpsonEven
import Luna: PhysData, Modes, Maths, Grid
import Luna.PhysData: wlfreq

"Transform A(ω) to A(t) on oversampled time grid - real field"
function to_time!(Ato::Array{T, D}, Aω, Aωo, IFTplan) where T<:Real where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (No-1)/(N-1) # Scale factor makes up for difference in FFT array length
    fill!(Aωo, 0)
    copy_scale!(Aωo, Aω, N, scale)
    mul!(Ato, IFTplan, Aωo)
end

"Transform A(ω) to A(t) on oversampled time grid - envelope"
function to_time!(Ato::Array{T, D}, Aω, Aωo, IFTplan) where T<:Complex where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = No/N # Scale factor makes up for difference in FFT array length
    fill!(Aωo, 0)
    copy_scale_both!(Aωo, Aω, N÷2, scale)
    mul!(Ato, IFTplan, Aωo)
end

"Transform oversampled A(t) to A(ω) on normal grid - real field"
function to_freq!(Aω, Aωo, Ato::Array{T, D}, FTplan) where T<:Real where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (N-1)/(No-1) # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale!(Aω, Aωo, N, scale)
end

"Transform oversampled A(t) to A(ω) on normal grid - envelope"
function to_freq!(Aω, Aωo, Ato::Array{T, D}, FTplan) where T<:Complex where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = N/No # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale_both!(Aω, Aωo, N÷2, scale)
end

"Copy first N elements from source to dest and simultaneously multiply by scale factor"
function copy_scale!(dest::Vector, source::Vector, N, scale)
    for i = 1:N
        dest[i] = scale * source[i]
    end
end

"""Copy first and last N elements from source to first and last N elements in dest
and simultaneously multiply by scale factor"""
function copy_scale_both!(dest::Vector, source::Vector, N, scale)
    for i = 1:N
        dest[i] = scale * source[i]
    end
    for i = 1:N
        dest[end-i+1] = scale * source[end-i+1]
    end
end

"copy_scale! for multi-dim arrays. Works along first axis"
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

"copy_scale_both! for multi-dim arrays. Works along first axis"
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


"Accumulate responses induced by Et in Pt"
function Et_to_Pt!(Pt, Ptbuf, Et, responses, density::Number)
    fill!(Pt, 0)
    for resp! in responses
        resp!(Ptbuf, Et)
        Pt .+= density .* Ptbuf
    end
end

function Et_to_Pt!(Pt, Ptbuf, Et, responses, density::AbstractVector)
    fill!(Pt, 0)
    for ii in eachindex(density)
        for resp! in responses[ii]
            resp!(Ptbuf, Et)
            Pt .+= density[ii] .* Ptbuf
        end
    end
end


function Et_to_Pt!(Pt, Ptbuf, Et, responses, density, idcs)
    for i in idcs
        Et_to_Pt!(view(Pt, :, i), Ptbuf, view(Et, :, i), responses, density)
    end
end

mutable struct TransModal{tsT, lT, TT, FTT, rT, gT, dT, ddT, nT}
    ts::tsT
    full::Bool
    dimlimits::lT
    Emω::Array{ComplexF64,2}
    Erω::Array{ComplexF64,2}
    Erωo::Array{ComplexF64,2}
    Er::Array{TT,2}
    Pr::Array{TT,2}
    Prbuf::Array{TT,2}
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

"Transform E(ω) -> Pₙₗ(ω) for modal field."
# FT - forward FFT for the grid
# resp - tuple of nonlinear responses
# if full is true, we integrate over whole cross section
function TransModal(tT, grid, ts::Modes.ToSpace, FT, resp, densityfun, norm!;
                    rtol=1e-3, atol=0.0, mfcn=300, full=false)
    Emω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    Erω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Erωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Er = Array{tT,2}(undef, length(grid.to), ts.npol)
    Pr = Array{tT,2}(undef, length(grid.to), ts.npol)
    Prbuf = similar(Pr)
    Prω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Prωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Prmω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    IFT = inv(FT)
    TransModal(ts, full, Modes.dimlimits(ts.ms[1]), Emω, Erω, Erωo, Er, Pr, Prbuf, Prω, Prωo, Prmω,
               FT, resp, grid, densityfun, densityfun(0.0), norm!, 0, 0.0, rtol, atol, mfcn)
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
        if size(xs, 1) > 1
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
        Modes.to_space!(t.Erω, t.Emω, x, t.ts, z=t.z)
        to_time!(t.Er, t.Erω, t.Erωo, inv(t.FT))
        # get nonlinear pol at r,θ
        Et_to_Pt!(t.Pr, t.Prbuf, t.Er, t.resp, t.density)
        @. t.Pr *= t.grid.towin
        to_freq!(t.Prω, t.Prωo, t.Pr, t.FT)
        @. t.Prω *= t.grid.ωwin
        t.norm!(t.Prω)
        # now project back to each mode
        # matrix product (nω x npol) * (npol x nmodes) -> (nω x nmodes)
        mul!(t.Prmω, t.Prω, transpose(t.ts.Ems))
        fval[:, i] .= pre.*reshape(reinterpret(Float64, t.Prmω), length(t.Emω)*2)
    end
end

function (t::TransModal)(nl, Eω, z)
    reset!(t, Eω, z)
    if t.full
        val, err = Cubature.pcubature_v(
            length(Eω)*2,
            (x, fval) -> pointcalc!(fval, x, t),
            t.dimlimits[2], t.dimlimits[3], 
            reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn, error_norm=Cubature.L2)
    else
        val, err = Cubature.pcubature_v(
            length(Eω)*2,
            (x, fval) -> pointcalc!(fval, x, t),
            (t.dimlimits[2][1],), (t.dimlimits[3][1],), 
            reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn, error_norm=Cubature.L2)
    end
    nl .= reshape(reinterpret(ComplexF64, val), size(nl))
end

function norm_modal(grid; shock=true)
    ω0 = PhysData.wlfreq(grid.referenceλ)
    withshock!(nl) = @. nl *= (-im * grid.ω/4)
    withoutshock!(nl) = @. nl *= (-im * ω0/4)
    shock ? withshock! : withoutshock!
end

struct TransModeAvg{TT, FTT, rT, gT, dT, nT, aT}
    Pto::Vector{TT}
    Ptobuf::Vector{TT}
    Eto::Vector{TT}
    Eωo::Vector{ComplexF64}
    Pωo::Vector{ComplexF64}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    norm!::nT
    aeff::aT # function which returns effective area
end

function show(io::IO, t::TransModeAvg)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    out = join(["TransModeAvg", grid, samples, resp], "\n  ")
    print(io, out)
end

function TransModeAvg(TT, grid, FT, resp, densityfun, norm!, aeff)
    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(TT, length(grid.to))
    Pto = similar(Eto)
    Ptobuf = similar(Eto)
    Pωo = similar(Eωo)
    TransModeAvg(Pto, Ptobuf, Eto, Eωo, Pωo, FT, resp, grid, densityfun, norm!, aeff)
end

function TransModeAvg(grid::Grid.RealGrid, FT, resp, densityfun, norm!, aeff)
    TransModeAvg(Float64, grid, FT, resp, densityfun, norm!, aeff)
end

function TransModeAvg(grid::Grid.EnvGrid, FT, resp, densityfun, norm!, aeff)
    TransModeAvg(ComplexF64, grid, FT, resp, densityfun, norm!, aeff)
end

const nlscale = sqrt(PhysData.ε_0*PhysData.c/2)

"Transform E(ω) -> Pₙₗ(ω) for mode-averaged field/envelope."
function (t::TransModeAvg)(nl, Eω, z)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT))
    @. t.Eto /= nlscale*sqrt(t.aeff(z))
    Et_to_Pt!(t.Pto, t.Ptobuf, t.Eto, t.resp, t.densityfun(z))
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
    pre = @. -im*shockterm/(2*PhysData.c^(3/2)*sqrt(2*PhysData.ε_0))
    function norm!(nl, z)
        βfun!(β, z)
        sqrtaeff = sqrt(aeff(z))
        for i in eachindex(nl)
            !grid.sidx[i] && continue
            nl[i] *= pre[i]*sqrtaeff/β[i]
        end
    end
end

"""
    TransRadial

Transform E(ω) -> Pₙₗ(ω) for radially symetric free-space propagation
"""
struct TransRadial{TT, HTT, FTT, nT, rT, gT, dT, iT}
    QDHT::HTT # Hankel transform (space to k-space)
    FT::FTT # Fourier transform (time to frequency)
    normfun::nT # Function which returns normalisation factor
    resp::rT # nonlinear responses (tuple of callables)
    grid::gT # time grid
    densityfun::dT # callable which returns density
    Pto::Array{TT,2} # Buffer array for NL polarisation on oversampled time grid
    Ptobuf::Vector{TT} # Buffer to be used in Et_to_Pt!
    Eto::Array{TT,2} # Buffer array for field on oversampled time grid
    Eωo::Array{ComplexF64,2} # Buffer array for field on oversampled frequency grid
    Pωo::Array{ComplexF64,2} # Buffer array for NL polarisation on oversampled frequency grid
    idcs::iT # CartesianIndices for Et_to_Pt! to iterate over
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

function TransRadial(TT, grid, HT, FT, responses, densityfun, normfun)
    Eωo = zeros(ComplexF64, (length(grid.ωo), HT.N))
    Eto = zeros(TT, (length(grid.to), HT.N))
    Pto = similar(Eto)
    Ptobuf = zeros(TT, length(grid.to))
    Pωo = similar(Eωo)
    idcs = CartesianIndices(size(Pto)[2:end])
    TransRadial(HT, FT, normfun, responses, grid, densityfun, Pto, Ptobuf, Eto, Eωo, Pωo, idcs)
end

"""
    TransRadial(grid, HT, FT, responses, densityfun, normfun)

Construct a `TransRadial` to calculate the reciprocal-domain nonlinear polarisation.

# Arguments
- `grid::AbstractGrid` : the grid used in the simulation
- `HT::QDHT` : the Hankel transform which defines the spatial grid
- `FT::FFTW.Plan` : the time-frequency Fourier transform for the oversampled time grid
- `responses` : `Tuple` of response functions
- `densityfun` : callable which returns the gas density as a function of `z`
- `normfun` : normalisation factor as fctn of `z`, can be created via [`norm_radial`](@ref)
"""
function TransRadial(grid::Grid.RealGrid, args...)
    TransRadial(Float64, grid, args...)
end

function TransRadial(grid::Grid.EnvGrid, args...)
    TransRadial(ComplexF64, grid, args...)
end

"""
    (t::TransRadial)(nl, Eω, z)

Calculate the reciprocal-domain (ω-k-space) nonlinear response due to the field `Eω` and
place the result in `nl`
"""
function (t::TransRadial)(nl, Eω, z)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT)) # transform ω -> t
    ldiv!(t.Eto, t.QDHT, t.Eto) # transform k -> r
    Et_to_Pt!(t.Pto, t.Ptobuf, t.Eto, t.resp, t.densityfun(z), t.idcs) # add up responses
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

mutable struct TransFree{TT, FTT, nT, rT, gT, xygT, dT, iT}
    FT::FTT # 3D Fourier transform (space to k-space and time to frequency)
    normfun::nT # Function which returns normalisation factor
    resp::rT # nonlinear responses (tuple of callables)
    grid::gT # time grid
    xygrid::xygT
    densityfun::dT # callable which returns density
    Pto::Array{TT, 3} # buffer for oversampled time-domain NL polarisation
    Ptobuf::Array{TT, 1} # buffer for Et_to_Pt!
    Eto::Array{TT, 3} # buffer for oversampled time-domain field
    Eωo::Array{ComplexF64, 3} # buffer for oversampled frequency-domain field
    Pωo::Array{ComplexF64, 3} # buffer for oversampled frequency-domain NL polarisation
    scale::Float64 # scale factor to be applied during oversampling
    idcs::iT # iterating over these slices Eto/Pto into Vectors, one at each position
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

function TransFree(TT, scale, grid, xygrid, FT, responses, densityfun, normfun)
    Ny = length(xygrid.y)
    Nx = length(xygrid.x)
    Eωo = zeros(ComplexF64, (length(grid.ωo), Ny, Nx))
    Eto = zeros(TT, (length(grid.to), Ny, Nx))
    Pto = similar(Eto)
    Ptobuf = zeros(TT, length(grid.to))
    Pωo = similar(Eωo)
    idcs = CartesianIndices((Ny, Nx))
    TransFree(FT, normfun, responses, grid, xygrid, densityfun,
              Pto, Ptobuf, Eto, Eωo, Pωo, scale, idcs)
end

"""
    TransFree(grid, xygrid, FT, responses, densityfun, normfun)

Construct a `TransFree` to calculate the reciprocal-domain nonlinear polarisation.

# Arguments
- `grid::AbstractGrid` : the grid used in the simulation
- `xygrid` : the spatial grid (instances of [`Grid.FreeGrid`](@ref))
- `FT::FFTW.Plan` : the full 3D (t-y-x) Fourier transform for the oversampled time grid
- `responses` : `Tuple` of response functions
- `densityfun` : callable which returns the gas density as a function of `z`
- `normfun` : normalisation factor as fctn of `z`, can be created via [`norm_free`](@ref)
"""
function TransFree(grid::Grid.RealGrid, args...)
    N = length(grid.ω)
    No = length(grid.ωo)
    scale = (No-1)/(N-1)
    TransFree(Float64, scale, grid, args...)
end

function TransFree(grid::Grid.EnvGrid, args...)
    N = length(grid.ω)
    No = length(grid.ωo)
    scale = No/N
    TransFree(ComplexF64, scale, grid, args...)
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
    Et_to_Pt!(t.Pto, t.Ptobuf, t.Eto, t.resp, t.densityfun(z), t.idcs) # add up responses
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