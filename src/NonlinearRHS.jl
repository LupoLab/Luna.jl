"Functions which define the modal decomposition. This includes

    1. Mode normalisation
    2. Modal decomposition of Pₙₗ
    3. Calculation of (modal) energy

Wishlist of types of decomposition we want to use:

Done:
    1. Mode-averaged waveguide
    2. Multi-mode waveguide (with or without polarisation)
        a. Azimuthal symmetry (radial integral only)
        b. Full 2-D integral
To Do:
    3. Free space
        a. Azimuthal symmetry (Hankel transform)
        b. Full 2-D (Fourier transform)"
module NonlinearRHS
import FFTW
import Cubature
import LinearAlgebra: mul!
import NumericalIntegration: integrate, SimpsonEven, Trapezoidal
import Base: show
import Luna: PhysData, Modes, Maths, Grid

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

"Normalisation factor for modal field."
function norm_modal(ω)
    out = -im .* ω ./ 4
    function norm(z)
        return out
    end
end

"Normalisation factor for mode-averaged field."
function norm_mode_average(ω, βfun!, Aeff)
    out = zero(ω)
    pre = @. PhysData.c^(3/2)*sqrt(2*PhysData.ε_0)/ω
    function norm(z)
        βfun!(out, ω, z)
        out .*= pre/sqrt(Aeff(z))
        return out
    end
    return norm
end

function Et_to_Pt!(Pt, Et, responses)
    for resp in responses
        resp(Pt, Et)
    end
end

mutable struct TransModal{tsT, lT, TT, FTT, rT, gT, dT, nT}
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
    normfun::nT
    ncalls::Int
    z::Float64
    rtol::Float64
    atol::Float64
    mfcn::Int
end

function show(io::IO, t::TransModal)
    grid = "grid type: $(typeof(t.grid))"
    nmodes = "no. of modes: $(t.nmodes)"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    full = "full: $(t.full)"
    out = join(["TransModal", nmodes, grid, samples, full, resp], "\n  ")
    print(io, out)
end

"Transform E(ω) -> Pₙₗ(ω) for modal field."
# FT - forward FFT for the grid
# resp - tuple of nonlinear responses
# if full is true, we integrate over whole cross section
function TransModal(tT, grid, ts::Modes.ToSpace, FT, resp, densityfun, normfun; rtol=1e-3, atol=0.0, mfcn=300, full=false)
    Emω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    Erω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Erωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Er = Array{tT,2}(undef, length(grid.to), ts.npol)
    Pr = Array{tT,2}(undef, length(grid.to), ts.npol)
    Prω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Prωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Prmω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    IFT = inv(FT)
    TransModal(ts, full, Modes.dimlimits(ts.ms[1]), Emω, Erω, Erωo, Er, Pr, Prω, Prωo, Prmω,
               FT, resp, grid, densityfun, normfun, 0, 0.0, rtol, atol, mfcn)
end

function TransModal(grid::Grid.RealGrid, args...; kwargs...)
    TransModal(Float64, grid, args...; kwargs...)
end

function TransModal(grid::Grid.EnvGrid, args...; kwargs...)
    TransModal(ComplexF64, grid, args...; kwargs...)
end

@noinline function reset!(t::TransModal, Emω::Array{ComplexF64,2}, z::Float64)
    t.Emω .= Emω
    t.ncalls = 0
    t.z = z
    t.dimlimits = Modes.dimlimits(t.ts.ms[1], z=z)
end

@noinline function pointcalc!(fval, xs, t::TransModal)
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
        fill!(t.Pr, 0.0)
        Et_to_Pt!(t.Pr, t.Er, t.resp)
        @. t.Pr *= t.grid.towin
        to_freq!(t.Prω, t.Prωo, t.Pr, t.FT)
        t.Prω .*= t.grid.ωwin.*t.normfun(t.z)
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
    nl .= t.densityfun(z) .* reshape(reinterpret(ComplexF64, val), size(nl))
end

struct TransModeAvg{TT, FTT, rT, gT, dT, nT, aT}
    Pto::Array{TT,1}
    Eto::Array{TT,1}
    Eωo::Array{ComplexF64,1}
    Pωo::Array{ComplexF64,1}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    normfun::nT
    aeff::aT # function which returns effective area
end

function show(io::IO, t::TransModeAvg)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    out = join(["TransModal", grid, samples, resp], "\n  ")
    print(io, out)
end

function TransModeAvg(TT, grid, FT, resp, densityfun, normfun, aeff)
    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(TT, length(grid.to))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    TransModeAvg(Pto, Eto, Eωo, Pωo, FT, resp, grid, densityfun, normfun, aeff)
end

function TransModeAvg(grid::Grid.RealGrid, FT, resp, densityfun, normfun, aeff)
    TransModeAvg(Float64, grid, FT, resp, densityfun, normfun, aeff)
end

function TransModeAvg(grid::Grid.EnvGrid, FT, resp, densityfun, normfun, aeff)
    TransModeAvg(ComplexF64, grid, FT, resp, densityfun, normfun, aeff)
end

"Transform E(ω) -> Pₙₗ(ω) for mode-averaged field/envelope."
function (t::TransModeAvg)(nl, Eω, z)
    fill!(t.Pto, 0)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT))
    t.Eto ./= sqrt(PhysData.ε_0*PhysData.c*t.aeff(z)/2)
    Et_to_Pt!(t.Pto, t.Eto, t.resp)
    @. t.Pto *= t.grid.towin
    to_freq!(nl, t.Pωo, t.Pto, t.FT)
    nl .*= t.grid.ωwin.*t.densityfun(z).*(-im.*t.grid.ω./2)./t.normfun(z)
end

"Calculate energy from modal field E(t)"
function energy_modal(grid::Grid.RealGrid)
    function energy_t(t, Et)
        Eta = Maths.hilbert(Et)
        return integrate(grid.t, abs2.(Eta), SimpsonEven())
    end

    prefac = 2π/(grid.ω[end]^2)
    function energy_ω(ω, Eω)
        prefac*integrate(ω, abs2.(Eω), SimpsonEven())
    end
    return energy_t, energy_ω
end

function energy_modal(grid::Grid.EnvGrid)
    function energy_t(t, Et)
        return integrate(grid.t, abs2.(Et), SimpsonEven())
    end

    δω = grid.ω[2] - grid.ω[1]
    Δω = length(grid.ω)*δω
    prefac = 2π*δω/(Δω^2)
    function energy_ω(ω, Eω)
        prefac*sum(abs2.(Eω))
    end
    return energy_t, energy_ω
end

end