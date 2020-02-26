"Functions which define the modal decomposition. This includes

    1. Mode normalisation
    2. Modal decomposition of Pₙₗ
    3. Calculation of (modal) energy

Types of decomposition that are available:
    1. Mode-averaged waveguide
    2. Multi-mode waveguide (with or without polarisation)
        a. Azimuthal symmetry (radial integral only)
        b. Full 2-D integral
    3. Free space
        a. Azimuthal symmetry (Hankel transform)
        b. Full 2-D (Fourier transform)"

module NonlinearRHS
import FFTW
import Cubature
import LinearAlgebra: mul!, ldiv!
import NumericalIntegration: integrate, SimpsonEven
import Luna: PhysData, Modes, Maths, Grid, Hankel

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
function norm_mode_average(ω, βfun!)
    out = zero(ω)
    function norm(z)
        βfun!(out, ω, z)
        out .*= PhysData.c^2 .* PhysData.ε_0 ./ ω
        return out
    end
    return norm
end

"Accumulate responses induced by Et in Pt"
function Et_to_Pt!(Pt, Et, responses)
    for resp in responses
        resp(Pt, Et)
    end
end

function Et_to_Pt!(Pt, Et, responses, idcs)
    for i in idcs
        Et_to_Pt!(view(Pt, :, i), view(Et, :, i), responses)
    end
end
        
mutable struct TransModal{IT, ET, TT, FTT, rT, gT, dT, nT, lT}
    nmodes::Int
    indices::IT
    dimlimits::lT
    full::Bool
    Exys::ET
    Ems::Array{Float64,2}
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

"Transform E(ω) -> Pₙₗ(ω) for modal field."
# R - max radial extent
# Exys - nmodes length collection of functions returning normalised Ex,Ey field given r,θ  
# FT - forward FFT for the grid
# resp - tuple of nonlinear responses
# if full is true, we integrate over whole cross section
function TransModal(tT, grid, dimlimits, Exys, FT, resp, densityfun, components, normfun; rtol=1e-3, atol=0.0, mfcn=300, full=false)
    # npol is the number of vector components, either 1 (linear pol) or 2 (full X-Y vec)
    if components == :Ey
        indices = 2
        npol = 1
    elseif components == :Ex
        indices = 1
        npol = 1
    elseif components == :Exy
        indices = 1:2
        npol = 2
    else
        error("components must be one of :Ex, :Ey or :Exy")
    end
    nmodes = length(Exys)
    Emω = Array{ComplexF64,2}(undef, length(grid.ω), nmodes)
    Ems = Array{Float64,2}(undef, nmodes, npol)
    Erω = Array{ComplexF64,2}(undef, length(grid.ω), npol)
    Erωo = Array{ComplexF64,2}(undef, length(grid.ωo), npol)
    Er = Array{tT,2}(undef, length(grid.to), npol)
    Pr = Array{tT,2}(undef, length(grid.to), npol)
    Prω = Array{ComplexF64,2}(undef, length(grid.ω), npol)
    Prωo = Array{ComplexF64,2}(undef, length(grid.ωo), npol)
    Prmω = Array{ComplexF64,2}(undef, length(grid.ω), nmodes)
    IFT = inv(FT)
    TransModal(nmodes, indices, dimlimits, full, Exys, Ems, Emω, Erω, Erωo, Er, Pr, Prω, Prωo, Prmω, FT,
                     resp, grid, densityfun, normfun, 0, 0.0, rtol, atol, mfcn)
end

function TransModal(grid::Grid.RealGrid, dimlimits, Exys, FT, resp, densityfun, components, normfun; rtol=1e-3, atol=0.0, mfcn=300, full=false)
    TransModal(Float64, grid, dimlimits, Exys, FT, resp, densityfun, components, normfun, rtol=rtol, atol=atol, mfcn=mfcn, full=full)
end

function TransModal(grid::Grid.EnvGrid, dimlimits, Exys, FT, resp, densityfun, components, normfun; rtol=1e-3, atol=0.0, mfcn=300, full=false)
    TransModal(ComplexF64, grid, dimlimits, Exys, FT, resp, densityfun, components, normfun, rtol=rtol, atol=atol, mfcn=mfcn, full=full)
end

function reset!(t::TransModal, Emω::Array{ComplexF64,2}, z)
    t.Emω .= Emω
    t.ncalls = 0
    t.z = z
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
        # get the field at r,θ
        for i = 1:t.nmodes
            t.Ems[i,:] .= t.Exys[i]((x1, x2))[t.indices] # field matrix (nmodes x npol)
        end
        mul!(t.Erω, t.Emω, t.Ems) # matrix product (nω x nmodes) * (nmodes x npol) -> (nω x npol)
        to_time!(t.Er, t.Erω, t.Erωo, inv(t.FT))
        # get nonlinear pol at r,θ
        fill!(t.Pr, 0.0)
        Et_to_Pt!(t.Pr, t.Er, t.resp)
        @. t.Pr *= t.grid.towin
        to_freq!(t.Prω, t.Prωo, t.Pr, t.FT)
        t.Prω .*= t.grid.ωwin.*t.normfun(t.z)
        # now project back to each mode
        # matrix product (nω x npol) * (npol x nmodes) -> (nω x nmodes)
        mul!(t.Prmω, t.Prω, transpose(t.Ems))
        fval[:, i] .= pre.*reshape(reinterpret(Float64, t.Prmω), length(t.Emω)*2)
    end
end

function (t::TransModal)(nl, Eω, z)
    reset!(t, Eω, z)
    if t.full
        val, err = Cubature.pcubature_v(length(Eω)*2, (x, fval) -> pointcalc!(fval, x, t), t.dimlimits[2], t.dimlimits[3], 
                                    reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn,
                                    error_norm=Cubature.L2)
    else
        val, err = Cubature.pcubature_v(length(Eω)*2, (x, fval) -> pointcalc!(fval, x, t),
                                    (t.dimlimits[2][1],), (t.dimlimits[3][1],), 
                                    reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn,
                                    error_norm=Cubature.L2)
    end
    nl .= t.densityfun(z) .* reshape(reinterpret(ComplexF64, val), size(nl))
end

struct TransModeAvg{TT, FTT, rT, gT, dT, nT}
    Pto::Array{TT,1}
    Eto::Array{TT,1}
    Eωo::Array{ComplexF64,1}
    Pωo::Array{ComplexF64,1}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    normfun::nT
end

function TransModeAvg(TT, grid, FT, resp, densityfun, normfun)
    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(TT, length(grid.to))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    TransModeAvg(Pto, Eto, Eωo, Pωo, FT, resp, grid, densityfun, normfun)
end

function TransModeAvg(grid::Grid.RealGrid, FT, resp, densityfun, normfun)
    TransModeAvg(Float64, grid, FT, resp, densityfun, normfun)
end

function TransModeAvg(grid::Grid.EnvGrid, FT, resp, densityfun, normfun)
    TransModeAvg(ComplexF64, grid, FT, resp, densityfun, normfun)
end

"Transform E(ω) -> Pₙₗ(ω) for mode-averaged field/envelope."
function (t::TransModeAvg)(nl, Eω, z)
    fill!(t.Pto, 0)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT))
    Et_to_Pt!(t.Pto, t.Eto, t.resp)
    @. t.Pto *= t.grid.towin
    to_freq!(nl, t.Pωo, t.Pto, t.FT)
    nl .*= t.grid.ωwin.*t.densityfun(z).*(-im.*t.grid.ω./2)./t.normfun(z)
end

"Calculate energy from modal field E(t)"
function energy_modal()
    function energyfun(t, Et)
        Eta = Maths.hilbert(Et)
        return abs(integrate(t, abs2.(Eta), SimpsonEven()))
    end
    return energyfun
end

"Calculate energy from modal envelope field E(t)"
function energy_env_modal()
    function energyfun(t, Et)
        return abs(integrate(t, abs2.(Et), SimpsonEven()))
    end
    return energyfun
end

"Calculate energy from field E(t) for mode-averaged field"
function energy_mode_avg(m)
    aeff = Modes.Aeff(m)
    function energyfun(t, Et)
        Eta = Maths.hilbert(Et)
        intg = abs(integrate(t, abs2.(Eta), SimpsonEven()))
        return intg * PhysData.c*PhysData.ε_0*aeff/2
    end
    return energyfun
end

"Calculate energy from envelope field E(t) for mode-averaged field"
function energy_env_mode_avg(m)
    aeff = Modes.Aeff(m)
    function energyfun(t, Et)
        intg = abs(integrate(t, abs2.(Et), SimpsonEven()))
        return intg * PhysData.c*PhysData.ε_0*aeff/2
    end
    return energyfun
end

"Transform E(ω) -> Pₙₗ(ω) for radially symetric free-space propagation"
struct TransRadial{TT, HTT, FTT, nT, rT, gT, dT, iT}
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
end

function TransRadial(TT, grid, HT, FT, responses, densityfun, normfun)
    Eωo = zeros(ComplexF64, (length(grid.ωo), HT.N))
    Eto = zeros(TT, (length(grid.to), HT.N))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    idcs = CartesianIndices(size(Pto)[2:end])
    TransRadial(HT, FT, normfun, responses, grid, densityfun, Pto, Eto, Eωo, Pωo, idcs)
end

function TransRadial(grid::Grid.RealGrid, args...)
    TransRadial(Float64, grid, args...)
end

function TransRadial(grid::Grid.EnvGrid, args...)
    TransRadial(ComplexF64, grid, args...)
end

function (t::TransRadial)(nl, Eω, z)
    fill!(t.Pto, 0)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT)) # transform ω -> t
    ldiv!(t.Eto, t.QDHT, t.Eto) # transform k -> r
    Et_to_Pt!(t.Pto, t.Eto, t.resp, t.idcs) # add up responses
    @. t.Pto *= t.grid.towin # apodisation
    mul!(t.Pto, t.QDHT, t.Pto) # transform r -> k
    to_freq!(nl, t.Pωo, t.Pto, t.FT) # transform t -> ω
    nl .*= t.grid.ωwin .* t.densityfun(z) .* (-im.*t.grid.ω)./(2 .* t.normfun(z))
end

"Normalisation factor for radial symmetry"
function norm_radial(ω, q, nfun)
    βsq = @. (nfun(2π*PhysData.c/ω)*ω/PhysData.c)^2 - (q.k^2)'
    βsq[βsq .< 0] .= 0
    out = @. sqrt(βsq)/(PhysData.μ_0*ω)
    out[ω .== 0, :] .= 1
    out[out .== 0] .= 1
    function norm(z)
        return out
    end
    return norm
end

function energy_radial(q)
    function energyfun(t, Et)
        Eta = Maths.hilbert(Et)
        intg = abs.(integrate(t, abs2.(Eta), SimpsonEven()))
        return 2π*PhysData.c*PhysData.ε_0/2 * Hankel.integrateR(intg, q)
    end
end

function energy_radial_env(q)
    function energyfun(t, Et)
        intg = abs.(integrate(t, abs2.(Et), SimpsonEven()))
        return 2π*PhysData.c*PhysData.ε_0/2 * Hankel.integrateR(intg, q)
    end
end

"Transform E(ω) -> Pₙₗ(ω) for full 3D free-space propagation"
mutable struct TransFree{TT, FTT, nT, rT, gT, dT, iT}
    FT::FTT # 3D Fourier transform (space to k-space and time to frequency)
    normfun::nT # Function which returns normalisation factor
    resp::rT # nonlinear responses (tuple of callables)
    grid::gT # time grid
    densityfun::dT # callable which returns density
    Pto::Array{TT, 3}
    Eto::Array{TT, 3}
    Eωo::Array{ComplexF64, 3}
    Pωo::Array{ComplexF64, 3}
    scale::Float64
    idcs::iT
end

function TransFree(TT, grid, FT, Ny, Nx, responses, densityfun, normfun)
    Eωo = zeros(ComplexF64, (length(grid.ωo), Ny, Nx))
    Eto = zeros(TT, (length(grid.to), Ny, Nx))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    N = length(grid.ω)
    No = length(grid.ωo)
    scale = (No-1)/(N-1)
    idcs = CartesianIndices((Ny, Nx))
    TransFree(FT, normfun, responses, grid, densityfun, Pto, Eto, Eωo, Pωo, scale, idcs)
end

function TransFree(grid::Grid.RealGrid, args...)
    TransFree(Float64, grid, args...)
end

function TransFree(grid::Grid.EnvGrid, args...)
    TransFree(ComplexF64, grid, args...)
end

function (t::TransFree)(nl, Eωk, z)
    fill!(t.Pto, 0)
    fill!(t.Eωo, 0)
    copy_scale!(t.Eωo, Eωk, length(t.grid.ω), t.scale)
    ldiv!(t.Eto, t.FT, t.Eωo) # transform (ω, ky, kx) -> (t, y, x)
    Et_to_Pt!(t.Pto, t.Eto, t.resp, t.idcs) # add up responses
    @. t.Pto *= t.grid.towin # apodisation
    mul!(t.Pωo, t.FT, t.Pto) # transform (t, y, x) -> (ω, ky, kx)
    copy_scale!(nl, t.Pωo, length(t.grid.ω), 1/t.scale)
    nl .*= t.grid.ωwin .* t.densityfun(z) .* (-im.*t.grid.ω)./(2 .* t.normfun(z))
end

"Normalisation factor for 3D propagation"
function norm_free(ω, x, y, nfun)
    kx = reshape(Maths.fftfreq(x), (1, 1, length(x)))
    ky = reshape(Maths.fftfreq(x), (1, length(y)))
    n = nfun.(2π*PhysData.c./ω)
    βsq = @. (n*ω/PhysData.c)^2 - kx^2 - ky^2
    βsq = FFTW.fftshift(βsq, (2, 3))
    βsq[βsq .< 0] .= 0
    out = @. sqrt.(βsq)/(PhysData.μ_0*ω)
    out[ω .== 0, :, :] .= 1
    out[out .== 0] .= 1
    function norm(z)
        return out
    end
    return norm
end

function energy_free(x, y)
    Dx = abs(x[2] - x[1])
    Dy = abs(y[2] - y[1])
    function energyfun(t, Et)
        Eta = Maths.hilbert(Et)
        intg = sum(abs2.(Eta)) * Dx * Dy * abs(t[2] - t[1])
        return PhysData.c*PhysData.ε_0/2 *intg
    end
end

function energy_free_env(x, y)
    Dx = abs(x[2] - x[1])
    Dy = abs(y[2] - y[1])
    function energyfun(t, Et)
        intg = sum(abs2.(Et)) * Dx * Dy * abs(t[2] - t[1])
        return PhysData.c*PhysData.ε_0/2 *intg
    end
end

end