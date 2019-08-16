"Functions which define the modal decomposition. This includes

    1. Mode normalisation
    2. Modal decomposition of Pₙₗ
    3. Calculation of (modal) energy

Wishlist of types of decomposition we want to use:

    1. Mode-averaged waveguide
    2. Multi-mode waveguide (with or without polarisation)
        a. Azimuthal symmetry (radial integral only)
        b. Full 2-D integral
    3. Free space
        a. Azimuthal symmetry (Hankel transform)
        b. Full 2-D (Fourier transform)"
module Modes
import FFTW
import LinearAlgebra: mul!
import NumericalIntegration: integrate, SimpsonEven
import Cubature
import Luna: PhysData, Capillary, Maths

"Transform A(ω) to A(t) on oversampled time grid."
function to_time!(Ato::Array{T, D}, Aω, Aωo, IFTplan) where T<:Real where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (No-1)/(N-1) # Scale factor makes up for difference in FFT array length
    fill!(Aωo, 0)
    copy_scale!(Aωo, Aω, N, scale)
    mul!(Ato, IFTplan, Aωo)
end

"Transform oversampled A(t) to A(ω) on normal grid."
function to_freq!(Aω, Aωo, Ato::Array{T, D}, FTplan) where T<:Real where D
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (N-1)/(No-1) # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale!(Aω, Aωo, N, scale)
end

"Copy first N elements from source to dest and simultaneously multiply by scale factor"
function copy_scale!(dest::Vector, source::Vector, N, scale)
    for i = 1:N
        dest[i] = scale * source[i]
    end
end

"copy_scale! for 2-dim arrays. Works along first axis"
function copy_scale!(dest::Array{T,2}, source::Array{T,2}, N, scale) where T
    for i in 1:size(dest,2)
        for j in 1:N
            dest[j,i] = scale * source[j,i]
        end
    end
end

"copy_scale! for multi-dim arrays. Works along first axis"
function copy_scale!(dest, source, N, scale)
    (size(dest)[2:end] == size(source)[2:end] 
     || error("dest and source must be same size except along first dimension"))
    idcs = CartesianIndices((N, size(dest)[2:end]...))
    _cpsc_core(dest, source, scale, idcs)
end

function _cpsc_core(dest, source, scale, idcs)
    for i in idcs
        dest[i] = scale * source[i]
    end
end

"Normalisation factor for mode-averaged field."
function norm_mode_average(ω, βfun)
    out = zero(ω)
    function norm(z)
        out .= PhysData.c^2 .* PhysData.ε_0 .* βfun(ω, 1, 1, z) ./ ω
        return out
    end
    return norm
end

function Et_to_Pt!(Pt, Et, responses)
    for resp in responses
        resp(Pt, Et)
    end
end

# npol is the number of vector components, either 1 (linear pol) or 2 (full X-Y vec)
mutable struct TransModalRadialMat{ET, FTT, rT, gT, dT}
    nmodes::Int
    full::Bool
    R::Float64
    Ets::ET
    Ems::Array{Float64,2}
    Emω::Array{ComplexF64,2}
    Erω::Array{ComplexF64,2}
    Erωo::Array{ComplexF64,2}
    Er::Array{Float64,2}
    Pr::Array{Float64,2}
    Prω::Array{ComplexF64,2}
    Prωo::Array{ComplexF64,2}
    Prmω::Array{ComplexF64,2}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    ncalls::Int
    rtol::Float64
    atol::Float64
    mfcn::Int
end

"Transform E(ω) -> Pₙₗ(ω) for modal field."
# get this working, then re-write for style/performance
# R - max radial extent
# Ets - function returning matrix nm x npol describing normalised Ex,Ey field given r,θ  
# FT - forward FFT for the grid
# resp - tuple of nonlinear responses
# if full is true, we integrate over whole cross section
function TransModalRadialMat(grid, R, Exy, FT, resp, densityfun; rtol=1e-3, atol=0.0, mfcn=300, full=false)
    nmodes, npol = size(Exy(0.0, 0.0))
    Emω = Array{ComplexF64,2}(undef, length(grid.ω), nmodes)
    Ems = Array{Float64,2}(undef, nmodes, npol)
    Erω = Array{ComplexF64,2}(undef, length(grid.ω), npol)
    Erωo = Array{ComplexF64,2}(undef, length(grid.ωo), npol)
    Er = Array{Float64,2}(undef, length(grid.to), npol)
    Pr = Array{Float64,2}(undef, length(grid.to), npol)
    Prω = Array{ComplexF64,2}(undef, length(grid.ω), npol)
    Prωo = Array{ComplexF64,2}(undef, length(grid.ωo), npol)
    Prmω = Array{ComplexF64,2}(undef, length(grid.ω), nmodes)
    IFT = inv(FT)
    TransModalRadialMat(nmodes, full, R, Exy, Ems, Emω, Erω, Erωo, Er, Pr, Prω, Prωo, Prmω, FT,
                     resp, grid, densityfun, 0, rtol, atol, mfcn)
end

function reset!(t::TransModalRadialMat, Emω::Array{ComplexF64,2})
    t.Emω .= Emω
    t.ncalls = 0
end

function (t::TransModalRadialMat)(xs, fval)
    # TODO: parallelize this in Julia 1.3
    for i in 1:size(xs, 2)
        r = xs[1, i]
        if size(xs, 1) > 1
            θ = xs[2, i]
            pre = r
        else
            θ = 0.0
            pre = 2π*r
        end
        # boundaries r <= 0, r >= R are zero
        if r <= 0.0 || r >=  t.R
            fval[:, i] .= 0.0
            continue
        end
        # get the field at r,θ
        t.Ems = t.Ets(r, θ) # field matrix (nmodes x npol)
        mul!(t.Erω, t.Emω, t.Ems) # matrix product (nω x nmodes) * (nmodes x npol) -> (nω x npol)
        to_time!(t.Er, t.Erω, t.Erωo, inv(t.FT))
        # get nonlinear pol at r,θ
        fill!(t.Pr, 0.0)
        Et_to_Pt!(t.Pr, t.Er, t.resp)
        @. t.Pr *= t.grid.towin
        to_freq!(t.Prω, t.Prωo, t.Pr, t.FT)
        t.Prω .*= t.grid.ωwin.*(-im.*t.grid.ω./4)
        # now project back to each mode
        # matrix product (nω x npol) * (npol x nmodes) -> (nω x nmodes)
        mul!(t.Prmω, t.Prω, transpose(t.Ems))
        fval[:, i] .= pre.*reshape(reinterpret(Float64, t.Prmω), length(t.Emω)*2)
    end
end

function (t::TransModalRadialMat)(nl, Eω, z)
    reset!(t, Eω)
    if t.full
        val, err = Cubature.pcubature_v(length(Eω)*2, (x, fval) -> t(x, fval), (0.0,0.0), (t.R,2π), 
                                    reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn,
                                    error_norm=Cubature.L2)
    else
        val, err = Cubature.pcubature_v(length(Eω)*2, (x, fval) -> t(x, fval), (0.0,), (t.R,), 
                                    reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn,
                                    error_norm=Cubature.L2)
    end
    nl .= t.densityfun(z) .* reshape(reinterpret(ComplexF64, val), size(nl))
end

"Transform E(ω) -> Pₙₗ(ω) for mode-averaged field, i.e. only FT and inverse FT."
function trans_mode_avg(grid)
    Nto = length(grid.to)
    Nt = length(grid.t)

    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(Float64, length(grid.to))
    Pto = similar(Eto)
    Pωo = similar(Eωo)

    FT = FFTW.plan_rfft(Eto, flags=FFTW.PATIENT)
    IFT = FFTW.plan_irfft(Eωo, Nto, flags=FFTW.PATIENT)

    function Pω!(Pω, Eω, z, responses)
        fill!(Pto, 0)
        to_time!(Eto, Eω, Eωo, IFT)
        Et_to_Pt!(Pto, Eto, responses)
        @. Pto *= grid.towin
        to_freq!(Pω, Pωo, Pto, FT)
    end

    return Pω!
end

"Transform E(ω) -> Pₙₗ(ω) for mode-averaged field, i.e. only FT and inverse FT."
function trans_env_mode_avg(grid)
    Nt = length(grid.t)

    Et = zeros(ComplexF64, length(grid.t))
    Pt = similar(Et)

    FT = FFTW.plan_fft(Et, flags=FFTW.PATIENT)
    IFT = FFTW.plan_ifft(Et, flags=FFTW.PATIENT)

    function Pω!(Pω, Eω, z, responses)
        fill!(Pt, 0)
        mul!(Et, IFT, Eω)
        Et_to_Pt!(Pt, Et, responses)
        @. Pt *= grid.twin
        mul!(Pω, FT, Pt)
    end

    return Pω!
end

"Calculate energy from modal field E(t)"
function energy_modal()
    function energyfun(t, Et)
        Eta = Maths.hilbert(Et)
        return abs(integrate(t, abs2.(Eta), SimpsonEven()))
    end
    return energyfun
end

"Calculate energy from field E(t) for mode-averaged field"
function energy_mode_avg(aeff)
    function energyfun(t, Et)
        Eta = Maths.hilbert(Et)
        intg = abs(integrate(t, abs2.(Eta), SimpsonEven()))
        return intg * PhysData.c*PhysData.ε_0*aeff/2
    end
    return energyfun
end

"Calculate energy from envelope field E(t) for mode-averaged field"
function energy_env_mode_avg(aeff)
    function energyfun(t, Et)
        intg = abs(integrate(t, abs2.(Et), SimpsonEven()))
        return intg * PhysData.c*PhysData.ε_0*aeff/2
    end
    return energyfun
end

end