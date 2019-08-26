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
import NumericalIntegration: integrate, SimpsonEven
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
    scale = (No-1)/(N-1) # Scale factor makes up for difference in FFT array length
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
    scale = (N-1)/(No-1) # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale_both!(Aω, Aωo, N÷2, scale)
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

"copy_scale_both! for 2-dim arrays. Works along first axis"
function copy_scale_both!(dest::Array{T,2}, source::Array{T,2}, N, scale) where T
    for i in 1:size(dest,2)
        for j = 1:N
            dest[j,i] = scale * source[j,i]
        end
        for j = 1:N
            dest[end-j+1,i] = scale * source[end-j+1,i]
        end
    end
end

"copy_scale! for multi-dim arrays. Works along first axis"
function copy_scale!(dest, source, N, scale)
    (size(dest)[2:end] == size(source)[2:end] 
     || error("dest and source must be same size except along first dimension"))
    idcs = CartesianIndices(size(dest)[2:end])
    _cpsc_core(dest, source, scale, N, idcs)
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
function norm_mode_average(ω, βfun)
    out = zero(ω)
    function norm(z)
        out .= PhysData.c^2 .* PhysData.ε_0 .* βfun(ω, z) ./ ω
        return out
    end
    return norm
end

function Et_to_Pt!(Pt, Et, responses)
    for resp in responses
        resp(Pt, Et)
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

function pointcalc!(t::TransModal, xs, fval)
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
        val, err = Cubature.pcubature_v(length(Eω)*2, (x, fval) -> pointcalc!(t, x, fval), t.dimlimits[2], t.dimlimits[3], 
                                    reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn,
                                    error_norm=Cubature.L2)
    else
        val, err = Cubature.pcubature_v(length(Eω)*2, (x, fval) -> pointcalc!(t, x, fval),
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

end