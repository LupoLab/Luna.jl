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
import Luna: PhysData, Capillary, Maths

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
        for resp in responses
            resp(Pto, Eto)
        end
        @. Pto *= grid.towin
        to_freq!(Pω, Pωo, Pto, FT)
    end

    return Pω!
end

"Transform E(ω) -> Pₙₗ(ω) for mode-averaged field, i.e. only FT and inverse FT."
function trans_env_mode_avg(grid)
    Nto = length(grid.to)
    Nt = length(grid.t)

    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = similar(Eωo)
    Pto = similar(Eto)
    Pωo = similar(Eωo)

    FT = FFTW.plan_fft(Eto, flags=FFTW.PATIENT)
    IFT = FFTW.plan_ifft(Eωo, flags=FFTW.PATIENT)

    function Pω!(Pω, Eω, z, responses)
        fill!(Pto, 0)
        to_time!(Eto, Eω, Eωo, IFT)
        for resp in responses
            resp(Pto, Eto)
        end
        @. Pto *= grid.towin
        to_freq!(Pω, Pωo, Pto, FT)
    end

    return Pω!
end

"Calculate energy from field E(t) for mode-averaged field"
function energy_mode_avg(a)
    Aeff = Capillary.Aeff(a)
    function energyfun(t, Et, m, n)
        Eta = Maths.hilbert(Et)
        intg = abs(integrate(t, abs2.(Eta), SimpsonEven()))
        return intg * PhysData.c*PhysData.ε_0*Aeff/2
    end
    return energyfun
end

"Calculate energy from envelope field E(t) for mode-averaged field"
function energy_env_mode_avg(a)
    Aeff = Capillary.Aeff(a)
    function energyfun(t, Et, m, n)
        intg = abs(integrate(t, abs2.(Et), SimpsonEven()))
        return intg * PhysData.c*PhysData.ε_0*Aeff/2
    end
    return energyfun
end

end