module Stats
import Luna: Maths, Grid, Modes, Utils
import Luna.PhysData: wlfreq, c, ε_0
import FFTW
import LinearAlgebra: mul!
import Printf: @sprintf

function ω0(grid)
    addstat! = let ω=grid.ω
        function addstat!(d, Eω, Et, z, dz)
            d["ω0"] = Maths.moment(ω, abs2.(Eω))
        end
    end
    return addstat!
end

function energy(grid, energyfun_ω)
    function addstat!(d, Eω, Et, z, dz)
        d["energy"] = energyfun_ω(grid.ω, Eω)
    end
    return addstat!
end

function energy(grid, energyfun_ω, N)
    function addstat!(d, Eω, Et, z, dz)
        d["energy"] = [energyfun_ω(grid.ω, Eω[:, i]) for i=1:N]
    end
    return addstat!
end

function energy_λ(grid, energyfun_ω, λlims; label=nothing, winwidth=0)
    λlims = collect(λlims)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    if isnothing(label)
        λnm = 1e9.*λlims
        label = @sprintf("%.2fnm_%.2fnm", minimum(λnm), maximum(λnm))
    end
    energy_window(grid, energyfun_ω, window; label=label)
end

function energy_window(grid, energyfun_ω, window; label)
    key = "energy_$label"
    function addstat!(d, Eω, Et, z, dz)
        d[key] = energyfun_ω(grid.ω, Eω.*window)
    end
    return addstat!
end

function peakpower(grid)
    function addstat!(d, Eω, Et, z, dz)
        d["peakpower"] = maximum(abs2.(Et))
    end
    return addstat!
end

function peakpower(grid, N)
    function addstat!(d, Eω, Et, z, dz)
        d["peakpower"] = [maximum(abs2.(Et[:, i])) for i=1:N]
    end
    return addstat!
end

function fwhm_t(grid)
    function addstat!(d, Eω, Et, z, dz)
        d["fwhm_t_min"] = Maths.fwhm(grid.t, abs2.(Et), :nearest, minmax=:min)
        d["fwhm_t_max"] = Maths.fwhm(grid.t, abs2.(Et), :nearest, minmax=:max)
    end
end

function fwhm_t(grid, N)
    function addstat!(d, Eω, Et, z, dz)
        d["fwhm_t"] = [Maths.fwhm(grid.t, abs2.(Et[:, i]), :nearest) for i=1:N]
    end
end

"""
    electrondensity(grid, ionrate, dfun, aeff; oversampling=1)

Create stats function to calculate the maximum electron density.

If oversampling > 1, the field is oversampled before the calculation
!!! warning
    Oversampling can lead to a significant performance hit
"""
function electrondensity(grid::Grid.RealGrid, ionrate!, dfun, aeff; oversampling=1)
    to, Eto = Maths.oversample(grid.t, complex(grid.t), factor=oversampling)
    δt = to[2] - to[1]
    function ionfrac!(out, Et)
        ionrate!(out, Et)
        Maths.cumtrapz!(out, δt) # in-place cumulative integration
        @. out = 1 - exp(-out)
    end
    frac = similar(to)
    function addstat!(d, Eω, Et, z, dz)
        # note: oversampling returns its arguments without any work done if factor=1
        to, Eto = Maths.oversample(grid.t, Et, factor=oversampling)
        ionfrac!(frac, real(Eto)/sqrt(ε_0*c*aeff(z)/2))
        d["electrondensity"] = maximum(frac)*dfun(z)
    end
end

function density(dfun)
    function addstat!(d, Eω, Et, z, dz)
        d["density"] = dfun(z)
    end
end

function zdz!(d, Eω, Et, z, dz)
    d["z"] = z
    d["dz"] = dz
end

"""
    plan_analytic(grid, Eω)

Plan a transform from the frequency-domain field `Eω` to the analytic time-domain field.

Returns both a buffer for the analytic field and a closure to do the transform.
"""
function plan_analytic(grid::Grid.EnvGrid, Eω)
    Eta = similar(Eω)
    Utils.loadFFTwisdom()
    iFT = FFTW.plan_ifft(Eω, 1, flags=FFTW.PATIENT)
    Utils.saveFFTwisdom()
    function analytic!(Eta, Eω)
        mul!(Eta, iFT, Eω) # for envelope fields, we only need to do the inverse transform
    end
    return Eta, analytic!
end

function plan_analytic(grid::Grid.RealGrid, Eω)
    s = collect(size(Eω))
    s[1] = (length(grid.ω) - 1)*2 # e.g. for 4097 rFFT samples, we need 8192 FFT samples
    Eta = Array{ComplexF64, ndims(Eω)}(undef, Tuple(s))
    Eωa = zero(Eta)
    idxhi = CartesianIndices(size(Eω)[2:end]) # index over all other dimensions
    Utils.loadFFTwisdom()
    iFT = FFTW.plan_ifft(Eωa, 1, flags=FFTW.PATIENT)
    Utils.saveFFTwisdom()
    function analytic!(Eta, Eω)
        copyto_fft!(Eωa, Eω, idxhi) # copy across to FFT-sampled buffer
        mul!(Eta, iFT, Eωa) # now do the inverse transform
    end
    return Eta, analytic!
end

"""
    copyto_fft!(Eωa, Eω, idxhi)

Copy the rFFT-sampled field `Eω` to the FFT-sampled buffer `Eωa`, ready for inverse FFT
"""
function copyto_fft!(Eωa, Eω, idxhi)
    n = size(Eω, 1)-1 # rFFT has sample at +fs/2, but FFT does not (only at -fs/2)
    for idx in idxhi
        for i in 1:n
            Eωa[i, idx] = 2*Eω[i, idx]
        end
    end
end

"""
    collect_stats(grid, Eω, funcs...)

Create a closure which collects statistics from the individual functions in `funcs`.

Each function given will be called with the arguments `(d, Eω, Et, z, dz)`, where
- d -> dictionary to store statistics values. each `func` should **mutate** this
- Eω -> frequency-domain field
- Et -> analytic time-domain field
- z -> current propagation distance
- dz -> current stepsize
"""
function collect_stats(grid, Eω, funcs...)
    # make sure z and dz are recorded
    if !(zdz! in funcs)
        funcs = (funcs..., zdz!)
    end
    Et, analytic! = plan_analytic(grid, Eω)
    f = let funcs=funcs
        function collect_stats(Eω, z, dz)
            d = Dict{String, Any}()
            analytic!(Et, Eω)
            for func in funcs
                func(d, Eω, Et, z, dz)
            end
            return d
        end
    end
    return f
end

end