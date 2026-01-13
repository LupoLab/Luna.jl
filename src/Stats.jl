module Stats
import Luna: Maths, Grid, Modes, Utils, settings, PhysData, Fields, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.NonlinearRHS: TransModal, TransModeAvg, Erω_to_Prω!
import Luna.Nonlinear: PlasmaCumtrapz
import Luna.Capillary: MarcatiliMode
import FFTW
import LinearAlgebra: mul!
import Printf: @sprintf
import Logging: @warn

"""
    ω0(grid)

Create stats function to calculate the centre of mass (first moment) of the spectral power
density.
"""
function ω0(grid)
    addstat! = let ω=grid.ω
        function addstat!(d, Eω, Et, z, dz)
            d["ω0"] = squeeze(Maths.moment(ω, abs2.(Eω); dim=1))
        end
    end
    return addstat!
end

squeeze(ω0::Array{T, 1}) where T = ω0[1]
squeeze(ω0::Array{T, 2}) where T = ω0[1, :]

"""
    energy(grid, energyfun_ω)

Create stats function to calculate the total energy.
"""
function energy(grid, energyfun_ω)
    function addstat!(d, Eω, Et, z, dz)
        if ndims(Eω) > 1
            d["energy"] = [energyfun_ω(Eω[:, i]) for i=1:size(Eω, 2)]
        else
            d["energy"] = energyfun_ω(Eω)
        end
    end
    return addstat!
end

"""
    energy_λ(grid, energyfun_ω, λlims; label)

Create stats function to calculate the energy in a wavelength region given by `λlims`.
If `label` is omitted, the stats dataset is named by the wavelength limits.
"""
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

"""
    energy_window(grid, energyfun_ω, window; label)

Create stats function to calculate the energy filtered by a `window`. The stats dataset will
be named `energy_[label]`.
"""
function energy_window(grid, energyfun_ω, window; label)
    key = "energy_$label"
    function addstat!(d, Eω, Et, z, dz)
        if ndims(Eω) > 1
            d[key] = [energyfun_ω(Eω[:, i].*window) for i=1:size(Eω, 2)]
        else
            d[key] = energyfun_ω(Eω.*window)
        end
    end
    return addstat!
end

"""
    peakpower(grid)

Create stats function to calculate the peak power.
"""
function peakpower(grid)
    function addstat!(d, Eω, Et, z, dz)
        if ndims(Et) > 1
            d["peakpower"] = dropdims(maximum(abs2.(Et), dims=1), dims=1)
            d["peakpower_allmodes"] = maximum(eachindex(grid.t)) do ii
                sum(abs2, Et[ii, :])
            end
        else
            d["peakpower"] = maximum(abs2, Et)
        end
    end
    return addstat!
end

"""
    peakpower(grid, Eω, window; label)

Create stats function to calculate the peak power within a frequency range defined by the
window function `window`. `window` must have the same length as `grid.ω`. The stats
dataset is labeled as `peakpower_[label]`.
"""
function peakpower(grid, Eω, window::Vector{<:Real}; label)
    Etbuf, analytic! = plan_analytic(grid, Eω) # output buffer and function for inverse FT
    Eωbuf = similar(Eω) # buffer for Eω with window applied
    Pt = zeros((length(grid.t), size(Eωbuf, 2)))
    key = "peakpower_$label"
    function addstat!(d, Eω, Et, z, dz)
        Eωbuf .= Eω .* window
        analytic!(Etbuf, Eωbuf)
        if ndims(Etbuf) > 1
            Pt .= abs2.(Etbuf)
            d[key] = dropdims(maximum(Pt, dims=1), dims=1)
            d[key*"_allmodes"] = maximum(eachindex(grid.t)) do ii
                sum(Pt[ii, :]; dims=2)
            end
        else
            d[key] = maximum(abs2, Etbuf)
        end
    end
end

"""
    peakpower(grid, Eω, λlims; label=nothing)

Create stats function to calculate the peak power within a frequency range defined by the
wavelength limits `λlims`. If `label` is given, the stats dataset is labeled as
`peakpower_[label]`, otherwise `label` is created automatically from `λlims`.
"""
function peakpower(grid, Eω, λlims::NTuple{2, <:Real}; label=nothing, winwidth=:auto)
    window = Processing.ωwindow_λ(grid.ω, λlims; winwidth=winwidth)
    if isnothing(label)
        λnm = 1e9.*λlims
        label = @sprintf("%.2fnm_%.2fnm", minimum(λnm), maximum(λnm))
    end
    peakpower(grid, Eω, window; label=label)
end


"""
    peakintensity(grid, aeff)

Create stats function to calculate the mode-averaged peak intensity given the effective area
`aeff(z)`.
"""
function peakintensity(grid, aeff)
    function addstat!(d, Eω, Et, z, dz)
        d["peakintensity"] = maximum(abs2, Et)/aeff(z)
    end
end

"""
    peakintensity(grid, mode)

Create stats function to calculate the peak intensity for several modes.
"""
function peakintensity(grid, modes::Modes.ModeCollection; components=:y)
    tospace = Modes.ToSpace(modes, components=components)
    npol = tospace.npol
    Et0 = zeros(ComplexF64, (length(grid.t), npol))
    function addstat!(d, Eω, Et, z, dz)
        Modes.to_space!(Et0, Et, (0, 0), tospace; z=z)
        if npol > 1
            d["peakintensity"] = c*ε_0/2 * maximum(eachindex(grid.t)) do ii
                sum(abs2, Et0[ii, :])
            end
        else
            d["peakintensity"] = c*ε_0/2 * maximum(abs2, Et0)
        end
    end
end

"""
    fwhm_t(grid)

Create stats function to calculate the temporal FWHM (pulse duration) for mode average.
"""
function fwhm_t(grid)
    function addstat!(d, Eω, Et, z, dz)
        Pt = abs2.(Et)
        if ndims(Et) > 1
            Ptsum = dropdims(sum(Pt; dims=2); dims=2)
            d["fwhm_t_min"] = [Maths.fwhm(grid.t, Pt[:, i], method=:linear)
                              for i=1:size(Et, 2)]
            d["fwhm_t_max"] = [Maths.fwhm(grid.t, Pt[:, i], method=:linear, minmax=:max)
                              for i=1:size(Et, 2)]
            d["fwhm_t_min_allmodes"] = Maths.fwhm(grid.t, Ptsum, method=:linear)
            d["fwhm_t_max_allmodes"] = Maths.fwhm(grid.t, Ptsum, method=:linear, minmax=:max)
        else
            d["fwhm_t_min"] = Maths.fwhm(grid.t, Pt, method=:linear, minmax=:min)
            d["fwhm_t_max"] = Maths.fwhm(grid.t, Pt, method=:linear, minmax=:max)
        end
    end
end

"""
    fwhm_r(grid, modes; components=:y)

Create stats function to calculate the radial FWHM (aka beam size) in a modal propagation.
"""
function fwhm_r(grid, modes; components=:y)
    tospace = Modes.ToSpace(modes, components=components)
    npol = tospace.npol
    Eω0 = zeros(ComplexF64, (length(grid.ω), npol))

    function addstat!(d, Eω, Et, z, dz)
        function f(r)
            Modes.to_space!(Eω0, Eω, (r, 0), tospace; z=z)
            sum(abs2, Eω0)
        end
        d["fwhm_r"] = 2*Maths.hwhm(f)
    end
end

"""
    electrondensity(grid, ionrate, dfun, aeff; oversampling=1)

Create stats function to calculate the maximum electron density in mode average.

If oversampling > 1, the field is oversampled before the calculation
!!! warning
    Oversampling can lead to a significant performance hit
"""
function electrondensity(grid::Grid.RealGrid, ionrate!, dfun, aeff; oversampling=1)
    to, Eto = Maths.oversample(grid.t, complex(grid.t), factor=oversampling)
    δt = to[2] - to[1]
    # ionfrac! stores the time-dependent ionisation fraction in out and returns the max
    # ionisation rate
    function ionfrac!(out, Et)
        ionrate!(out, Et)
        ratemax = maximum(out)
        Maths.cumtrapz!(out, δt) # in-place cumulative integration
        @. out = 1 - exp(-out)
        return ratemax
    end
    frac = similar(to)
    function addstat!(d, Eω, Et, z, dz)
        # note: oversampling returns its arguments without any work done if factor==1
        to, Eto = Maths.oversample(grid.t, Et, factor=oversampling)
        @. Eto /= sqrt(ε_0*c*aeff(z)/2)
        ratemax = ionfrac!(frac, real(Eto))
        d["electrondensity"] = frac[end]*dfun(z)
        d["peak_ionisation_rate"] = ratemax
    end
end

"""
    electrondensity(grid, ionrate, dfun, modes; oversampling=1)

Create stats function to calculate the maximum electron density for multimode simulations.

If oversampling > 1, the field is oversampled before the calculation
!!! warning
    Oversampling can lead to a significant performance hit
"""
function electrondensity(grid::Grid.RealGrid, ionrate!, dfun,
                         modes::Modes.ModeCollection;
                         components=:y, oversampling=1)
    to, Eto = Maths.oversample(grid.t, complex(grid.t), factor=oversampling)
    δt = to[2] - to[1]
    # ionfrac! stores the time-dependent ionisation fraction in out and returns the max
    # ionisation rate
    function ionfrac!(out, Et)
        ionrate!(out, Et)
        ratemax = maximum(out)
        Maths.cumtrapz!(out, δt) # in-place cumulative integration
        @. out = 1 - exp(-out)
        return ratemax
    end
    tospace = Modes.ToSpace(modes, components=components)
    frac = similar(to)
    npol = tospace.npol
    Et0 = zeros(ComplexF64, (length(to), npol))
    function addstat!(d, Eω, Et, z, dz)
        # note: oversampling returns its arguments without any work done if factor==1
        to, Eto = Maths.oversample(grid.t, Et, factor=oversampling)
        Modes.to_space!(Et0, Eto, (0, 0), tospace; z=z)
        if npol > 1
            ratemax = ionfrac!(frac, hypot.(real(Et0[:, 1]), real(Et0[:, 2])))
        else
            ratemax = ionfrac!(frac, real(Et0[:, 1]))
        end
        d["electrondensity"] = frac[end]*dfun(z)
        d["peak_ionisation_rate"] = ratemax
    end
end

"""
    mode_reconstruction_error(t::TransModal)

Create a stats function to calculate and collect the mode reconstruction error in the
induced polarisation on axis at every step.
"""
function mode_reconstruction_error(t::TransModal)
    Prω_recon = similar(t.Prω)
    difference = similar(Prω_recon)
    nl = similar(t.Emω)
    function addstat!(d, Eω, Et, z, dz)
        t(nl, Eω, z)
        x = (0.0, 0.0) # on-axis coordinate
        # reconstruct 
        Modes.to_space!(Prω_recon, nl, x, t.ts, z=z)
        # in going to modes and back we've picked up two factors of the mode normalisation
        Prω_recon .*= 1/2*sqrt(PhysData.ε_0/PhysData.μ_0)
        Erω_to_Prω!(t, x)
        difference .= Prω_recon .- t.Prω
        d["mode_reconstruction_error"] = sqrt(sum(abs2, difference))/sqrt(sum(abs2, Prω_recon))
        d["transverse_points"] = float(t.ncalls) # convert to Float64 to enable NaN padding
        d["transverse_integral_error_abs"] = sqrt(sum(abs2, t.err)/length(t.err))
        d["transverse_integral_error_rel"] = d["transverse_integral_error_abs"]/sqrt(sum(abs2, nl)/length(nl))
    end
end

"""
    density(dfun)

Create stats function to capture the gas density as defined by `dfun(z)`
"""
function density(dfun)
    function addstat!(d, Eω, Et, z, dz)
        d["density"] = dfun(z)
    end
end

"""
    pressure(dfun, gas)

Create stats function to capture the pressure. Like [`density`](@ref) but converts to
pressure.
"""
function pressure(dfun, gas)
    function addstat!(d, Eω, Et, z, dz)
        d["pressure"] = PhysData.pressure(gas, dfun(z))
    end
end

function pressure(dfun, gases::Tuple)
    function addstat!(d, Eω, Et, z, dz)
        dens = dfun(z)
        for (di, gi) in zip(dens, gases)
            d["pressure_$gi"] = PhysData.pressure(gi, di)
        end
    end
end

"""
    core_radius(a)

Create stats function to capture core radius as defined by `a` (either a `Number` or a 
callable `a(z)`)
"""
function core_radius(a::Number)
    function addstat!(d, Eω, Et, z, dz)
        d["core_radius"] = a
    end
end

function core_radius(afun)
    function addstat!(d, Eω, Et, z, dz)
        d["core_radius"] = afun(z)
    end
end

"""
    zdw(mode)

Create stats function to capture the zero-dispersion wavelength (ZDW).

!!! warning
    Since [`Modes.zdw`](@ref) is based on root-finding of a derivative, this can be slow!
"""
function zdw(mode::Modes.AbstractMode; λmin=100e-9, λmax=3000e-9)
    λ00 = Modes.zdw(mode; λmin=λmin, λmax=λmax, z=0)
    function addstat!(d, Eω, Et, z, dz)
        d["zdw"] = missnan(Modes.zdw(mode, λ00; z=z))
        λ00 = d["zdw"]
    end
end

"""
    zdw(mode)

Create stats function to capture the zero-dispersion wavelength (ZDW).

!!! warning
    Since [`Modes.zdw`](@ref) is based on root-finding of a derivative, this can be slow!
"""
function zdw(modes; λmin=100e-9, λmax=3000e-9)
    λ00 = zeros(length(modes))
    for (ii, mode) in enumerate(modes)
        tmp = Modes.zdw(mode; λmin=λmin, λmax=λmax, z=0)
        if ismissing(tmp)
            λ00[ii] = λmin
        else
            λ00[ii] = tmp
        end
    end
    function addstat!(d, Eω, Et, z, dz)
        d["zdw"] = [missnan(Modes.zdw(modes[ii], λ00[ii]; z=z)) for ii in eachindex(modes)]
        λ00 .= d["zdw"]
    end
end

# convert missing to NaN
missnan(x) = ismissing(x) ? NaN : x

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
    iFT = FFTW.plan_ifft(copy(Eω), 1, flags=settings["fftw_flag"])
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
    iFT = FFTW.plan_ifft(Eωa, 1, flags=settings["fftw_flag"])
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

function default(grid, Eω, mode::Modes.AbstractMode, linop, transform;
                 windows=nothing, gas=nothing, onaxis=false, userfuns=Any[])
    _, energyfunω = Fields.energyfuncs(grid)
    funs = [ω0(grid), energy(grid, energyfunω), peakpower(grid),
            fwhm_t(grid), zdw_linop(mode, linop),
            density(transform.densityfun)]
    if !isnothing(gas)
        push!(funs, pressure(transform.densityfun, gas))
    end
    if onaxis
        push!(funs, peakintensity(grid, (mode,)))
    else
        push!(funs, peakintensity(grid, transform.aeff))
    end
    for resp in transform.resp
        if resp isa PlasmaCumtrapz
            ir = resp.ratefunc
            if onaxis
                push!(funs, electrondensity(grid, resp.ratefunc, transform.densityfun, (mode,)))
            else
                push!(funs, electrondensity(grid, resp.ratefunc, transform.densityfun, transform.aeff))
            end
        end
    end
    if !isnothing(windows)
        for win in windows
            push!(funs, energy_λ(grid, energyfunω, win))
        end
    end
    for (idx, uf) in enumerate(userfuns)
        if uf in funs
            @warn("userfun $idx is already present in the default set and will be ignored")
        else
            push!(funs, uf)
        end
    end
    collect_stats(grid, Eω, funs...)
end

function default(grid, Eω, modes::Modes.ModeCollection, linop, transform;
                 windows=nothing, gas=nothing, mode_error=true, userfuns=Any[])
    _, energyfunω = Fields.energyfuncs(grid)
    pol = transform.ts.indices == 1:2 ? :xy : transform.ts.indices == 1 ? :x : :y
    funs = [ω0(grid), energy(grid, energyfunω), peakpower(grid),
            peakintensity(grid, modes, components=pol), fwhm_t(grid),
            zdw_linop(modes, linop), density(transform.densityfun),
            fwhm_r(grid, modes; components=pol)]
    if !isnothing(gas)
        push!(funs, pressure(transform.densityfun, gas))
    end
    if mode_error
        push!(funs, mode_reconstruction_error(transform))
    end
    for resp in transform.resp
        if resp isa PlasmaCumtrapz
            ir = resp.ratefunc
            ed = electrondensity(grid, resp.ratefunc, transform.densityfun, modes,
                                 components=pol)
            push!(funs, ed)
        end
    end
    if !isnothing(windows)
        for win in windows
            push!(funs, energy_λ(grid, energyfunω, win))
        end
    end
    for (idx, uf) in enumerate(userfuns)
        if uf in funs
            @warn("userfun $uf is already present in the default set and will be ignored")
        else
            push!(funs, uf)
        end
    end
    collect_stats(grid, Eω, funs...)
end

# For constant linop, ZDW is also constant
function zdw_linop(mode::Modes.AbstractMode, linop::AbstractArray)
    zdw = missnan(Modes.zdw(mode))
    (d, Eω, Et, z, dz) -> d["zdw"] = zdw
end

function zdw_linop(modes, linop::AbstractArray)
    zdw = [missnan(Modes.zdw(mode)) for mode in modes]
    (d, Eω, Et, z, dz) -> d["zdw"] = zdw
end

zdw_linop(mode_s, linop) = zdw(mode_s)

end