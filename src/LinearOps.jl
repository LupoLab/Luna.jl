module LinearOps
import FFTW
import Hankel
import Luna: Modes, Grid, PhysData, Maths
import Luna.PhysData: wlfreq, c, crystal_internal_angle

getω0(grid::Grid.EnvGrid) = grid.ω0
getω0(grid::Grid.RealGrid) = 0.0

function fill_linop_matrix!(out, grid, β1::Number, βref::Number, k2, kperp2, idcs)
    ω0 = getω0(grid)
    for ii in idcs
        for ip in axes(k2, 2)
            for iω in eachindex(grid.ω)
                βsq = k2[iω, ip] - kperp2[ii]
                if βsq < 0
                    # negative βsq -> evanescent fields -> attenuation
                    out[iω, ip, ii] = -im*(-β1*(grid.ω[iω] - ω0) - βref) - min(sqrt(abs(βsq)), 200)
                else
                    out[iω, ip, ii] = -im*(sqrt(βsq) - β1*(grid.ω[iω] - ω0) - βref)
                end
            end
        end
    end
end

#=================================================#
#===============    FREE SPACE     ===============#
#=================================================#
function transverse_k2(xygrid::Grid.FreeGrid)
    kperp2 = @. xygrid.kx^2 + (xygrid.ky^2)'
    idcs = CartesianIndices((length(xygrid.kx), length(xygrid.ky)))
    kperp2, idcs
end

function transverse_k2(xgrid::Grid.Free2DGrid)
    kperp2 = xgrid.kx.^2
    idcs = CartesianIndices(xgrid.kx)
    kperp2, idcs
end

function transverse_k2(xygrid::Hankel.QDHT)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.kx), length(xygrid.ky)))
    kperp2, idcs
end


"""
    make_const_linop(grid, xygrid, n, β1, β0)

Make constant linear operator for free-space propagation. `n` is the refractive index (array),
β1 is 1/velocity of the reference frame and β0 is the wavevector at the reference wavelength.
"""
function make_const_linop(grid::Grid.AbstractGrid,
                          xygrid::Union{Grid.FreeGrid, Grid.Free2DGrid, Hankel.QDHT},
                          n::AbstractVecOrMat, β1::Number, β0::Number)
    kperp2, idcs = transverse_k2(xygrid)
    k2 = @. (n*grid.ω/c)^2
    out = zeros(ComplexF64, (length(grid.ω), size(n, 2), size(idcs)...))
    fill_linop_matrix!(out, grid, β1, β0, k2, kperp2, idcs)
    return out
end

thg_default(grid::Grid.RealGrid) = true
thg_default(grid::Grid.EnvGrid) = false

checkthg(grid::Grid.EnvGrid, thg) = nothing
checkthg(grid::Grid.RealGrid, thg) = thg || error("`thg` must be `true` for `RealGrid`s")

getβ0_n(grid::Grid.RealGrid, nfun, thg) = 0.0
getβ0_n(grid::Grid.EnvGrid, nfun, thg) = thg ? 0.0 : grid.ω0/c * nfun(wlfreq(grid.ω0))[end]

function make_const_linop(grid::Grid.AbstractGrid,
                          xygrid::Union{Grid.FreeGrid, Grid.Free2DGrid, Hankel.QDHT},
                          nfun, thg::Bool=thg_default(grid))
    checkthg(grid, thg)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(wlfreq(ωfirst))) # 1 if single ref index, 2 if nx, ny
    n = zeros(Float64, (length(grid.ω), np))
    for (ii, si) in enumerate(grid.sidx)
        if si
            n[ii, :] .= nfun(wlfreq(grid.ω[ii]))
        end
    end
    β1 = PhysData.dispersion_func(1, λ -> nfun(λ)[end])(grid.referenceλ)
    β0 = getβ0_n(grid, nfun, thg)
    make_const_linop(grid, xygrid, n, β1, β0)
end

function make_const_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid, nfuns::Tuple)
    nfunx, nfuny = nfuns
    # here nfunx(λ, δθ) also takes the angle and returns n_x(λ, θ)
    # nfuny(λ; z) just takes wavelength
    out = zeros(ComplexF64, (length(grid.ω), 2, length(xygrid.kx), length(xygrid.ky)))
    β1 = PhysData.dispersion_func(1, nfuny)(grid.referenceλ)
    for (iω, si) in enumerate(grid.sidx)
        if si
            ny = nfuny(wlfreq(grid.ω[iω]))
            ksq_ypol = (ny*grid.ω[iω]/c)^2
            for (ikx, kxi) in enumerate(xygrid.kx)
                δθ = crystal_internal_angle(nfunx, grid.ω[iω], kxi)
                nx = nfunx(wlfreq(grid.ω[iω]), δθ)
                for (iky, kyi) in enumerate(xygrid.ky)
                    k_xpol = nx*grid.ω[iω]/c
                    βsq_xpol = k_xpol^2 - kxi^2 - kyi^2
                    β_xpol = βsq_xpol < 0 ? -min(sqrt(abs(βsq_xpol)), 200) : sqrt(βsq_xpol)
                    out[iω, 1, ikx, iky] = -im*(β_xpol - β1*grid.ω[iω])

                    βsq_ypol = ksq_ypol - kxi^2 - kyi^2
                    β_ypol = βsq_ypol < 0 ? -min(sqrt(abs(βsq_ypol)), 200) : sqrt(βsq_ypol)
                    out[iω, 2, ikx, iky] = -im*(β_ypol - β1*grid.ω[iω])
                end
            end
        end
    end
    out
end

"""
    make_linop(grid, xygrid, nfun)

Make z-dependent linear operator for free-space propagation. `nfun(ω; z)` should return the
refractive index as a function of frequency `ω` and (kwarg) propagation distance `z`.
"""
function make_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid, nfun)
    kperp2 = @. xygrid.kx^2 + (xygrid.ky^2)'
    idcs = CartesianIndices((length(xygrid.kx), length(xygrid.ky)))
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx] ./ c).^2
        fill_linop_matrix!(out, grid, β1, 0.0, k2, kperp2, idcs)
    end
end

function make_linop(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid, nfun; thg=false)
    kperp2 = @. xygrid.kx^2 + (xygrid.ky^2)'
    idcs = CartesianIndices((length(xygrid.kx), length(xygrid.ky)))
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z).*grid.ω[grid.sidx]./c).^2
        βref = thg ? 0.0 : grid.ω0/c * nfun(grid.ω0; z=z)
        fill_linop_matrix!(out, grid, β1, βref, k2, kperp2, idcs)
    end
end

#=================================================#
#============   FREE SPACE (2D)   ================#
#=================================================#

function make_const_linop(grid::Grid.RealGrid, xgrid::Grid.Free2DGrid, nfuns::Tuple)
    nfunx, nfuny = nfuns
    # here nfunx(λ, δθ) also takes the angle and returns n_x(λ, θ)
    # nfuny(λ; z) just takes wavelength
    out = zeros(ComplexF64, (length(grid.ω), 2, length(xgrid.kx)))
    β1 = PhysData.dispersion_func(1, nfuny)(grid.referenceλ)
    for (iω, si) in enumerate(grid.sidx)
        if si
            ny = nfuny(wlfreq(grid.ω[iω]))
            ksq_ypol = (ny*grid.ω[iω]/c)^2
            for (ik, kxi) in enumerate(xgrid.kx)
                δθ = crystal_internal_angle(nfunx, grid.ω[iω], kxi)
                nx = nfunx(wlfreq(grid.ω[iω]), δθ)
                k_xpol = nx*grid.ω[iω]/c
                βsq_xpol = k_xpol^2 - kxi^2
                β_xpol = βsq_xpol < 0 ? -min(sqrt(abs(βsq_xpol)), 200) : sqrt(βsq_xpol)
                out[iω, 1, ik] = -im*(β_xpol - β1*grid.ω[iω])

                βsq_ypol = ksq_ypol - kxi^2
                β_ypol = βsq_ypol < 0 ? -min(sqrt(abs(βsq_ypol)), 200) : sqrt(βsq_ypol)
                out[iω, 2, ik] = -im*(β_ypol - β1*grid.ω[iω])
            end
        end
    end
    out
end

"""
    make_linop(grid, xgrid, nfun)

Make z-dependent linear operator for free-space propagation. `nfun(ω; z)` should return the
refractive index as a function of frequency `ω` and (kwarg) propagation distance `z`.
"""
function make_linop(grid::Grid.RealGrid, xgrid::Grid.Free2DGrid, nfun)
    kperp2 = xgrid.kx.^2
    idcs = CartesianIndices(xgrid.kx)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    k2 = zeros(Float64, (length(grid.ω), np))
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)[1]
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        for (ii, si) in enumerate(grid.sidx)
            if si
                k2[ii, :] .= (nfun(grid.ω[ii]; z) .* grid.ω[ii]./c).^2
            end
        end
        fill_linop_matrix!(out, grid, β1, 0.0, k2, kperp2, idcs)
    end
end

function make_linop(grid::Grid.EnvGrid, xgrid::Grid.Free2DGrid, nfun; thg=false)
    kperp2 = xgrid.kx.^2
    idcs = CartesianIndices(xgrid.kx)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    k2 = zeros(Float64, (length(grid.ω), np))
    nfunλ(z) = λ -> nfun(wlfreq(λ); z)[1]
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        for (ii, si) in enumerate(grid.sidx)
            if si
                k2[ii, :] .= (nfun(grid.ω[ii]; z) .* grid.ω[ii]./c).^2
            end
        end
        βref = thg ? 0.0 : grid.ω0/c * nfun(grid.ω0; z=z)[end]
        fill_linop_matrix!(out, grid, β1, βref, k2, kperp2, idcs)
    end
end

#=================================================#
#==============   RADIAL SYMMETRY   ==============#
#=================================================#
"""
    make_const_linop(grid, q::QDHT, n, β1)

Make constant linear operator for radial free-space. `n` is the refractive index (array)
and β1 is 1/velocity of the reference frame.
"""
function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT,
                          n::AbstractVecOrMat, β1::Number)
    out = Array{ComplexF64}(undef, (length(grid.ω), size(n, 2), q.N))
    k2 = @. (n*grid.ω/c)^2
    kr2 = q.k.^2
    fill_linop_matrix!(out, grid, β1, 0.0, k2, kr2, eachindex(q.k))
    return out
end

function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT, nfun)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst)) # 1 if single ref index, 2 if nx, ny
    n = zeros(Float64, (length(grid.ω), np))
    for (ii, si) in enumerate(grid.sidx)
        if si
            n[ii, :] .= nfun(wlfreq(grid.ω[ii]))
        end
    end
    β1 = PhysData.dispersion_func(1, λ -> nfun(λ)[end])(grid.referenceλ)
    make_const_linop(grid, q, n, β1)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT, nfun; thg=false)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst)) # 1 if single ref index, 2 if nx, ny
    n = zeros(Float64, (length(grid.ω), np))
    for (ii, si) in enumerate(grid.sidx)
        if si
            n[ii, :] .= nfun(wlfreq(grid.ω[ii]))
        end
    end
    β1 = PhysData.dispersion_func(1, λ -> nfun(λ)[end])(grid.referenceλ)
    if thg
        β0const = 0.0
    else
        β0const = grid.ω0/c * nfun(2π*c./grid.ω0)[1]
    end
    make_const_linop(grid, q, n, β1, β0const)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT,
                          n::AbstractVecOrMat, β1::Number, β0ref::Number)
    out = Array{ComplexF64}(undef, (length(grid.ω), size(n, 2), q.N))
    k2 = @. (n*grid.ω/c)^2
    kr2 = q.k.^2
    fill_linop_matrix!(out, grid, β1, β0ref, k2, kr2, eachindex(q.k))
    return out
end

"""
    make_linop(grid, q::QDHT, nfun)

Make z-dependent linear operator for radial free-space propagation. `nfun(ω; z)` should
return the refractive index as a function of frequency `ω` and (kwarg) propagation
distance `z`.
"""
function make_linop(grid::Grid.RealGrid, q::Hankel.QDHT, nfun)
    kr2 = q.k.^2
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    k2 = zeros(Float64, (length(grid.ω), np))
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)[end]
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx, :] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx]./c).^2
        fill_linop_matrix!(out, grid, β1, 0.0, k2, kr2, eachindex(q.k))
    end
end

function make_linop(grid::Grid.EnvGrid, q::Hankel.QDHT, nfun; thg=false)
    kr2 = q.k.^2
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    k2 = zeros(Float64, (length(grid.ω), np))
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)[end]
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx, :] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx]./c).^2
        βref = thg ? 0.0 : grid.ω0/c * nfun(grid.ω0; z=z)[end]
        fill_linop_matrix!(out, grid, β1, βref, k2, kr2, eachindex(q.k))
    end
end

#=================================================#
#===============   MODE AVERAGE   ================#
#=================================================#

"""
    αlim!(α)

Limit α so that we do not get overflow in exp(α*dz)
"""
function αlim!(α)
    # magic number: this is 130 dB/cm
    # a test script sensitive to this is test_main_rect_env.jl
    clamp!(α, 0.0, 3000.0)
end

"""
    conj_clamp(n, ω)

Simultaneously conjugate and clamp the effective index `n` to safe levels.

The real part is lower-bounded at 1e-3 and the imaginary part upper-bounded at an attenuation
coefficient `α` of 3000 (130 dB/cm). The limits are somewhat arbitrary and chosen empirically
from previous bugs. See https://github.com/LupoLab/Luna/pull/142.

See also [`αlim!`](@ref).
"""
conj_clamp(n, ω) = clamp(real(n), 1e-3, Inf) - im*clamp(imag(n), 0, 3000*c/ω)

function make_const_linop(grid::Grid.AbstractGrid, βfun!, αfun!, β1::Number, β0::Number)
    ω0 = getω0(grid)
    β = similar(grid.ω)
    βfun!(β, 0)
    α = similar(grid.ω)
    αfun!(α, 0)
    αlim!(α)
    linop = @. -im*(β - β1*(grid.ω - ω0) - β0) - α/2
    linop[.!grid.sidx] .= 0
    return linop
end

getβ0_mode(grid::Grid.RealGrid, mode, λ0, thg) = 0.0
getβ0_mode(grid::Grid.EnvGrid, mode, λ0, thg) = thg ? 0.0 : Modes.β(mode, wlfreq(λ0))

"""
    make_const_linop(grid, mode, λ0)

Make constant linear operator for mode-averaged propagation in mode `mode` with a reference
wavelength `λ0`.
"""
function make_const_linop(grid::Grid.AbstractGrid, mode::Modes.AbstractMode, λ0;
                          thg::Bool=thg_default(grid))
    checkthg(grid, thg)
    β1 = Modes.dispersion(mode, 1, wlfreq(λ0))
    β0 = getβ0_mode(grid, mode, λ0, thg)
    βconst = zero(grid.ω)
    βconst[grid.sidx] = Modes.β.(mode, grid.ω[grid.sidx])
    βconst[.!grid.sidx] .= 1
    function βfun!(out, z)
        out .= βconst
    end
    αconst = zero(grid.ω)
    αconst[grid.sidx] = Modes.α.(mode, grid.ω[grid.sidx])
    function αfun!(out, z)
        out .= αconst
    end
    make_const_linop(grid, βfun!, αfun!, β1, β0), βfun!, β1, αfun!
end


"""
    neff_β_grid(grid, mode, λ0; ref_mode=1)

Create closures which return the effective index and propagation constant
as a function of the frequency grid **index**, rather than the frequency itself.
Any [`Modes.AbstractMode`](@ref) may define its own method for `neff_β_grid` to
accelerate repeated calculation on the same frequency grid.
"""
function neff_β_grid(grid, mode, λ0)
    let grid=grid, mode=mode
        _neff(iω; z) = Modes.neff(mode, grid.ω[iω]; z=z)
        _β(iω; z) = Modes.β(mode, grid.ω[iω]; z=z)
        _neff, _β
    end
end

function make_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff, β = neff_β_grid(grid, mode, λ0)
    linop! = let neff=neff, ω=grid.ω, mode=mode, ω0=wlfreq(λ0)
        function linop!(out, z)
            fill!(out, 0.0)
            β1 = Modes.dispersion(mode, 1, ω0, z=z)::Float64
            for iω in sidcs
                nc = conj_clamp(neff(iω; z=z), ω[iω])
                out[iω] = -im*(ω[iω]/c*nc - ω[iω]*β1)
            end
        end
    end
    βfun! = let β=β, ω=grid.ω
        function βfun!(out, z)
            fill!(out, 1.0)
            for iω in sidcs
                out[iω] = β(iω; z=z)
            end
        end
    end
    return linop!, βfun!
end

function make_linop(grid::Grid.EnvGrid, mode::Modes.AbstractMode, λ0; thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff, β = neff_β_grid(grid, mode, λ0)
    linop! = let neff=neff, ω=grid.ω, mode=mode, ω0=wlfreq(λ0), sidcs=sidcs
        function linop!(out, z)
            fill!(out, 0.0)
            β1 = Modes.dispersion(mode, 1, ω0, z=z)::Float64
            if !thg
                βref = Modes.β(mode, ω0, z=z)
            end
            for iω in sidcs
                nc = conj_clamp(neff(iω; z=z), ω[iω])
                out[iω] = -im*(ω[iω]/c*nc - (ω[iω] - grid.ω0)*β1)
                if !thg
                    out[iω] -= -im*βref
                end
            end
        end
    end
    βfun! = let β=β, sidcs=sidcs
        function βfun!(out, z)
            fill!(out, 1.0)
            for iω in sidcs
                out[iω] = β(iω, z=z)
            end
        end
    end
    return linop!, βfun!
end

#=================================================#
#=================   MULTIMODE   =================#
#=================================================#
"""
    make_const_linop(grid, modes, λ0; ref_mode=1)

Make constant (z-invariant) linear operator for multimode propagation. The frame velocity is
taken as the group velocity at wavelength `λ0` in the mode given by `ref_mode` (which
indexes into `modes`)
"""
function make_const_linop(grid::Grid.RealGrid, modes::Modes.ModeCollection, λ0; ref_mode=1)
    β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0))
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[grid.sidx] = Modes.β.(modes[i], grid.ω[grid.sidx])
        βconst[.!grid.sidx] .= 1
        α = zeros(length(grid.ω))
        α[grid.sidx] .= Modes.α.(modes[i], grid.ω[grid.sidx])
        αlim!(α)
        linops[:,i] = im.*(-βconst .+ grid.ω.*β1) .- α./2
    end
    linops
end

function make_const_linop(grid::Grid.EnvGrid, modes::Modes.ModeCollection, λ0; ref_mode=1, thg=false)
    β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0))
    if thg
        βref = 0.0
    else
        βref = Modes.β(modes[ref_mode], wlfreq(λ0))
    end
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[grid.sidx] = Modes.β.(modes[i], grid.ω[grid.sidx])
        βconst[.!grid.sidx] .= 1
        α = Modes.α.(modes[i], grid.ω)
        αlim!(α)
        linops[:,i] = -im.*(βconst .- (grid.ω .- grid.ω0).*β1 .- βref) .- α./2
    end
    linops
end

"""
    neff_grid(grid, modes, λ0; ref_mode=1)

Create a closure that returns the effective index as a function of the frequency grid and mode
**index**, rather than the mode and frequency themselves. Any [`Modes.AbstractMode`](@ref)
may define its one method for `neff_grid` to accelerate repeated calculation on the same
frequency grid.
"""
function neff_grid(grid, modes, λ0; ref_mode=1)
    _neff = let grid=grid, modes=modes
        _neff(iω, iim; z) = Modes.neff(modes[iim], grid.ω[iω]; z=z)
    end
    _neff
end

function make_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff = neff_grid(grid, modes, λ0; ref_mode=ref_mode)
    linop! = let neff=neff, ω=grid.ω, modes=modes, ω0=wlfreq(λ0), ref_mode=ref_mode
        function linop!(out, z)
            β1 = Modes.dispersion(modes[ref_mode], 1, ω0, z=z)::Float64
            fill!(out, 0.0)
            for i in eachindex(modes)
                for iω in sidcs
                    nc = conj_clamp(neff(iω, i; z=z), ω[iω])
                    out[iω, i] = -im*(ω[iω]/c*nc - ω[iω]*β1)
                end
            end
        end
    end
end

function make_linop(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff = neff_grid(grid, modes, λ0; ref_mode=ref_mode)
    linop! = let neff=neff, ω=grid.ω, modes=modes, ω0=wlfreq(λ0), ref_mode=ref_mode
        function linop!(out, z)
            β1 = Modes.dispersion(modes[ref_mode], 1, ω0, z=z)::Float64
            fill!(out, 0.0)
            if !thg
                βref = Modes.β(modes[ref_mode], ω0, z=z)
            end
            for i in eachindex(modes)
                for iω in sidcs
                    nc = conj_clamp(neff(iω, i; z=z), ω[iω])
                    out[iω, i] = -im*(ω[iω]/c*nc - (ω[iω] - grid.ω0)*β1)
                    if !thg
                        out[iω, i] -= -im*βref
                    end
                end
            end
        end
    end
end


end
