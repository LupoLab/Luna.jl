module LinearOps
import FFTW
import Hankel
import Luna: Modes, Grid, PhysData, Maths
import Luna.PhysData: wlfreq

#=================================================#
#===============    FREE SPACE     ===============#
#=================================================#
"""
    make_const_linop(grid, xygrid, n, frame_vel)

Make constant linear operator for full 3D propagation. `n` is the refractive index (array)
and β1 is 1/velocity of the reference frame.
"""
function make_const_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid,
                          n::AbstractArray, β1::Number)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    k2[grid.sidx] .= (n[grid.sidx] .* grid.ω[grid.sidx] ./ PhysData.c).^2
    out = zeros(ComplexF64, (length(grid.ω), length(xygrid.ky), length(xygrid.kx)))
    _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs)
    return out
end

function make_const_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid, nfun)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(2π*PhysData.c./grid.ω[grid.sidx])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    make_const_linop(grid, xygrid, n, β1)
end

function make_const_linop(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
                          n::AbstractArray, β1::Number, β0ref::Number; thg=false)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    k2[grid.sidx] .= (n[grid.sidx].*grid.ω[grid.sidx]./PhysData.c).^2
    out = zeros(ComplexF64, (length(grid.ω), length(xygrid.ky), length(xygrid.kx)))
    _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs, β0ref; thg=thg)
    return out
end

function make_const_linop(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid, nfun,     
                          thg=false)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(wlfreq.(grid.ω[grid.sidx]))
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    if thg
        β0const = 0.0
    else
        β0const = grid.ω0/PhysData.c * nfun(wlfreq(grid.ω0))
    end
    make_const_linop(grid, xygrid, n, β1, β0const; thg=thg)
end

"""
    make_linop(grid, xygrid, nfun)

Make z-dependent linear operator for free-space propagation. `nfun(ω; z)` should return the
refractive index as a function of frequency `ω` and (kwarg) propagation distance `z`.
"""
function make_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid, nfun)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx] ./ PhysData.c).^2
        _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs)
    end
end

# Internal routine -- function barrier aids with JIT compilation
function _fill_linop_xy!(out, grid::Grid.RealGrid, β1::Float64, k2, kperp2, idcs)
    for ii in idcs
        for iω in eachindex(grid.ω)
            βsq = k2[iω] - kperp2[ii]
            if βsq < 0
                # negative βsq -> evanescent fields -> attenuation
                out[iω, ii] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
            else
                out[iω, ii] = -im*(sqrt(βsq) - β1*grid.ω[iω])
            end
        end
    end
end

function make_linop(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid, nfun; thg=false)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z).*grid.ω[grid.sidx]./PhysData.c).^2
        βref = thg ? 0.0 : grid.ω0/PhysData.c * nfun(grid.ω0; z=z)
        _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs, βref; thg=thg)
    end
end

function _fill_linop_xy!(out, grid::Grid.EnvGrid, β1::Float64, k2, kperp2, idcs, βref; thg)
    for ii in idcs
        for iω in eachindex(grid.ω)
            βsq = k2[iω] - kperp2[ii]
            if βsq < 0
                # negative βsq -> evanescent fields -> attenuation
                out[iω, ii] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
            else
                out[iω, ii] = -im*(sqrt(βsq) - β1*grid.ω[iω])
            end
            if !thg
                out[iω, ii] -= -im*βref
            end
        end
    end
end

#=================================================#
#==============   RADIAL SYMMETRY   ==============#
#=================================================#
"""
    make_const_linop(grid, q::QDHT, n, frame_vel)

Make constant linear operator for radial free-space. `n` is the refractive index (array)
and β1 is 1/velocity of the reference frame.
"""
function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT,
                          n::AbstractArray, β1::Number)
    out = Array{ComplexF64}(undef, (length(grid.ω), q.N))
    k2 = @. (n*grid.ω/PhysData.c)^2
    kr2 = q.k.^2
    _fill_linop_r!(out, grid, β1, k2, kr2, q.N)
    return out
end

function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT, nfun)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(2π*PhysData.c./grid.ω[grid.sidx])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    make_const_linop(grid, q, n, β1)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT, nfun; thg=false)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(2π*PhysData.c./grid.ω[grid.sidx])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    if thg
        β0const = 0.0
    else
        β0const = grid.ω0/PhysData.c * nfun(2π*PhysData.c./grid.ω0)
    end
    make_const_linop(grid, q, n, β1, β0const; thg=thg)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT,
                          n::AbstractArray, β1::Number, β0ref::Number; thg=false)
    out = Array{ComplexF64}(undef, (length(grid.ω), q.N))
    k2 = @. (n*grid.ω/PhysData.c)^2
    kr2 = q.k.^2
    _fill_linop_r!(out, grid, β1, k2, kr2, q.N, β0ref, thg)
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
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx]./PhysData.c).^2
        _fill_linop_r!(out, grid, β1, k2, kr2, q.N)
    end
end

function _fill_linop_r!(out, grid::Grid.RealGrid, β1, k2, kr2, Nr)
    for ir = 1:Nr
        for iω = 1:length(grid.ω)
            βsq = k2[iω] - kr2[ir]
            if βsq < 0
                # negative βsq -> evanescent fields -> attenuation
                out[iω, ir] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
            else
                out[iω, ir] = -im*(sqrt(βsq) - β1*grid.ω[iω])
            end
        end
    end
end

function make_linop(grid::Grid.EnvGrid, q::Hankel.QDHT, nfun; thg=false)
    kr2 = q.k.^2
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx]./PhysData.c).^2
        βref = thg ? 0.0 : grid.ω0/PhysData.c * nfun(grid.ω0; z=z)
        _fill_linop_r!(out, grid, β1, k2, kr2, q.N, βref, thg)
    end
end

function _fill_linop_r!(out, grid::Grid.EnvGrid, β1, k2, kr2, Nr, βref, thg)
    for ir = 1:Nr
        for iω = 1:length(grid.ω)
            βsq = k2[iω] - kr2[ir]
            if βsq < 0
                # negative βsq -> evanescent fields -> attenuation
                out[iω, ir] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
            else
                out[iω, ir] = -im*(sqrt(βsq) - β1*grid.ω[iω])
            end
            if !thg
                out[iω, ir] -= -im*βref
            end
        end
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
conj_clamp(n, ω) = clamp(real(n), 1e-3, Inf) - im*clamp(imag(n), 0, 3000*PhysData.c/ω)

function make_const_linop(grid::Grid.RealGrid, βfun!, αfun!, β1)
    β = similar(grid.ω)
    βfun!(β, 0)
    α = similar(grid.ω)
    αfun!(α, 0)
    αlim!(α)
    linop = @. -im*(β-β1*grid.ω) - α/2
    linop[.!grid.sidx] .= 0
    return linop
end

function make_const_linop(grid::Grid.EnvGrid, βfun!, αfun!, β1, β0ref)
    β = similar(grid.ω)
    βfun!(β, 0)
    α = similar(grid.ω)
    αfun!(α, 0)
    αlim!(α)
    linop = -im.*(β .- β1.*(grid.ω .- grid.ω0) .- β0ref) .- α./2
    linop[.!grid.sidx] .= 0
    return linop
end

"""
    make_const_linop(grid, mode, λ0)

Make constant linear operator for mode-averaged propagation in mode `mode` with a reference
wavelength `λ0`.
"""
function make_const_linop(grid::Grid.EnvGrid, mode::Modes.AbstractMode, λ0; thg=false)
    β1const = Modes.dispersion(mode, 1, wlfreq(λ0))
    if thg
        β0const = 0.0
    else
        β0const = Modes.β(mode, wlfreq(λ0))
    end
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
    make_const_linop(grid, βfun!, αfun!, β1const, β0const), βfun!, β1const, αfun!
end

function make_const_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    β1const = Modes.dispersion(mode, 1, wlfreq(λ0))
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
    make_const_linop(grid, βfun!, αfun!, β1const), βfun!, β1const, αfun!
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
                out[iω] = -im*(ω[iω]/PhysData.c*nc - ω[iω]*β1)
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
                out[iω] = -im*(ω[iω]/PhysData.c*nc - (ω[iω] - grid.ω0)*β1)
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
function make_const_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
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

function make_const_linop(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false)
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
                    out[iω, i] = -im*(ω[iω]/PhysData.c*nc - ω[iω]*β1)
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
                    out[iω, i] = -im*(ω[iω]/PhysData.c*nc - (ω[iω] - grid.ω0)*β1)
                    if !thg
                        out[iω, i] -= -im*βref
                    end
                end
            end
        end
    end
end



#=================================================#
#=========  UNITARY PHASE OF THE LINOP  ==========#
#=================================================#

"""
    AbstractUnitaryPhase

Callable which fills an array with the accumulated **unitary** phase of a linear operator,
`Φ(ω, z) = ∫₀ᶻ imag(linop(ω, z')) dz'`, i.e. the phase the linear operator would impart in
the absence of loss or gain. Subtypes implement `(p::AbstractUnitaryPhase)(out, z)`, which
**mutates** `out` and returns it.

This is the propagator required by the modified shot-noise model: the noise field must feel
dispersion (unitary, and the phase evolution of the medium's own vacuum modes) but neither
loss nor gain. See [`unitary_phase`](@ref) and
[`NonlinearRHS.ModifiedNoise`](@ref Luna.NonlinearRHS.ModifiedNoise).
"""
abstract type AbstractUnitaryPhase end

"""
    ConstUnitaryPhase(dφ)

[`AbstractUnitaryPhase`](@ref) for a `z`-invariant linear operator, for which
`Φ(ω, z) = imag(linop)·z` exactly.
"""
struct ConstUnitaryPhase{aT<:AbstractArray} <: AbstractUnitaryPhase
    dφ::aT # imag(linop), i.e. dΦ/dz, constant in z
end

(p::ConstUnitaryPhase)(out, z) = (@. out = p.dφ * z; out)

"""
    TabulatedUnitaryPhase(z, Φ, dΦ, atol, err)

[`AbstractUnitaryPhase`](@ref) for a `z`-dependent linear operator. `Φ` and `dΦ` hold the
accumulated phase and its derivative `imag(linop)` at the nodes `z`, stacked along the last
dimension. The nodes are **not** uniformly spaced: they are placed by
[`unitary_phase`](@ref) so that the interpolation error stays below `atol`, and `err` records
the largest error actually measured during construction. Between nodes the phase is
interpolated with a cubic Hermite, which uses both the tabulated value and the tabulated
derivative. Outside `[z[1], z[end]]` it extrapolates linearly.
"""
struct TabulatedUnitaryPhase{aT<:AbstractArray} <: AbstractUnitaryPhase
    z::Vector{Float64}
    Φ::aT
    dΦ::aT
    atol::Float64 # tolerance the nodes were placed to satisfy, in radians
    err::Float64 # largest interpolation error measured while placing them, in radians
end

function (p::TabulatedUnitaryPhase)(out, z)
    zs = p.z
    d = ndims(p.Φ)
    if z >= zs[end] || z <= zs[1] # linear extrapolation off either end of the table
        k = z <= zs[1] ? 1 : length(zs)
        Φk = selectdim(p.Φ, d, k)
        dk = selectdim(p.dΦ, d, k)
        δ = z - zs[k]
        @. out = Φk + dk*δ
        return out
    end
    k = searchsortedlast(zs, z)
    h = zs[k+1] - zs[k]
    s = (z - zs[k])/h
    # cubic Hermite basis
    s2 = s*s
    s3 = s2*s
    h00 = 2s3 - 3s2 + 1
    h10 = h*(s3 - 2s2 + s)
    h01 = -2s3 + 3s2
    h11 = h*(s3 - s2)
    Φk = selectdim(p.Φ, d, k)
    Φk1 = selectdim(p.Φ, d, k+1)
    dk = selectdim(p.dΦ, d, k)
    dk1 = selectdim(p.dΦ, d, k+1)
    @. out = h00*Φk + h10*dk + h01*Φk1 + h11*dk1
    out
end

"""
    unitary_phase(linop, grid, sz=size(linop); atol=1e-6, margin=0.05, maxdepth=24, maxnodes=4096)

Create an [`AbstractUnitaryPhase`](@ref) for the linear operator `linop`, which may be either
a constant array or a mutating function `linop!(out, z)`. `sz` is the size of the array
`linop!` fills (equal to `size(linop)` in the constant case, and to the size of the noise
field in general).

A constant `linop` needs no work: `Φ = imag(linop)·z` exactly.

For a `z`-dependent `linop!` the accumulated phase is tabulated by **adaptive bisection**.
Each candidate interval is integrated with Simpson's rule and then checked against the thing
that actually matters — the cubic Hermite interpolant that will be used to read the table
back. The interpolant's error peaks at the interval midpoint, where it is compared against
the directly integrated value there, and the interval is bisected until that error is at most
`atol` radians (across every element of `linop`). The largest error that survives is recorded
in the returned object and reported.

This matters because Luna's `z`-dependent operators are usually *not* smooth. A pressure
gradient built by [`Capillary.gradient`](@ref Luna.Capillary.gradient) goes as
`√(p₀² + z/L(p₁² − p₀²))`, which has a `1/√z` cusp in its derivative at the entrance whenever
`p₀ = 0`, and a multi-section fill has a derivative discontinuity at every junction; tapers
are whatever function the user supplies. Uniform nodes converge at second order or worse
across such a feature, and give no way to tell how far off they are. Bisection puts nodes
only where they are needed — a kink costs a handful of extra intervals rather than a finer
grid everywhere — so the table is usually *smaller* than a uniform one of equal accuracy.

The table covers `[0, (1 + margin)·zmax]` rather than `[0, zmax]`, because `RK45.solve` runs
`while tn <= tmax` and so overshoots the end of the fibre on its final step -- and that step's
stages are what the last saved plane is interpolated from. `linop!` has to be defined a little
past `zmax` for this, which it already does: `RK45.make_prop!` evaluates it there too.
Anything beyond the margin still falls back to linear extrapolation.

`maxdepth` and `maxnodes` bound the work if `linop!` is genuinely discontinuous in `z`; if
either bound stops refinement before `atol` is met, a warning reports the error achieved.
"""
unitary_phase(linop::AbstractArray, grid, sz=size(linop); kwargs...) =
    ConstUnitaryPhase(imag.(linop))

function unitary_phase(linop!, grid, sz; atol=1e-6, margin=0.05, maxdepth=24, maxnodes=4096)
    N = length(sz)
    buf = Array{ComplexF64}(undef, sz)
    neval = Ref(0)
    dat = function (z)
        neval[] += 1
        linop!(buf, z)
        imag.(buf)
    end
    zmax = float(grid.zmax)*(1 + margin)
    znodes = Float64[0.0]
    dnodes = Array{Float64, N}[dat(0.0)]
    deltas = Array{Float64, N}[] # accumulated phase across each accepted interval
    worst = Ref(0.0)
    _refine!(znodes, dnodes, deltas, worst, dat, 0.0, zmax, dnodes[1], dat(zmax), nothing,
             atol, 0, maxdepth, maxnodes)

    n = length(znodes)
    Φ = zeros(Float64, (sz..., n))
    dΦ = zeros(Float64, (sz..., n))
    d = N + 1
    selectdim(dΦ, d, 1) .= dnodes[1]
    for k = 2:n
        selectdim(Φ, d, k) .= selectdim(Φ, d, k-1) .+ deltas[k-1]
        selectdim(dΦ, d, k) .= dnodes[k]
    end
    if worst[] > atol
        @warn("Noise phase table did not reach its tolerance: $n nodes, "*
              "max interpolation error $(worst[]) rad against a tolerance of $atol rad. "*
              "The linear operator may be discontinuous in z.")
    else
        @info("Noise phase table: $n nodes, $(neval[]) linop evaluations, "*
              "max interpolation error $(worst[]) rad.")
    end
    TabulatedUnitaryPhase(znodes, Φ, dΦ, float(atol), worst[])
end

#= Bisect [a, b] until the cubic Hermite interpolant built from the endpoint values and
   derivatives is within atol of the true phase at the midpoint, which is where its error
   peaks. `dmid` is the already-evaluated derivative at the midpoint, if the caller has it
   (a bisection's two children each inherit one of the parent's quarter points), so each
   call costs two new evaluations of linop rather than three. =#
function _refine!(znodes, dnodes, deltas, worst, dat, a, b, da, db, dmid,
                  atol, depth, maxdepth, maxnodes)
    h = b - a
    m = a + h/2
    dm = isnothing(dmid) ? dat(m) : dmid
    dq1 = dat(a + h/4)
    dq2 = dat(a + 3h/4)
    ΔΦ = @. h/12*(da + 4dq1 + 2dm + 4dq2 + db) # Φ(b) - Φ(a), Simpson on each half
    err = 0.0
    for i in eachindex(ΔΦ)
        hermite = ΔΦ[i]/2 + h*(da[i] - db[i])/8 # Hermite at the midpoint, minus Φ(a)
        exact = h/12*(da[i] + 4dq1[i] + dm[i]) # Simpson over [a, m]
        err = max(err, abs(hermite - exact))
    end
    if err <= atol || depth >= maxdepth || length(znodes) >= maxnodes
        push!(znodes, b)
        push!(dnodes, db)
        push!(deltas, ΔΦ)
        worst[] = max(worst[], err)
        return
    end
    _refine!(znodes, dnodes, deltas, worst, dat, a, m, da, dm, dq1,
             atol, depth+1, maxdepth, maxnodes)
    _refine!(znodes, dnodes, deltas, worst, dat, m, b, dm, db, dq2,
             atol, depth+1, maxdepth, maxnodes)
end

end
