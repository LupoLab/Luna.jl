module LinearOps
import FFTW
import Hankel
import Luna: Modes, Grid, PhysData, Maths
import Luna.PhysData: wlfreq

#=================================================#
#===============    FREE SPACE     ===============#
#=================================================#
"""
    make_const_linop(grid, xygrid, n, β1)

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
#============   FREE SPACE (2D)   ================#
#=================================================#
"""
    make_const_linop(grid, xgrid, n, β1)

Make constant linear operator for 2D free-space propagation. `n` is the refractive index (array)
and β1 is 1/velocity of the reference frame.
"""
function make_const_linop(grid::Grid.RealGrid, xgrid::Grid.Free2DGrid,
                          n::AbstractArray, β1::Number)
    kperp2 = xgrid.kx.^2
    idcs = CartesianIndices(xgrid.kx)
    k2 = @. (n*grid.ω/PhysData.c)^2
    out = zeros(ComplexF64, (length(grid.ω), size(n, 2), length(xgrid.kx)))
    _fill_linop_x!(out, grid, β1, k2, kperp2, idcs)
    return out
end

function make_const_linop(grid::Grid.RealGrid, xgrid::Grid.Free2DGrid, nfun)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    n = zeros(Float64, (length(grid.ω), np))
    for (ii, si) in enumerate(grid.sidx)
        if si
            n[ii, :] .= nfun(2π*PhysData.c./grid.ω[ii])
        end
    end
    β1 = PhysData.dispersion_func(1, λ -> nfun(λ)[1])(grid.referenceλ)
    make_const_linop(grid, xgrid, n, β1)
end

function make_const_linop(grid::Grid.EnvGrid, xgrid::Grid.Free2DGrid,
                          n::AbstractArray, β1::Number, β0ref::Number; thg=false)
    kperp2 = xgrid.kx.^2
    idcs = CartesianIndices(xgrid.kx)
    k2 = @. (n*grid.ω/PhysData.c)^2
    out = zeros(ComplexF64, (length(grid.ω), size(n, 2), length(xgrid.kx)))
    _fill_linop_x!(out, grid, β1, k2, kperp2, idcs, β0ref; thg=thg)
    return out
end

function make_const_linop(grid::Grid.EnvGrid, xgrid::Grid.Free2DGrid, nfun,     
                          thg=false)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    n = zeros(Float64, (length(grid.ω), np))
    for (ii, si) in enumerate(grid.sidx)
        if si
            n[ii, :] .= nfun(2π*PhysData.c./grid.ω[ii])
        end
    end
    β1 = PhysData.dispersion_func(1, λ -> nfun(λ)[1])(grid.referenceλ)
    if thg
        β0const = 0.0
    else
        β0const = grid.ω0/PhysData.c * nfun(wlfreq(grid.ω0))
    end
    make_const_linop(grid, xgrid, n, β1, β0const; thg=thg)
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
                k2[ii, :] .= (nfun(wlfreq(grid.ω[ii]; z)) .* grid.ω[ii]./PhysData.c).^2
            end
        end
        _fill_linop_x!(out, grid, β1, k2, kperp2, idcs)
    end
end

# Internal routine -- function barrier aids with JIT compilation
function _fill_linop_x!(out, grid::Grid.RealGrid, β1::Float64, k2, kperp2, idcs)
    for ii in idcs
        for ip in axes(k2, 2)
            for iω in eachindex(grid.ω)
                βsq = k2[iω, ip] - kperp2[ii]
                if βsq < 0
                    # negative βsq -> evanescent fields -> attenuation
                    out[iω, ip, ii] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
                else
                    out[iω, ip, ii] = -im*(sqrt(βsq) - β1*grid.ω[iω])
                end
            end
        end
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
                k2[ii, :] .= (nfun(wlfreq(grid.ω[ii]); z) .* grid.ω[ii]./PhysData.c).^2
            end
        end
        βref = thg ? 0.0 : grid.ω0/PhysData.c * nfun(grid.ω0; z=z)[1]
        _fill_linop_x!(out, grid, β1, k2, kperp2, idcs, βref; thg=thg)
    end
end

function _fill_linop_x!(out, grid::Grid.EnvGrid, β1::Float64, k2, kperp2, idcs, βref; thg)
    for ii in idcs
        for ip in axes(k2, 2)
            for iω in eachindex(grid.ω)
                βsq = k2[iω, ip] - kperp2[ii]
                if βsq < 0
                    # negative βsq -> evanescent fields -> attenuation
                    out[iω, ip, ii] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
                else
                    out[iω, ip, ii] = -im*(sqrt(βsq) - β1*grid.ω[iω])
                end
                if !thg
                    out[iω, ip, ii] -= -im*βref
                end
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
                          n::AbstractMatrix, β1::Number)
    out = Array{ComplexF64}(undef, (length(grid.ω), size(n, 2), q.N))
    k2 = @. (n*grid.ω/PhysData.c)^2
    kr2 = q.k.^2
    _fill_linop_r!(out, grid, β1, k2, kr2, q.N)
    return out
end

function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT, nfun)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    n = zeros(Float64, (length(grid.ω), np))
    for (ii, si) in enumerate(grid.sidx)
        if si
            n[ii, :] .= nfun(2π*PhysData.c./grid.ω[ii])
        end
    end
    β1 = PhysData.dispersion_func(1, λ -> nfun(λ)[1])(grid.referenceλ)
    make_const_linop(grid, q, n, β1)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT, nfun; thg=false)
    ωfirst = grid.ω[findfirst(grid.sidx)]
    np = length(nfun(ωfirst; z=0)) # 1 if single ref index, 2 if nx, ny
    n = zeros(Float64, (length(grid.ω), np))
    for (ii, si) in enumerate(grid.sidx)
        if si
            n[ii, :] .= nfun(2π*PhysData.c./grid.ω[ii])
        end
    end
    β1 = PhysData.dispersion_func(1, λ -> nfun(λ)[1])(grid.referenceλ)
    if thg
        β0const = 0.0
    else
        β0const = grid.ω0/PhysData.c * nfun(2π*PhysData.c./grid.ω0)[1]
    end
    make_const_linop(grid, q, n, β1, β0const; thg=thg)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT,
                          n::AbstractMatrix, β1::Number, β0ref::Number; thg=false)
    out = Array{ComplexF64}(undef, (length(grid.ω), size(n, 2), q.N))
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
        for ip in axes(k2, 2)
            for iω = 1:length(grid.ω)
                βsq = k2[iω, ip] - kr2[ir]
                if βsq < 0
                    # negative βsq -> evanescent fields -> attenuation
                    out[iω, ip, ir] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
                else
                    out[iω, ip, ir] = -im*(sqrt(βsq) - β1*grid.ω[iω])
                end
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

function _fill_linop_r!(out, grid::Grid.EnvGrid, β1, k2::AbstractMatrix, kr2, Nr, βref, thg)
    for ir = 1:Nr
        for ip = 1:2
            for iω = 1:length(grid.ω)
                βsq = k2[iω, ip] - kr2[ir]
                if βsq < 0
                    # negative βsq -> evanescent fields -> attenuation
                    out[iω, ip, ir] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
                else
                    out[iω, ip, ir] = -im*(sqrt(βsq) - β1*grid.ω[iω])
                end
                if !thg
                    out[iω, ip, ir] -= -im*βref
                end
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


end