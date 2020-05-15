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
    k2[2:end] .= (n[2:end] .* grid.ω[2:end] ./ PhysData.c).^2
    out = zeros(ComplexF64, (length(grid.ω), length(xygrid.ky), length(xygrid.kx)))
    _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs)
    return out
end

function make_const_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid, nfun)
    n = zero(grid.ω)
    n[2:end] = nfun.(2π*PhysData.c./grid.ω[2:end])
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
        k2[2:end] .= (nfun.(grid.ω[2:end]; z=z) .* grid.ω[2:end] ./ PhysData.c).^2
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
    n[2:end] = nfun.(2π*PhysData.c./grid.ω[2:end])
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
        k2[2:end] .= (nfun.(grid.ω[2:end]; z=z) .* grid.ω[2:end]./PhysData.c).^2
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
    βfun!(β, grid.ω, 0)
    α = similar(grid.ω)
    αfun!(α, grid.ω, 0)
    αlim!(α)
    linop = @. -im*(β-β1*grid.ω) - α/2
    linop[1] = 0
    return linop
end

function make_const_linop(grid::Grid.EnvGrid, βfun!, αfun!, β1, β0ref)
    β = similar(grid.ω)
    βfun!(β, grid.ω, 0)
    α = similar(grid.ω)
    αfun!(α, grid.ω, 0)
    αlim!(α)
    linop = -im.*(β .- β1.*(grid.ω .- grid.ω0) .- β0ref) .- α./2
    linop[.!grid.sidx] .= 0
    return linop
end

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
    function βfun!(out, ω, z)
        out .= βconst
    end
    αconst = zero(grid.ω)
    αconst[grid.sidx] = Modes.α.(mode, grid.ω[grid.sidx])
    function αfun!(out, ω, z)
        out .= αconst
    end
    make_const_linop(grid, βfun!, αfun!, β1const, β0const), βfun!, β1const, αfun!
end

function make_const_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    β1const = Modes.dispersion(mode, 1, wlfreq(λ0))
    βconst = zero(grid.ω)
    βconst[2:end] = Modes.β.(mode, grid.ω[2:end])
    βconst[1] = 1
    function βfun!(out, ω, z)
        out .= βconst
    end
    αconst = zero(grid.ω)
    αconst[2:end] = Modes.α.(mode, grid.ω[2:end])
    function αfun!(out, ω, z)
        out .= αconst
    end
    make_const_linop(grid, βfun!, αfun!, β1const), βfun!, β1const, αfun!
end

function make_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    function linop!(out, z)
        β1 = Modes.dispersion(mode, 1, wlfreq(λ0), z=z)::Float64
        for iω = 2:length(grid.ω)
            nc = conj_clamp(Modes.neff(mode, grid.ω[iω], z=z), grid.ω[iω])
            out[iω] = -im*(grid.ω[iω]/PhysData.c*nc - grid.ω[iω]*β1)
        end
        out[1] = 0
    end
    function βfun!(out, ω, z)
        out[2:end] .= Modes.β.(mode, ω[2:end], z=z)
        out[1] = 1.0
    end
    return linop!, βfun!
end

function make_linop(grid::Grid.EnvGrid, mode::Modes.AbstractMode, λ0; thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    function linop!(out, z)
        fill!(out, 0.0)
        β1 = Modes.dispersion(mode, 1, wlfreq(λ0), z=z)::Float64
        if !thg
            βref = Modes.β(mode, wlfreq(λ0), z=z)
        end
        for iω in sidcs
            nc = conj_clamp(Modes.neff(mode, grid.ω[iω], z=z), grid.ω[iω])
            out[iω] = -im*(grid.ω[iω]/PhysData.c*nc - (grid.ω[iω] - grid.ω0)*β1)
            if !thg
                out[iω] -= -im*βref
            end
        end
    end
    function βfun!(out, ω, z)
        fill!(out, 1.0)
        out[grid.sidx] .= Modes.β.(mode, ω[grid.sidx], z=z)
    end
    return linop!, βfun!
end

#=================================================#
#=================   MULTIMODE   =================#
#=================================================#

function make_const_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
    β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0))
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[2:end] = Modes.β.(modes[i], grid.ω[2:end])
        βconst[1] = 1
        α = zeros(length(grid.ω))
        α[2:end] .= Modes.α.(modes[i], grid.ω[2:end])
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

function make_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
    function linop!(out, z)
        β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0), z=z)::Float64
        fill!(out, 0.0)
        for i in eachindex(modes)
            for iω = 2:length(grid.ω)
                nc = conj_clamp(Modes.neff(modes[i], grid.ω[iω], z=z), grid.ω[iω])
                out[iω, i] = -im*(grid.ω[iω]/PhysData.c*nc - grid.ω[iω]*β1)
            end
        end
    end
end

function make_linop(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    function linop!(out, z)
        β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0), z=z)::Float64
        fill!(out, 0.0)
        if !thg
            βref = Modes.β(modes[ref_mode], wlfreq(λ0), z=z)
        end
        for i in eachindex(modes)
            for iω in sidcs
                nc = conj_clamp(Modes.neff(modes[i], grid.ω[iω], z=z), grid.ω[iω])
                out[iω, i] = -im*(grid.ω[iω]/PhysData.c*nc - (grid.ω[iω] - grid.ω0)*β1)
                if !thg
                    out[iω, i] -= -im*βref
                end
            end
        end
    end
end


end