module LinearOps
import FFTW
import Luna: Modes, Grid, PhysData, Hankel, Maths

function make_const_linop(grid::Grid.RealGrid, x::AbstractArray, y::AbstractArray,
                          n::AbstractArray, frame_vel::Number)
    kx = reshape(Maths.fftfreq(x), (1, 1, length(x)));
    ky = reshape(Maths.fftfreq(x), (1, length(y)));
    βsq = @. (n*grid.ω/PhysData.c)^2 - kx^2 - ky^2
    βsq[βsq .< 0] .= 0
    β = .-sqrt.(βsq)
    β1 = -1/frame_vel
    return @. im*(β-β1*grid.ω) 
end

function make_const_linop(grid::Grid.RealGrid, x::AbstractArray, y::AbstractArray, nfun)
    n = zero(grid.ω)
    n[2:end] = nfun.(2π*PhysData.c./grid.ω[2:end])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    make_const_linop(grid, x, y, n, 1/β1)
end

function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT,
                          n::AbstractArray, frame_vel::Number)
    βsq = (n.*grid.ω./PhysData.c).^2 .- (q.k.^2)'
    βsq[βsq .< 0] .= 0
    β = .-sqrt.(βsq)
    β1 = -1/frame_vel
    # TODO loss
    return @. im*(β-β1*grid.ω)
end

function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT, nfun)
    n = zero(grid.ω)
    n[2:end] = nfun.(2π*PhysData.c./grid.ω[2:end])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    make_const_linop(grid, q, n, 1/β1)
end

function make_const_linop(grid::Grid.RealGrid, βfun, αfun, frame_vel)
    β = .-βfun(grid.ω, 0)
    α = αfun(grid.ω, 0)
    β1 = -1/frame_vel(0)
    return @. im*(β-β1*grid.ω) - α/2
end

function make_const_linop(grid::Grid.EnvGrid, βfun, αfun, frame_vel, β0ref)
    β = βfun(grid.ω, 0)
    α = αfun(grid.ω, 0)
    β1 = 1/frame_vel(0)
    return -im.*(β .- β1.*(grid.ω .- grid.ω0) .- β0ref) .- α./2
end

function make_const_linop(grid::Grid.EnvGrid, mode::T, λ0; thg=false) where T <: Modes.AbstractMode
    β1const = Modes.dispersion(mode, 1, λ=λ0)
    if thg
        β0const = 0.0
    else
        β0const = Modes.β(mode, λ=λ0)
    end
    βconst = zero(grid.ω)
    βconst[grid.sidx] = Modes.β.(mode, grid.ω[grid.sidx])
    βconst[.!grid.sidx] .= 1
    βfun(ω, z) = βconst
    frame_vel(z) = 1/β1const
    αfun(ω, z) = 0.0 # TODO deal with loss properly
    make_const_linop(grid, βfun, αfun, frame_vel, β0const), βfun, frame_vel, αfun
end

function make_const_linop(grid::Grid.RealGrid, mode::T, λ0) where T <: Modes.AbstractMode
    β1const = Modes.dispersion(mode, 1; λ=λ0)
    βconst = zero(grid.ω)
    βconst[2:end] = Modes.β.(mode, grid.ω[2:end])
    βconst[1] = 1
    βfun(ω, z) = βconst
    frame_vel(z) = 1/β1const
    αfun(ω, z) = 0.0 # TODO deal with loss properly
    make_const_linop(grid, βfun, αfun, frame_vel), βfun, frame_vel, αfun
end

function make_const_linop(grid::Grid.RealGrid, modes, λ0::Number; ref_mode=1)
    vel = 1/Modes.dispersion(modes[ref_mode], 1, λ=λ0)
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[2:end] = Modes.β.(modes[i], grid.ω[2:end])
        βconst[1] = 1
        α = 0.0 # TODO deal with loss properly
        linops[:,i] = im.*(-βconst .+ grid.ω./vel) .- α./2
    end
    linops
end

function make_const_linop(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false)
    vel = 1/Modes.dispersion(modes[ref_mode], 1, λ=λ0)
    if thg
        βref = 0.0
    else
        βref = Modes.β(modes[ref_mode], λ=λ0)
    end
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[grid.sidx] = Modes.β.(modes[i], grid.ω[grid.sidx])
        βconst[.!grid.sidx] .= 1
        α = 0.0 # TODO deal with loss properly
        linops[:,i] = -im.*(βconst .- (grid.ω .- grid.ω0)./vel .- βref) .- α./2
    end
    linops
end

end