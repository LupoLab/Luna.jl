module LinearOps
import FFTW
import Luna: Modes, Grid, PhysData
import Luna.PhysData: change

function make_const_linop(grid::Grid.RealGrid, βfun, αfun, frame_vel)
    β = similar(grid.ω)
    βfun(β, grid.ω, 0)
    β .*= -1
    # β = .-βfun(grid.ω, 0)
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

function make_const_linop(grid::Grid.EnvGrid, mode::Modes.AbstractMode, λ0; thg=false)
    β1const = Modes.dispersion(mode, 1, change(λ0))
    if thg
        β0const = 0.0
    else
        β0const = Modes.β(mode, change(λ0))
    end
    βconst = zero(grid.ω)
    βconst[grid.sidx] = Modes.β.(mode, grid.ω[grid.sidx])
    βconst[.!grid.sidx] .= 1
    βfun(ω, z) = βconst
    frame_vel(z) = 1/β1const
    αfun(ω, z) = Modes.α.(mode, ω)
    make_const_linop(grid, βfun, αfun, frame_vel, β0const), βfun, frame_vel, αfun
end

function make_const_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    β1const = Modes.dispersion(mode, 1, change(λ0))
    βconst = zero(grid.ω)
    βconst[2:end] = Modes.β.(mode, grid.ω[2:end])
    βconst[1] = 1
    function βfun!(out, ω, z)
        out .= βconst
    end
    frame_vel(z) = 1/β1const
    αfun(ω, z) = Modes.α.(mode, ω)
    make_const_linop(grid, βfun!, αfun, frame_vel), βfun!, frame_vel, αfun
end

function make_const_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
    vel = 1/Modes.dispersion(modes[ref_mode], 1, change(λ0))
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[2:end] = Modes.β.(modes[i], grid.ω[2:end])
        βconst[1] = 1
        α = Modes.α.(modes[i], grid.ω)
        linops[:,i] = im.*(-βconst .+ grid.ω./vel) .- α./2
    end
    linops
end

function make_const_linop(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false)
    vel = 1/Modes.dispersion(modes[ref_mode], 1, change(λ0))
    if thg
        βref = 0.0
    else
        βref = Modes.β(modes[ref_mode], change(λ0))
    end
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[grid.sidx] = Modes.β.(modes[i], grid.ω[grid.sidx])
        βconst[.!grid.sidx] .= 1
        α = Modes.α.(modes[i], grid.ω)
        linops[:,i] = -im.*(βconst .- (grid.ω .- grid.ω0)./vel .- βref) .- α./2
    end
    linops
end

function make_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    function linop!(out, z)
        out .= -im.*grid.ω./PhysData.c.*Modes.neff.(mode, grid.ω, z=z)
        out[1] = 1
        out .-= -im.*grid.ω.*Modes.dispersion(mode, 1, change(λ0), z=z)
    end
    function βfun!(out, ω, z)
        out .= Modes.β.(mode, ω, z=z)
        out[1] = 1
    end
    return linop!, βfun!
end

function make_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
    function linop!(out, z)
        β1 = Modes.dispersion(modes[ref_mode], 1, change(λ0), z=z)
        nmodes = length(modes)
        for i = 1:nmodes
            out[2:end, i] .= -im.*(
                grid.ω[2:end]./PhysData.c.*Modes.neff.(modes[i], grid.ω[2:end], z=z)
                .- grid.ω[2:end] .* β1
                )
            out[1, i] = 1
        end
    end
end


end