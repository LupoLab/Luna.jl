module Luna
import FFTW
import NumericalIntegration
import Logging
import Printf: @sprintf
import LinearAlgebra: mul!, ldiv!
include("Maths.jl")
include("Hankel.jl")
include("PhysData.jl")
include("Grid.jl")
include("RK45.jl")
include("Modes.jl")
include("Capillary.jl")
include("Nonlinear.jl")
include("Ionisation.jl")
include("NonlinearRHS.jl")
include("Output.jl")
include("Stats.jl")

function make_linop(grid::Grid.RealGrid, βfun, αfun, frame_vel)
    β = .-βfun(grid.ω, 0)
    α = αfun(grid.ω, 0)
    β1 = -1/frame_vel(0)
    return @. im*(β-β1*grid.ω) - α/2
end

function make_linop(grid::Grid.EnvGrid, βfun, αfun, frame_vel; thg=false)
    β0ref = thg ? 0.0 : βfun(grid.ω0)
    β = βfun(grid.ω, 0)
    α = αfun(grid.ω, 0)
    β1 = 1/frame_vel(0)
    return -im.*(β .- β1.*(grid.ω .- grid.ω0) .- β0ref) - α/2
end

function make_const_linop(grid::Grid.EnvGrid, mode::T, λ0; thg=false) where T <: Modes.AbstractMode
    β1const = Modes.dispersion(mode, 1, λ=λ0)
    β0const = Modes.β(mode, λ=λ0)
    βconst = zero(grid.ω)
    βconst[grid.sidx] = Modes.β(mode, grid.ω[grid.sidx])
    βconst[.!grid.sidx] .= 1
    βfun(ω, z) = βconst
    frame_vel(z) = 1/β1const
    αfun(ω, z) = 0.0 # TODO deal with loss properly
    make_linop(grid, βfun, αfun, frame_vel, thg=thg), βfun, frame_vel, αfun
end

function make_const_linop(grid::Grid.RealGrid, mode::T, λ0) where T <: Modes.AbstractMode
    β1const = Modes.dispersion(mode, 1; λ=λ0)
    βconst = zero(grid.ω)
    βconst[2:end] = Modes.β(mode, grid.ω[2:end])
    βconst[1] = 1
    βfun(ω, z) = βconst
    frame_vel(z) = 1/β1const
    αfun(ω, z) = 0.0 # TODO deal with loss properly
    make_linop(grid, βfun, αfun, frame_vel), βfun, frame_vel, αfun
end

function make_const_linop(grid::Grid.RealGrid, modes::Tuple, λ0; ref_mode=1)
    vel = 1/Modes.dispersion(modes[ref_mode], 1, λ=λ0)
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[2:end] = Modes.β(modes[i], grid.ω[2:end])
        βconst[1] = 1
        α = 0.0 # TODO deal with loss properly
        linops[:,i] = im.*(-βconst .+ grid.ω./vel) .- α./2
    end
    linops
end

function make_const_linop(grid::Grid.EnvGrid, modes::Tuple, λ0; ref_mode=1, thg=false)
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
        βconst[grid.sidx] = Modes.β(modes[i], grid.ω[grid.sidx])
        βconst[.!grid.sidx] .= 1
        α = 0.0 # TODO deal with loss properly
        linops[:,i] = -im.*(βconst .- (grid.ω .- grid.ω0)./vel .- βref) .- α./2
    end
    linops
end

function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs)
    xo1 = Array{Float64}(undef, length(grid.to))
    FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=FFTW.PATIENT)
    Eω = make_init(grid, inputs, energyfun, FT)
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs)
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=FFTW.MEASURE)
    xo1 = Array{ComplexF64}(undef, length(grid.to))
    FTo1 = FFTW.plan_fft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun)
    Eω = make_init(grid, inputs, energyfun, FT)
    Eω, transform, FT
end

function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs,
               modes, components; full=false) where T
    Exys = []
    for mode in modes
        push!(Exys, Modes.Exy(mode))
    end
    if components == :Exy
        npol = 2
    else
        npol = 1
    end
    xt = Array{Float64}(undef, length(grid.t))
    FTt = FFTW.plan_rfft(xt, 1, flags=FFTW.MEASURE)
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    Eω[:,1] .= make_init(grid, inputs, energyfun, FTt)
    x = Array{Float64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_rfft(x, 1, flags=FFTW.MEASURE)
    xo1 = Array{Float64}(undef, length(grid.to), npol)
    FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransModal(grid, Modes.dimlimits(modes[1]), Exys, FTo1,
                                 responses, densityfun, components, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs,
               modes, components; full=false) where T
    Exys = []
    for mode in modes
        push!(Exys, Modes.Exy(mode))
    end
    if components == :Exy
        npol = 2
    else
        npol = 1
    end
    xt = Array{ComplexF64}(undef, length(grid.t))
    FTt = FFTW.plan_fft(xt, 1, flags=FFTW.MEASURE)
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    Eω[:,1] .= make_init(grid, inputs, energyfun, FTt)
    x = Array{ComplexF64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_fft(x, 1, flags=FFTW.MEASURE)
    xo1 = Array{ComplexF64}(undef, length(grid.to), npol)
    FTo1 = FFTW.plan_fft(xo1, 1, flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransModal(grid, Modes.dimlimits(modes[1]), Exys, FTo1,
                                 responses, densityfun, components, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    Eω, transform, FT
end

function make_init(grid, inputs, energyfun, FT)
    out = fill(0.0 + 0.0im, length(grid.ω))
    for input in inputs
        out .+= scaled_input(grid, input, energyfun, FT)
    end
    return out
end

function scaled_input(grid, input, energyfun, FT)
    Et = input.func(grid.t)
    energy = energyfun(grid.t, Et)
    Et_sc = sqrt(input.energy)/sqrt(energy) .* Et
    return FT * Et_sc
end

function run(Eω, grid,
             linop, transform, FT, output; max_dz=Inf)


    Et = FT \ Eω

    z = 0
    dz = 1e-3

    window! = let window=grid.ωwin, twindow=grid.twin, FT=FT, Et=Et
        function window!(Eω)
            Eω .*= window
            ldiv!(Et, FT, Eω)
            Et .*= twindow
            mul!(Eω, FT, Et)
        end
    end

    function stepfun(Eω, z, dz, interpolant)
        window!(Eω)
        output(Eω, z, dz, interpolant)
    end

    RK45.solve_precon(
        transform, linop, Eω, z, dz, grid.zmax, stepfun=stepfun, max_dt=max_dz)
end

end # module
