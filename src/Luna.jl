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
include("AbstractModes.jl")
include("Capillary.jl")
include("Nonlinear.jl")
include("Ionisation.jl")
include("Modes.jl")
include("Output.jl")
include("Stats.jl")


function make_linop(grid, βfun, αfun, frame_vel)
    β = .-βfun(grid.ω, 1, 1, 0)
    α = αfun(grid.ω, 1, 1, 0)
    β1 = -1/frame_vel(0)
    return @. im*(β-β1*grid.ω) - α/2
end

function make_fnl(grid, transform, densityfun, normfun, responses)
    Pω! = transform
    function fnl!(nl, Eω, z)
        Pω!(nl, Eω, z, responses)
        nl .*= grid.ωwin.*densityfun(z).*(-im.*grid.ω./2)./normfun(z)
    end
    return fnl!
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
    energy = energyfun(grid.t, Et, input.m, input.n)
    Et_sc = sqrt(input.energy)/sqrt(energy) .* Et
    return FT * Et_sc
end

function run(grid,
             linop, normfun, energyfun, densityfun, inputs, responses,
             transform, FT, output; max_dz=Inf)

    Eω = make_init(grid, inputs, energyfun, FT)
    Et = FT \ Eω

    fnl! = make_fnl(grid, transform, densityfun, normfun, responses)

    z = 0
    dz = 1e-3
    zmax = grid.zmax
    saveN = 201

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

    zout, Eout, steps = RK45.solve_precon(
        fnl!, linop, Eω, z, dz, zmax, saveN, stepfun=stepfun, max_dt=max_dz)

    return zout, Eout
end

end # module
