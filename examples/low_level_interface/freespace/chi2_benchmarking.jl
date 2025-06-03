using Luna
using BenchmarkTools
import Rotations: RotZY, RotYZ, RotMatrix, RotMatrix3
import LinearAlgebra: mul!
using Metal

θ = deg2rad(23.3717)
ϕ = deg2rad(30)
material = :BBO

Et = rand(2^12, 2)
out = zero(Et)

c = Nonlinear.Chi2Field(θ, ϕ, PhysData.χ2(material))
# r = Nonlinear.Kerr_field(PhysData.χ3(material))

function toprofile(n)
    for i in 1:n
        c(out, Et, 1)
    end
end
##
@profview toprofile(1)
@profview toprofile(100)
@btime c(out, Et, 1)

##
Enl2 = rand(2^12, 6)'
Et2 = rand(2^12, 3)'
##
Et2_m = MtlArray(convert(Matrix{Float32}, Et2))
χ2_m = MtlArray(convert(Matrix{Float32}, c.χ2_toLab))
Enl2_m = MtlArray(convert(Matrix{Float32}, Enl2))
@btime mul!(Et2_m, χ2_m, Enl2_m)

##
@btime mul!(Et2, c.χ2_toLab, Enl2)
