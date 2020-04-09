import Luna.Simple: solveprop, plotprop

solution = solveprop(radius=13e-6, gas=:Ar, pressure=5.0, flength=15e-2, loss=false,
                     τfwhm=30e-15, λ0=800e-9, energy=1e-6, shape=:gaussian,
                     minλ=160e-9, maxλ=3000e-9, τgrid=1e-12,
                     thg=false)

plotprop(solution)