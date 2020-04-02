module Stats
import Luna: Maths, Grid, Modes, Utils
import FFTW
import LinearAlgebra: mul!

function ω0(grid)
    addstat! = let ω=grid.ω
        function addstat!(d, Eω, Et, z, dz)
            d["ω0"] = Maths.moment(ω, abs2.(Eω))
        end
    end
    return addstat!
end

function energy(grid, energyfun_ω)
    function addstat!(d, Eω, Et, z, dz)
        d["energy"] = energyfun_ω(grid.ω, Eω)
    end
    return addstat!
end

function energy(grid, energyfun_ω, N)
    function addstat!(d, Eω, Et, z, dz)
        d["energy"] = [energyfun_ω(grid.ω, Eω[:, i]) for i=1:N]
    end
    return addstat!
end

function peakpower(grid)
    function addstat!(d, Eω, Et, z, dz)
        d["peakpower"] = maximum(abs2.(Et))
    end
    return addstat!
end

function peakpower(grid, N)
    function addstat!(d, Eω, Et, z, dz)
        d["peakpower"] = [maximum(abs2.(Et[:, i])) for i=1:N]
    end
    return addstat!
end

function electrondensity(Presp, dfun)
    function addstat!(d, Eω, Et, z, dz)
        d["electrondensity"] = maximum(Presp.fraction)*dfun(z)
    end
end

function density(dfun)
    function addstat!(d, Eω, Et, z, dz)
        d["density"] = dfun(z)
    end
end

function zdz!(d, Eω, Et, z, dz)
    d["z"] = z
    d["dz"] = dz
end

"""
    plan_analytic(grid, Eω)

Plan a transform from the frequency-domain field `Eω` to the analytic time-domain field.

Returns both a buffer for the analytic field and a closure to do the transform.
"""
function plan_analytic(grid::Grid.EnvGrid, Eω)
    Eta = similar(Eω)
    Utils.loadFFTwisdom()
    iFT = FFTW.plan_ifft(Eω, 1, flags=FFTW.PATIENT)
    Utils.saveFFTwisdom()
    function analytic!(Eta, Eω)
        mul!(Eta, iFT, Eω) # for envelope fields, we only need to do the inverse transform
    end
    return Eta, analytic!
end

function plan_analytic(grid::Grid.RealGrid, Eω)
    s = collect(size(Eω))
    s[1] = (length(grid.ω) - 1)*2 # e.g. for 4097 rFFT samples, we need 8192 FFT samples
    Eta = Array{ComplexF64, ndims(Eω)}(undef, Tuple(s))
    Eωa = zero(Eta)
    idxhi = CartesianIndices(size(Eω)[2:end]) # index over all other dimensions
    Utils.loadFFTwisdom()
    iFT = FFTW.plan_ifft(Eωa, 1, flags=FFTW.PATIENT)
    Utils.saveFFTwisdom()
    function analytic!(Eta, Eω)
        copyto_fft!(Eωa, Eω, idxhi) # copy across to FFT-sampled buffer
        mul!(Eta, iFT, Eωa) # now do the inverse transform
    end
    return Eta, analytic!
end

"""
    copyto_fft!(Eωa, Eω, idxhi)

Copy the rFFT-sampled field `Eω` to the FFT-sampled buffer `Eωa`, ready for inverse FFT
"""
function copyto_fft!(Eωa, Eω, idxhi)
    n = size(Eω, 1)-1 # rFFT has sample at +fs/2, but FFT does not (only at -fs/2)
    for idx in idxhi
        for i in 1:n
            Eωa[i, idx] = 2*Eω[i, idx]
        end
    end
end

"""
    collect_stats(grid, Eω, funcs...)

Create a closure which collects statistics from the individual functions in `funcs`.

Each function given will be called with the arguments `(d, Eω, Et, z, dz)`, where
- d -> dictionary to store statistics values. each `func` should **mutate** this
- Eω -> frequency-domain field
- Et -> analytic time-domain field
- z -> current propagation distance
- z -> current stepsize
"""
function collect_stats(grid, Eω, funcs...)
    # make sure z and dz are recorded
    if !(zdz! in funcs)
        funcs = (funcs..., zdz!)
    end
    Et, analytic! = plan_analytic(grid, Eω)
    f = let funcs=funcs
        function collect_stats(Eω, z, dz)
            d = Dict{String, Any}()
            analytic!(Et, Eω)
            for func in funcs
                func(d, Eω, Et, z, dz)
            end
            return d
        end
    end
    return f
end

end