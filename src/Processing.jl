module Processing
import FFTW
import Luna: Maths
import Luna.PhysData: wlfreq
import Luna.Grid: RealGrid, EnvGrid

function arrivaltime(grid, Eω; λlims, winwidth=0, kwargs...)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    arrivaltime(grid, Eω, window; kwargs...)
end

function arrivaltime(grid::RealGrid, Eω, ωwindow::Vector{<:Real}; method=:moment, oversampling=1)
    Et = FFTW.irfft(Eω .* ωwindow, length(grid.t), 1)
    to, Eto = Maths.oversample(grid.t, Et)
    arrivaltime(to, abs2.(Maths.hilbert(Eto)); method=method)
end

function arrivaltime(grid::EnvGrid, Eω, ωwindow::Vector{<:Real}; method=:moment, oversampling=1)
    Et = FFTW.ifft(Eω .* ωwindow, 1)
    to, Eto = Maths.oversample(grid.t, Et)
    arrivaltime(to, abs2.(Eto); method=method)
end

function arrivaltime(t, It::Vector{<:Real}; method)
    if method == :moment
        Maths.moment(t, It)
    elseif method == :peak
        t[argmax(It)]
    else
        error("Unknown arrival time method $method")
    end
end

function arrivaltime(t, It::Array{<:Real, N}; method) where N
    if method == :moment
        dropdims(Maths.moment(t, It; dim=1); dims=1)
    elseif method == :peak
        out = Array{Float64, ndims(It)-1}(undef, size(It)[2:end])
        cidcs = CartesianIndices(size(It)[2:end])
        for ii in cidcs
            out[ii] = arrivaltime(t, It[:, ii]; method=:peak)
        end
        out
    else
        error("Unknown arrival time method $method")
    end
end

end