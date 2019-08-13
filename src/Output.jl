module Output
import HDF5

mutable struct HDF5Output{sT, N}
    fpath::AbstractString  # Path to output file
    save_cond::sT  # Condition callable for saving
    ydims::NTuple{N, Int64}  # Dimensions of one array to be saved
    yname::AbstractString  # Name for solution (e.g. "Eω")
    tname::AbstractString  # Name for propagation direction (e.g. "z")
    saved::Integer  # How many points have been saved so far
end

function HDF5Output(fpath, tmin, tmax, saveN::Integer, ydims;
                    yname="Eω", tname="z")
    save_cond = GridCondition(tmin, tmax, saveN)
    HDF5Output(fpath, save_cond, ydims, yname, tname)
end

function HDF5Output(fpath, save_cond, ydims, yname, tname)
    dims = (ydims..., 1)
    maxdims = (ydims..., -1)
    HDF5.h5open(fpath, "cw") do file
        HDF5.d_create(file, yname*"_real", HDF5.datatype(Float64), (dims, maxdims),
                      "chunk", dims)
        HDF5.d_create(file, yname*"_imag", HDF5.datatype(Float64), (dims, maxdims),
                      "chunk", dims)
        HDF5.d_create(file, tname, HDF5.datatype(Float64), ((1,), (-1,)),
                      "chunk", (1,))
    end
    HDF5Output(fpath, save_cond, ydims, yname, tname, 0)
end

function (o::HDF5Output)(y, t, dt, yfun)
    save, ts = o.save_cond(y, t, dt, o.saved)
    while save
        HDF5.h5open(o.fpath, "r+") do file
            idcs = fill(:, length(o.ydims))
            s = collect(size(file[o.yname*"_real"]))
            if s[end] < o.saved+1
                s[end] += 1
                HDF5.set_dims!(file[o.yname*"_real"], Tuple(s))
                HDF5.set_dims!(file[o.yname*"_imag"], Tuple(s))
            end
            file[o.yname*"_real"][idcs..., o.saved+1] = real(yfun(ts))
            file[o.yname*"_imag"][idcs..., o.saved+1] = imag(yfun(ts))
            s = collect(size(file[o.tname]))
            if s[end] < o.saved+1
                s[end] += 1
                HDF5.set_dims!(file[o.tname], Tuple(s))
            end
            file[o.tname][o.saved+1] = ts
            o.saved += 1
        end
        save, ts = o.save_cond(y, t, dt, o.saved)
    end

end

struct GridCondition
    grid::Vector{Float64}
    saveN::Integer
end

function GridCondition(tmin, tmax, saveN)
    GridCondition(range(tmin, stop=tmax, length=saveN), saveN)
end

function (cond::GridCondition)(y, t, dt, saved)
    save = (saved < cond.saveN) && cond.grid[saved+1] < t
    return save, cond.grid[saved+1]
end

end
