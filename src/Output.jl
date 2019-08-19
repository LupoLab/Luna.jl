module Output
import HDF5
import Logging

"Output handler for writing only to memory"
mutable struct MemoryOutput{sT, N}
    save_cond::sT
    ydims::NTuple{N, Int64}  # Dimensions of one array to be saved
    yname::AbstractString  # Name for solution (e.g. "Eω")
    tname::AbstractString  # Name for propagation direction (e.g. "z")
    saved::Integer  # How many points have been saved so far
    data::Dict{String, Any}  # The actual data
end

function MemoryOutput(tmin, tmax, saveN::Integer, ydims;
                      yname="Eω", tname="z")
    save_cond = GridCondition(tmin, tmax, saveN)
    MemoryOutput(save_cond, ydims, yname, tname)
end

function MemoryOutput(save_cond, ydims, yname, tname)
    dims = init_dims(ydims, save_cond)
    data = Dict{String, Any}()
    data[yname] = Array{ComplexF64}(undef, dims)
    data[tname] = Array{Float64}(undef, (dims[end],))
    MemoryOutput(save_cond, ydims, yname, tname, 0, data)
end

"""Calling the output handler saves data in the arrays
    Arguments:
        y: current function value
        t: current propagation point
        dt: current stepsize
        yfun: callable which returns interpolated function value at different t
    Note that from RK45.jl, this will be called with yn and tn as arguments.
"""
function (o::MemoryOutput)(y, t, dt, yfun)
    save, ts = o.save_cond(y, t, dt, o.saved)
    while save
        s = size(o.data[o.yname])
        if s[end] < o.saved+1
            o.data[o.yname] = cat(o.data[o.yname], yfun(ts),
                                   dims=ndims(o.data[o.yname]))
            o.data[o.tname] = vcat(o.data[o.tname], ts)
        else
            idcs = fill(:, length(o.ydims))
            o.data[o.yname][idcs..., o.saved+1] = yfun(ts)
            o.data[o.tname][o.saved+1] = ts
        end
        o.saved += 1
        save, ts = o.save_cond(y, t, dt, o.saved)
    end

end

"Calling the output on a dictionary simply writes the items to the file"
function (o::MemoryOutput)(d::Dict)
    for (k, v) in pairs(d)
        if haskey(o.data, k)
            error("Key $k already present in dataset!")
        end
        o.data[k] = v
    end
end

"Output handler for writing to an HDF5 file"
mutable struct HDF5Output{sT, N, S}
    fpath::AbstractString  # Path to output file
    save_cond::sT  # callable, determines when data is saved and where it is interpolated
    ydims::NTuple{N, Int64}  # Dimensions of one array to be saved
    yname::AbstractString  # Name for solution (e.g. "Eω")
    tname::AbstractString  # Name for propagation direction (e.g. "z")
    saved::Integer  # How many points have been saved so far
    statsfun::S  # Callable, returns dictionary of statistics
    stats_tmp::Vector{Dict{String, Any}}  # Temporary storage for statistics between saves
end

"Simple constructor"
function HDF5Output(fpath, tmin, tmax, saveN::Integer, ydims, statsfun;
                    yname="Eω", tname="z")
    save_cond = GridCondition(tmin, tmax, saveN)
    HDF5Output(fpath, save_cond, ydims, yname, tname, statsfun)
end

"Internal constructor - creates datasets in the file"
function HDF5Output(fpath, save_cond, ydims, yname, tname, statsfun)
    idims = init_dims(ydims, save_cond)
    cdims = collect(idims)
    cdims[1] *= 2 # Allow for interleaving of real, imag, real, imag...
    dims = Tuple(cdims)
    mdims = copy(cdims)
    mdims[end] = -1
    maxdims = Tuple(mdims)
    if isfile(fpath)
        Logging.@warn("Output file $(fpath) already exists and will be overwritten!")
        rm(fpath)
    end
    HDF5.h5open(fpath, "cw") do file
        HDF5.d_create(file, yname, HDF5.datatype(Float64), (dims, maxdims),
                      "chunk", dims)
        HDF5.d_create(file, tname, HDF5.datatype(Float64), ((dims[end],), (-1,)),
                      "chunk", (1,))
        HDF5.g_create(file, "stats")
    end
    stats0 = Vector{Dict{String, Any}}()
    HDF5Output(fpath, save_cond, ydims, yname, tname, 0, statsfun, stats0)
end

"""Calling the output handler writes data to the file
    Arguments:
        y: current function value
        t: current propagation point
        dt: current stepsize
        yfun: callable which returns interpolated function value at different t
    Note that from RK45.jl, this will be called with yn and tn as arguments.
"""
function (o::HDF5Output)(y, t, dt, yfun)
    save, ts = o.save_cond(y, t, dt, o.saved)
    o.stats_tmp = vcat(o.stats_tmp, o.statsfun(y, t, dt))
    if save
        HDF5.h5open(o.fpath, "r+") do file
            while save
                idcs = fill(:, length(o.ydims))
                s = collect(size(file[o.yname]))
                if s[end] < o.saved+1
                    s[end] += 1
                    HDF5.set_dims!(file[o.yname], Tuple(s))
                end
                file[o.yname][idcs..., o.saved+1] = reinterpret(Float64, yfun(ts))
                s = collect(size(file[o.tname]))
                if s[end] < o.saved+1
                    s[end] += 1
                    HDF5.set_dims!(file[o.tname], Tuple(s))
                end
                file[o.tname][o.saved+1] = ts
                o.saved += 1
                save, ts = o.save_cond(y, t, dt, o.saved)
            end
            append_datasets(file["stats"], o.stats_tmp)
            o.stats_tmp = Vector{Dict{String, Any}}()
        end
    end

end

function append_datasets(parent, a::Array{Dict{String,Any},1})
    N = length(a)
    names = HDF5.names(parent)
    for (k, v) in pairs(a[1])
        if ~(k in names)
            create_dataset(parent, k, v)
        end
        s = collect(size(parent[k]))
        curN = s[end]
        if ~(k in names)
            curN -= 1 # new dataset - overwrite initial value
        end
        s[end] += N
        if ~(k in names)
            s[end] -= 1 # new dataset - overwrite initial value
        end
        HDF5.set_dims!(parent[k], Tuple(s))
        for ii = 1:N
            parent[k][curN+ii] = a[ii][k]
        end
    end
end

function create_dataset(parent, name, x::Number)
    HDF5.d_create(parent, name, HDF5.datatype(typeof(x)), ((1,), (-1,)),
                  "chunk", (1,))
end

function create_dataset(parent, name, x::AbstractArray)
    dims = (size(x)..., 1)
    maxdims = (size(x)..., -1)
    HDF5.d_create(parent, name, HDF5.datatype(eltype(x)), (dims, maxdims),
                  "chunk", dims)

end

"Calling the output on a dictionary simply writes the items to the file"
function (o::HDF5Output)(d::Dict)
    HDF5.h5open(o.fpath, "r+") do file
        for (k, v) in pairs(d)
            if HDF5.exists(file, k)
                error("File $(o.fpath) already has dataset $(k)!")
            end
            file[k] = v
        end
    end
end

"Condition callable that distributes save points evenly on a grid"
struct GridCondition
    grid::Vector{Float64}
    saveN::Integer
end

function GridCondition(tmin, tmax, saveN)
    GridCondition(range(tmin, stop=tmax, length=saveN), saveN)
end

function (cond::GridCondition)(y, t, dt, saved)
    save = (saved < cond.saveN) && cond.grid[saved+1] <= t
    return save, save ? cond.grid[saved+1] : 0
end

"Condition which saves every native point of the propagation"
function always(y, t, dt, saved)
    return true, t
end

"Condition which saves every nth native point"
function every_nth(n)
    i = 0
    cond = let i = i, n = n
        function condition(y, t, dt, saved)
            return i % n == 0, t
            i += 1
        end
    end
    return cond
end

"""Making initial array dimensions.
For a GridCondition, we know in advance how many points there will be.
"""
function init_dims(ydims, save_cond::GridCondition)
    return (ydims..., save_cond.saveN)
end

"For other conditions, we do not know in advance."
function init_dims(ydims, save_cond)
    return (ydims..., 1)
end
            
end
