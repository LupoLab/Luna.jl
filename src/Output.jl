module Output
import HDF5
using H5Zblosc
import Logging
import Base: getindex, show, haskey
using EllipsisNotation
import EllipsisNotation: Ellipsis
import Printf: @sprintf
import Luna: Scans, Utils
import FileWatching.Pidfile: mkpidlock

abstract type AbstractOutput end

"Output handler for writing only to memory"
mutable struct MemoryOutput{sT, S} <: AbstractOutput
    save_cond::sT
    yname::AbstractString  # Name for solution (e.g. "Eω")
    tname::AbstractString  # Name for propagation direction (e.g. "z")
    saved::Integer  # How many points have been saved so far
    data::Dict{String, Any}  # The actual data
    statsfun::S  # Callable, returns dictionary of statistics
end

function MemoryOutput(tmin, tmax, saveN::Integer, statsfun=nostats;
                      yname="Eω", tname="z", script=nothing)
    save_cond = GridCondition(tmin, tmax, saveN)
    MemoryOutput(save_cond, yname, tname, statsfun, script)
end

function MemoryOutput(save_cond, yname, tname, statsfun=nostats, script=nothing)
    data = Dict{String, Any}()
    data["stats"] = Dict{String, Any}()
    data["meta"] = Dict{String, Any}()
    data["meta"]["sourcecode"] = Utils.sourcecode()
    data["meta"]["git_commit"] = Utils.git_commit()
    if !isnothing(script)
        data["meta"]["script_code"] = script
    end
    MemoryOutput(save_cond, yname, tname, 0, data, statsfun)
end

function initialise(o::MemoryOutput, y)
    dims = init_dims(size(y), o.save_cond)
    o.data[o.yname] = Array{ComplexF64}(undef, dims)
    o.data[o.tname] = Array{Float64}(undef, (dims[end],))
end

"getindex works interchangeably so when switching from one Output to
another, subsequent code can stay the same"
getindex(o::MemoryOutput, ds::AbstractString) = o.data[ds]
getindex(o::MemoryOutput, ds::AbstractString, I...) = o.data[ds][I...]

show(io::IO, o::MemoryOutput) = print(io, "MemoryOutput$(collect(keys(o.data)))")

haskey(o::MemoryOutput, key) = haskey(o.data, key)

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
    append_stats!(o, o.statsfun(y, t, dt))
    !haskey(o.data, o.yname) && initialise(o, y)
    while save
        s = size(o.data[o.yname])
        if s[end] < o.saved+1
            o.data[o.yname] = fastcat(o.data[o.yname], yfun(ts))
            push!(o.data[o.tname], ts)
        else
            idcs = fill(:, ndims(y))
            o.data[o.yname][idcs..., o.saved+1] = yfun(ts)
            o.data[o.tname][o.saved+1] = ts
        end
        o.saved += 1
        save, ts = o.save_cond(y, t, dt, o.saved)
    end
end

function append_stats!(o::MemoryOutput, d)
    for (k, v) in pairs(d)
        append_stat!(o, k, v)
    end
end

function append_stat!(o::MemoryOutput, name, value::Number)
    if ~haskey(o.data["stats"], name)
        o.data["stats"][name] = [value]
    else
        push!(o.data["stats"][name], value)
    end
end

function append_stat!(o::MemoryOutput, name, value::AbstractArray)
    if ~haskey(o.data["stats"], name)
        dims = size(value)
        o.data["stats"][name] = reshape(value, (size(value)..., 1))
    else
        o.data["stats"][name] = fastcat(o.data["stats"][name], value)
    end
end

"Calling the output on a dictionary writes the items to the array"
function (o::MemoryOutput)(d::Dict; force=false, meta=false, group=nothing)
    for (k, v) in pairs(d)
        o(k, v; force=force, meta=meta, group=group)
    end
end

"Calling the output with a key, value pair writes the value to the array."
function (o::MemoryOutput)(key::AbstractString, val; force=false, meta=false, group=nothing)
    parent = meta ? o.data["meta"] : o.data
    if haskey(parent, key)
        if force
            Logging.@warn("Key $key already exists and will be overwritten.")
        else
            error("Key $key already present in dataset.")
        end
    end
    if !isnothing(group)
        if !haskey(parent, group)
            parent[group] = Dict{String, Any}()
        end
        parent[group][key] = val
    else
        parent[key] = val
    end
end

function tofile(fpath, o::MemoryOutput)
    Utils.save_dict_h5(fpath, o.data)
end

function fastcat(A, v)
    Av = vec(A)
    append!(Av, vec(v))
    dims = size(A)
    return reshape(Av, (dims[1:end-1]..., dims[end]+1))
end

"Output handler for writing to an HDF5 file"
mutable struct HDF5Output{sT, S} <: AbstractOutput
    fpath::AbstractString  # Path to output file
    save_cond::sT  # callable, determines when data is saved and where it is interpolated
    yname::AbstractString  # Name for solution (e.g. "Eω")
    tname::AbstractString  # Name for propagation direction (e.g. "z")
    saved::Integer  # How many points have been saved so far
    statsfun::S  # Callable, returns dictionary of statistics
    stats_tmp::Vector{Dict{String, Any}}  # Temporary storage for statistics between saves
    compression::Bool # whether to use compression
    cache::Bool # whether to cache latest solution point (for continuing after interrupt)
    cachehash::UInt64 # safety hash to prevent cache-continuing for different propagations
    readonly::Bool
end

"Simple constructor"
function HDF5Output(fpath, tmin, tmax, saveN::Integer, statsfun=nostats;
                    yname="Eω", tname="z", compression=false, script=nothing, cache=true,
                    readonly=false)
    save_cond = GridCondition(tmin, tmax, saveN)
    HDF5Output(fpath, save_cond, yname, tname, statsfun, compression, script, cache, readonly)
end

"Internal constructor - creates the file"
function HDF5Output(fpath, save_cond, yname, tname, statsfun, compression,
                    script=nothing, cache=true, readonly=false)
    if isfile(fpath) && cache
        HDF5.h5open(fpath, "cw") do file
            if HDF5.haskey(file["meta"], "cache")
                saved = read(file["meta"]["cache"]["saved"])
                chash = hash((sort(keys(file["stats"])), size(file[yname])[1:end-1]))
            else
                error("cached HDF5Output created, file exists, but has no cache")
            end
        end
    elseif !readonly
        if isfile(fpath)
            Logging.@warn("output file $(fpath) already exists and will be overwritten!")
            rm(fpath)
        end
        fdir, fname = splitdir(fpath)
        isdir(fdir) || mkpath(fdir)
        HDF5.h5open(fpath, "cw") do file
            HDF5.create_group(file, "stats")
            HDF5.create_group(file, "meta")
            file["meta"]["sourcecode"] = Utils.sourcecode()
            file["meta"]["git_commit"] = Utils.git_commit()
            if !isnothing(script)
                file["meta"]["script_code"] = script
            end
            if cache
                HDF5.create_group(file["meta"], "cache")
            end
        end
        chash = UInt64(0)
        saved = 0
    else
        chash = UInt64(0)
        saved = 0
    end
    stats0 = Vector{Dict{String, Any}}()
    HDF5Output(fpath, save_cond, yname, tname, saved, statsfun, stats0,
               compression, cache, chash, readonly)
end

function HDF5Output(fpath::AbstractString)
    isfile(fpath) || error("Cannot open read-only HDF5Output: file not found")
    HDF5Output(fpath, 0, 0, 1; readonly=true)
end

function initialise(o::HDF5Output, y)
    ydims = size(y)
    idims = init_dims(ydims, o.save_cond)
    cdims = collect(idims)
    dims = Tuple(cdims)
    chdims = (dims[1:end-1]..., 1) # Chunk size is that of one z-point
    mdims = copy(cdims)
    mdims[end] = -1
    maxdims = Tuple(mdims)
    HDF5.h5open(o.fpath, "r+") do file
        if o.compression
            HDF5.create_dataset(file, o.yname, HDF5.datatype(ComplexF64), (dims, maxdims),
                          chunk=chdims, blosc=3)
        else
            HDF5.create_dataset(file, o.yname, HDF5.datatype(ComplexF64), (dims, maxdims),
                          chunk=chdims)
        end
        HDF5.create_dataset(file, o.tname, HDF5.datatype(Float64), ((dims[end],), (-1,)),
                      chunk=(1,))
        statsnames = sort(collect(keys(o.stats_tmp[end])))
        o.cachehash = hash((statsnames, size(y)))
        file["meta"]["cachehash"] = o.cachehash
        if o.cache
            file["meta"]["cache"]["t"] = typemin(0.0)
            file["meta"]["cache"]["dt"] = typemin(0.0)
            file["meta"]["cache"]["y"] = y
            file["meta"]["cache"]["saved"] = 0
        end
    end
end

# for single String index, read whole data set
function getindex(o::HDF5Output, idx::AbstractString)
    HDF5.h5open(o.fpath, "r") do file
        read(file[idx])
    end
end

# more indices -> read slice of data
function getindex(o::HDF5Output, ds::AbstractString,
                  I::Union{AbstractRange, Int, Colon, Ellipsis}...)
    HDF5.h5open(o.fpath, "r") do file
        file[ds][to_indices(file[ds], I)...]
    end
end

# indexing with an array, e.g. o["Eω", :, [1, 2, 3]] has to be handled separately
function getindex(o::HDF5Output, ds::AbstractString,
                  I::Union{AbstractRange, Int, Colon, Array, Ellipsis}...)
    if count(isa.(I, Array)) > 1
        error("Only one dimension can be index with an array.")
    end
    HDF5.h5open(o.fpath, "r") do file
        dset = file[ds]
        idcs = to_indices(dset, I)
        adim = findfirst(isa.(idcs, Array)) # which of the indices is the array
        arr = idcs[adim] # the array itself
        T = eltype(dset)
        ret = Array{T}(undef, map(length, idcs))
        Ilo = idcs[1:adim-1]
        Ihi = idcs[adim+1:end]
        for ii in eachindex(arr)
            ret[Ilo..., ii, Ihi...] .= dset[Ilo..., arr[ii], Ihi...]
        end
        ret
    end
end

function show(io::IO, o::HDF5Output)
    if isfile(o.fpath)
        fields = HDF5.h5open(o.fpath) do file
            keys(file)
        end
        print(io, "HDF5Output$(fields)")
    else
        print(io, "HDF5Output[FILE DELETED]")
    end
end

function haskey(o::HDF5Output, key)
    if isfile(o.fpath)
        return HDF5.h5open(o.fpath) do file
            haskey(file, key)
        end
    else
        return false
    end
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
    o.readonly && error("Cannot add data to read-only output!")
    save, ts = o.save_cond(y, t, dt, o.saved)
    push!(o.stats_tmp, o.statsfun(y, t, dt))
    if save
        HDF5.h5open(o.fpath, "r+") do file
            !HDF5.haskey(file, o.yname) && initialise(o, y)
            statsnames = sort(collect(keys(o.stats_tmp[end])))
            cachehash = hash((statsnames, size(y)))
            cachehash == o.cachehash || error(
                "the hash for this propagation does not agree with cache in file")
            while save
                s = collect(size(file[o.yname]))
                idcs = fill(:, length(s)-1)
                if s[end] < o.saved+1
                    s[end] += 1
                    HDF5.set_extent_dims(file[o.yname], Tuple(s))
                end
                file[o.yname][idcs..., o.saved+1] = yfun(ts)
                s = collect(size(file[o.tname]))
                if s[end] < o.saved+1
                    s[end] += 1
                    HDF5.set_extent_dims(file[o.tname], Tuple(s))
                end
                file[o.tname][o.saved+1] = ts
                o.saved += 1
                save, ts = o.save_cond(y, t, dt, o.saved)
            end
            append_stats!(file["stats"], o.stats_tmp)
            o.stats_tmp = Vector{Dict{String, Any}}()
            if o.cache
                write(file["meta"]["cache"]["t"], t)
                write(file["meta"]["cache"]["dt"], dt)
                write(file["meta"]["cache"]["y"], y)
                write(file["meta"]["cache"]["saved"], o.saved)
            end
        end
    end
end

function append_stats!(parent, a::Array{Dict{String,Any},1})
    N = length(a)
    names = HDF5.keys(parent)
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
        HDF5.set_extent_dims(parent[k], Tuple(s))
        for ii = 1:N
            parent[k][fill(:, ndims(a[ii][k]))..., curN+ii] = a[ii][k]
        end
    end
end

function create_dataset(parent, name, x::Number)
    HDF5.create_dataset(parent, name, HDF5.datatype(typeof(x)), ((1,), (-1,)),
                  chunk=(1,))
end

function create_dataset(parent, name, x::AbstractArray)
    dims = (size(x)..., 1)
    maxdims = (size(x)..., -1)
    HDF5.create_dataset(parent, name, HDF5.datatype(eltype(x)), (dims, maxdims),
                  chunk=dims)
end

"Calling the output on a dictionary writes the items to the file"
function (o::HDF5Output)(d::AbstractDict; force=false, meta=false, group=nothing)
    o.readonly && error("Cannot add data to read-only output!")
    HDF5.h5open(o.fpath, "r+") do file
        parent = meta ? file["meta"] : file
        for (k, v) in pairs(d)
            if HDF5.haskey(parent, k)
                if force
                    Logging.@warn("Dataset $k already present in file $(o.fpath)"*
                                  " and will be overwritten")
                    HDF5.delete_object(parent, k)
                else
                    Logging.@warn("File $(o.fpath) already has dataset $k. Pass force=true"*
                              " to overwrite")
                end
            end
            isa(v, BitArray) && (v = Array{Bool, 1}(v))
            if !isnothing(group)
                if !HDF5.haskey(parent, group)
                    HDF5.create_group(parent, group)
                end
                if HDF5.haskey(parent[group], k)
                    write(parent[group][k], v)
                else
                    parent[group][k] = v
                end
            else
                if HDF5.haskey(parent, k)
                    write(parent[k], v)
                else
                    parent[k] = v
                end
            end
        end
    end
end

"Calling the output on a key, value pair writes the value to the file"
function (o::HDF5Output)(key::AbstractString, val; force=false, meta=false, group=nothing)
    o.readonly && error("Cannot add data to read-only output!")
    HDF5.h5open(o.fpath, "r+") do file
        parent = meta ? file["meta"] : file
        if HDF5.haskey(parent, key)
            if force
                Logging.@warn("Dataset $key already present in file $(o.fpath)"*
                              " and will be overwritten")
                HDF5.delete_object(parent, key)
            else
                Logging.@warn("File $(o.fpath) already has dataset $(key). Pass force=true"*
                              " to overwrite")
            end
        end
        isa(val, BitArray) && (val = Array{Bool, 1}(val))
        if !isnothing(group)
            if !HDF5.haskey(parent, group)
                HDF5.create_group(parent, group)
            end
            if HDF5.haskey(parent[group], key)
                write(parent[group][key], val)
            else
                parent[group][key] = val
            end
        else
            if HDF5.haskey(parent, key)
                write(parent[key], val)
            else
                parent[key] = val
            end
        end
    end
end

"""
    check_cache(o::HDF5Output, y, t, dt)

Check for an existing cached propagation in the output `o` and return this cache if present.
"""
function check_cache(o::HDF5Output, y, t, dt)
    if !o.cache || !haskey(o["meta"]["cache"], "t")
        return y, t, dt
    end
    tc = o["meta"]["cache"]["t"]
    if tc < t
        return y, t, dt
    end
    yc = o["meta"]["cache"]["y"]
    dtc = o["meta"]["cache"]["dt"]
    return yc, tc, dtc
end

# For other outputs (e.g. MemoryOutput or another function), checking the cache does nothing.
check_cache(o, y, t, dt) = y, t, dt

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
            save = i % n == 0
            i += 1
            return save, t
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

function nostats(args...)
    return Dict{String, Any}()
end

"""
    ScanHDF5Output(scan, scanidx, args...; fname=nothing, fdir=nothing, kwargs...)

Create an [`HDF5Output`](@ref) for the given `scan` at the current `scanidx` and automatically
save the scan arrays and current values of the scan variables in the file. If given,
`fdir` is used as a directory in which to store the scan output. `fname` can be used to
manually name files. The running scan index will be appended to `fname` for each file.
"""
function ScanHDF5Output(scan, scanidx, args...; fname=nothing, fdir=nothing, kwargs...)
    fpath = isnothing(fname) ?
        Scans.makefilename(scan, scanidx) :
        Scans.makefilename(fname, scanidx)
    if !isnothing(fdir)
        fpath = joinpath(fdir, fpath)
        if !isdir(fdir)
            mkpath(fdir)
        end
    end
    savescan(HDF5Output(fpath, args...; kwargs...), scan, scanidx)
end

function ScanMemoryOutput(scan, scanidx, args...; kwargs...)
    savescan(MemoryOutput(args...; kwargs...), scan, scanidx)
end

function savescan(out, scan, scanidx)
    out("scanidx", scanidx, meta=true)
    vars = Dict{String, Any}()
    arrays = Dict{String, Any}()
    order = String[]
    shape = Int[] # scan shape
    # create grid of scan points
    for (var, arr) in zip(scan.variables, scan.arrays)
        push!(order, string(var))
        push!(shape, length(arr))
        arrays[string(var)] = arr
        vars[string(var)] = Scans.getvalue(scan, var, scanidx)
    end
    out(vars; meta=true, group="scanvars")
    out(arrays; meta=true, group="scanarrays")
    out("scanorder", order; meta=true)
    out("scanshape", shape; meta=true)
    out
end

"""
    @ScanHDF5Output(scan, scanidx, args...)

Create an [`HDF5Output`](@ref) for the given `scan` at the current `scanidx` and automatically
save the running script, scan arrays, and current values of the scan variables in the file.
All arguments, including keyword arguments, after `scanidx` are identical to `HDF5Output`.
"""
macro ScanHDF5Output(scan, scanidx, args...)
    code = ""
    try
        script = string(__source__.file)
        code = open(script, "r") do file
            read(file, String)
        end
    catch
    end
    for arg in args
        if isa(arg, Expr) && arg.head == :(=)
            arg.head = :kw
        end
    end
    ex = Expr(:call, :ScanHDF5Output, esc(scan), esc(scanidx))
    for arg in args
        push!(ex.args, esc(arg))
    end
    push!(ex.args, Expr(:kw, :script, code))
    ex
end

# Auto-generate @MemoryOutput and @HDF5Output macros
for op in (:MemoryOutput, :HDF5Output)
    eval(
        quote
            macro $op(args...)
                script_code = ""
                try
                    script = string(__source__.file)
                    code = open(script, "r") do file
                        read(file, String)
                    end
                    script_code = script*"\n"*code
                catch
                end
                for arg in args
                    if isa(arg, Expr) && arg.head == :(=)
                        arg.head = :kw
                    end
                end
                exp = Expr(:call, $op)
                for arg in args
                    push!(exp.args, esc(arg))
                end
                push!(exp.args, Expr(:kw, :script, script_code))
                exp
            end
        end
    )
end

"""
    scansave(scan, scanidx; grid, stats, fpath, script, kwargs...)

While running the given `scan`, save the variables given as keyword arguments into the scan grid as determined from the variables of the `scan`. Special keyword arguments are:

- `grid::AbstractGrid`: Save the simulation grid in a dictionary (but only once)
- `stats`: The statistics dictionary from a simulation is saved in scan grid and `NaN`-padded to account for variable lengths in the output arrays
- `fpath`: Path to the file. Defaults to the scan name plus "_collected"
- `script`: Path to the Julia scrpt file running the scan. Can be grabbed automatically using the macro [`@scansave`](@ref).

Another keyword argument `lock_stale_age` sets the time in seconds after which the function ignores
the lock on the file and writes to it anyway. Defaults to 300 s (5 min).
(Note that if the PID of the locking process appears valid, this is automatically increased 25x.)
"""
function scansave(scan, scanidx; stats=nothing, fpath=nothing,
                                 grid=nothing, script=nothing,
                                 lock_stale_age=300, kwargs...)
    fpath = isnothing(fpath) ? "$(scan.name)_collected.h5" : fpath
    lockpath = fpath*"_scanlock"
    mkpidlock(lockpath; stale_age=lock_stale_age) do # note no indent to keep this legible
    if !isfile(fpath)
        # First save - set up file structure
        HDF5.h5open(fpath, "cw") do file
            group = HDF5.create_group(file, "scanvariables")
            order = String[]
            shape = Int[] # scan shape
            # create grid of scan points
            for (k, var) in zip(scan.variables, scan.arrays)
                group[string(k)] = var
                push!(order, string(k))
                push!(shape, length(var))
            end
            file["scanorder"] = order
            if !isnothing(stats)
                group = HDF5.create_group(file, "stats")
                for (k, v) in pairs(stats)
                    dims = (size(v)..., shape...)
                    #= last dimension of a statistics array is number of steps,
                        so save in chunks of 100 steps =#
                    fixeddims_v = size(v)[1:end-1]
                    mdims = (fixeddims_v..., -1, shape...)
                    chdims = (fixeddims_v..., 100, fill(1, length(shape))...)
                    HDF5.create_dataset(group, k, HDF5.datatype(eltype(v)), (dims, mdims),
                                  chunk=chdims)
                end
                group["valid_length"] = zeros(Int, shape...)
            end
            if !isnothing(grid)
                group = HDF5.create_group(file, "grid")
                for (k, v) in pairs(grid)
                    isa(v, BitArray) && (v = Array{Bool, 1}(v))
                    group[k] = v
                end
            end
            if !isnothing(script)
                script_code = ""
                try
                    code = open(script, "r") do file
                        read(file, String)
                    end
                    script_code = script*"\n"*code
                catch
                end
                file["script"] = script_code
            end
            # deal with other keyword arguments (additional quantities to be saved)
            for (k, v) in kwargs
                # dimensions of the array
                dims = (size(v)..., shape...)
                # chunk size is dimension of one array
                chdims = (size(v)..., fill(1, length(shape))...)
                HDF5.create_dataset(file, string(k), HDF5.datatype(eltype(v)), (dims, dims),
                              chunk=chdims)
            end
        end
    end
    HDF5.h5open(fpath, "r+") do file
        scanshape = Tuple([length(ai) for ai in scan.arrays])
        cidcs = CartesianIndices(scanshape)
        scanidcs = Tuple(cidcs[scanidx])
        if !isnothing(stats)
            for (k, v) in pairs(stats)
                if size(v)[end] > size(file["stats"][k])[ndims(v)]
                    #= new point has more stats points than before - extend dataset
                        stats arrays are of shape (N1, N2,... Ns) where N1 etc are fixed and
                        Ns depends on the number of steps
                        stats *datasets" have shape (N1, N2,... Ns, Nx, Ny,...) where Nx, Ny...
                        are the lengths of the scan arrays (see scanshape above)=#
                    oldlength = size(file["stats"][k])[ndims(v)] # current Ns
                    newlength = size(v)[end] # new Ns
                    newdims = (size(v)..., scanshape...) # (N1, N2,..., new Ns, Nx, Ny,...)
                    HDF5.set_extent_dims(file["stats"][k], newdims) # set new dimensions
                    # For existing shorter arrays, fill everything above their length with NaN
                    nanidcs = (fill(:, (ndims(v)-1))..., oldlength+1:newlength)
                    allscan = fill(:, length(scanshape)) # = (1:Nx, 1:Ny,...)
                    file["stats"][k][nanidcs..., allscan...] = NaN
                end
                # Stats array has shape (N1, N2,... Ns) - fill everything up to Ns with data
                sidcs = (fill(:, (ndims(v)-1))..., 1:size(v)[end])
                file["stats"][k][sidcs..., scanidcs...] = v
                # save number of valid points we just saved
                file["stats"]["valid_length"][scanidcs...] = size(v)[end]
                # fill everything after Ns with NaN
                nanidcs = (fill(:, (ndims(v)-1))..., size(v)[end]+1:size(file["stats"][k])[ndims(v)])
                file["stats"][k][nanidcs..., scanidcs...] = NaN
            end
        end
        for (k, v) in pairs(kwargs)
            sidcs = fill(:, ndims(v))
            file[string(k)][sidcs..., scanidcs...] = v
        end 
    end
    end # mkpidlock do
end

"""
    @scansave(scan, scanidx; kwargs...)

Like [`scansave`](@ref) but also saves the script being run automatically.
"""
macro scansave(scan, scanidx, kwargs...)
    global script = string(__source__.file)
    ex = :(scansave($(esc(scan)), $(esc(scanidx)),
                    script=script))
    for arg in kwargs
        if isa(arg, Expr) && arg.head == :(=)
            arg.head = :kw
            push!(ex.args, esc(arg))
        else
            # To a macro, arguments and keyword arguments look the same, so check manually
            error("@scansave only accepts keyword arguments")
        end
    end
    ex
end

"""
    scansave_stats_array(stats_dict)

Convert the statistics dictionary created by [`scansave`](@ref) into a dictionary containing
arrays of arrays. This removes unused elements in the arrays and the need to use `valid_length`
to avoid including `NaN`s.
"""
function scansave_stats_array(stats_dict)
    vl = stats_dict["valid_length"]
    scanshape = size(vl)
    sidcs = CartesianIndices(scanshape)
    out = Dict{String, Any}()
    for (k, v) in stats_dict
        if k == "valid_length"
            continue
        end
        zdim = ndims(v) - length(scanshape) # find size of each statistic entry
        a = Array{Array{eltype(v), zdim}}(undef, scanshape) # array of arrays
        for ii in sidcs
            a[ii] = v[fill(:, zdim-1)..., 1:vl[ii], ii]
        end
        out[k] = a
    end
    out
end
end
