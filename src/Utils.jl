module Utils
import Dates
import FFTW
import Logging
import LibGit2
import Pidfile: mkpidlock
import HDF5
import Luna: @hlock, settings
import Printf: @sprintf
import Scratch: get_scratch!
import Luna

subzero = '\u2080'
subscript(digit::Char) = string(Char(codepoint(subzero)+parse(Int, digit)))
subscript(num::AbstractString) = prod([subscript(chi) for chi in num])
subscript(num::Int) = subscript(string(num))

unsubscript(digit::Char) = string(codepoint(digit)-codepoint(subzero))
unsubscript(num::AbstractString) = prod([unsubscript(chi) for chi in num])

function git_commit()
    try
        repo = LibGit2.GitRepo(lunadir())
        commit = string(LibGit2.GitHash(LibGit2.head(repo)))
        LibGit2.isdirty(repo) && (commit *= " (dirty)")
        return commit
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

function git_branch()
    try
        repo = LibGit2.GitRepo(lunadir())
        n = string(LibGit2.name(LibGit2.head(repo)))
        branch = split(n, "/")[end]
        return branch
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

srcdir() = dirname(@__FILE__)

lunadir() = dirname(srcdir())

datadir() = joinpath(srcdir(), "data")

cachedir() = get_scratch!(Luna, "lunacache")

function sourcecode()
    src = dirname(@__FILE__)
    luna = dirname(src)
    out = "#= Date: $(Dates.now())\n"
    out *= "git branch: $(git_branch())\n"
    out *= "git commit: $(git_commit())\n"
    out *= "hostname: $(gethostname())\n"
    out *= "=#"
    for folder in (src, luna)
        for obj in readdir(folder)
            if isfile(joinpath(folder, obj))
                if split(obj, ".")[end] in ("md", "jl", "txt", "toml") # avoid binary files
                    out *= "\n" * "#" * "="^8 * obj * "="^8 * "#" * "\n"^2
                    open(joinpath(folder, obj), "r") do file
                        out *= read(file, String)
                    end
                end
            end
        end
    end
    return out
end

function FFTWthreads()
    if Threads.nthreads() == 1
        1
    else
        settings["fftw_threads"] == 0 ? 4*Threads.nthreads() : settings["fftw_threads"]
    end
end

function loadFFTwisdom()
    FFTW.set_num_threads(FFTWthreads())
    fpath = joinpath(cachedir(), "FFTWcache_$(FFTWthreads())threads")
    lockpath = joinpath(cachedir(), "FFTWlock")
    isdir(cachedir()) || mkpath(cachedir())
    if isfile(fpath)
        Logging.@info("Found FFTW wisdom at $fpath")
        pidlock = mkpidlock(lockpath)
        ret = FFTW.import_wisdom(fpath)
        close(pidlock)
    else
        Logging.@info("No FFTW wisdom found")
    end
end

function saveFFTwisdom()
    fpath = joinpath(cachedir(), "FFTWcache_$(FFTWthreads())threads")
    lockpath = joinpath(cachedir(), "FFTWlock")
    pidlock = mkpidlock(lockpath)
    isfile(fpath) && rm(fpath)
    isdir(cachedir()) || mkpath(cachedir())
    FFTW.export_wisdom(fpath)
    close(pidlock)
    Logging.@info("FFTW wisdom saved to $fpath")
end

function save_dict_h5(fpath, d; force=false, rmold=false)
    if isfile(fpath) && rmold
        rm(fpath)
    end

    function dict2h5(k::AbstractString, v, parent)
        if HDF5.haskey(parent, k) && !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = v
    end

    function dict2h5(k::AbstractString, v::BitArray, parent)
        if HDF5.haskey(parent, k) && !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = Array{Bool, 1}(v)
    end

    function dict2h5(k::AbstractString, v::Nothing, parent)
        if HDF5.haskey(parent, k) && !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = Float64[]
    end

    function dict2h5(k::AbstractString, v::AbstractDict, parent)
        if !HDF5.haskey(parent, k)
            subparent = HDF5.create_group(parent, k)
        else
            subparent = parent[k]
        end
        for (kk, vv) in pairs(v)
            dict2h5(kk, vv, subparent)
        end
    end
    
    @hlock HDF5.h5open(fpath, "cw") do file
        for (k, v) in pairs(d)
            dict2h5(k, v, file)
        end
    end
end

function save_dict_h5(fpath, t::NamedTuple; kwargs...)
    d = Dict{String, Any}()
    for (k, v) in pairs(t)
        d[string(k)] = v
    end
    save_dict_h5(fpath, d; kwargs...)
end

function load_dict_h5(fpath)
    isfile(fpath) || error("Error loading file $fpath: file does not exist")

    function h52dict(x::HDF5.Dataset)
        return read(x)
    end

    function h52dict(x::Union{HDF5.Group, HDF5.File})
        dd = Dict{String, Any}()
        for n in keys(x)
            dd[n] = h52dict(x[n])
        end
        return dd
    end

    d = @hlock HDF5.h5open(fpath) do file
        h52dict(file)
    end
end

function format_elapsed(ms::Dates.Millisecond)
    stot = Dates.value(ms)/1000 # total seconds
    seconds = stot % 60
    stot -= seconds
    mtot = stot ÷ 60
    minutes = mtot % 60
    mtot -= minutes
    hours = mtot ÷ 60
    out = @sprintf("%.3f seconds", seconds)
    minstr = abs(minutes) == 1 ? "minute" : "minutes"
    hrstr = abs(hours) == 1 ? "hour" : "hours"
    if abs(hours) > 0
        out = @sprintf("%d %s, ", minutes, minstr) * out
        out = @sprintf("%d %s, ", hours, hrstr) * out
    elseif abs(minutes) > 0
        out = @sprintf("%d %s, ", minutes, minstr) * out
    end
    out
end

end