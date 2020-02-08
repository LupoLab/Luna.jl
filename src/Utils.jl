module Utils
import Dates
import FFTW
import Logging
import Pidfile: mkpidlock

function git_commit()
    wd = dirname(@__FILE__)
    try
        commit = read(`git -C $wd describe --always --tags --dirty`, String)
        commit = commit[1:end-1] # Strip newline off the end
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

function git_branch()
    try
        wd = dirname(@__FILE__)
        b = read(`git -C $wd rev-parse --abbrev-ref HEAD`, String)
        return b[1:end-1] # Strip newline off the end
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

srcdir() = dirname(@__FILE__)

lunadir() = dirname(srcdir())

datadir() = joinpath(srcdir(), "data")

cachedir() = joinpath(homedir(), ".luna")

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

function loadFFTwisdom()
    fpath = joinpath(cachedir(), "FFTWcache")
    lockpath = joinpath(cachedir(), "FFTWlock")
    if isfile(fpath)
        Logging.@info("Found FFTW wisdom at $fpath")
        pidlock = mkpidlock(lockpath)
        ret = FFTW.import_wisdom(fpath)
        close(pidlock)
        success = (ret != 0)
        Logging.@info(success ? "FFTW wisdom loaded" : "Loading FFTW wisdom failed")
        return success
    else
        Logging.@info("No FFTW wisdom found")
        return false
    end
end

function saveFFTwisdom()
    fpath = joinpath(cachedir(), "FFTWcache")
    lockpath = joinpath(cachedir(), "FFTWlock")
    pidlock = mkpidlock(lockpath)
    isfile(fpath) && rm(fpath)
    isdir(cachedir()) || mkpath(cachedir())
    FFTW.export_wisdom(fpath)
    close(pidlock)
    Logging.@info("FFTW wisdom saved to $fpath")
end

macro scaninit()
    esc(quote
        SCANSETUP = "--setup" in ARGS
        SCANVARS = Symbol[]
        if !SCANSETUP
            start, stop = ARGS
            start = parse(Int, start) + 1
            stop = parse(Int, stop) + 1
        end
    end)
end

macro scanvar(expr)
    expr.head == :(=) || error("@scanvar must be applied to an assignment expression")
    global lhs = expr.args[1]
    isa(lhs, Symbol) || error("@scanvar expressions must assign to a variable")
    quote
        $(esc(expr))
        push!($(esc(:SCANVARS)), lhs)
    end
end

function interpolate!(ex)
    if ex.head === :($)
        var = ex.args[1]
        if var == :SCANIDX
            return :SCANIDX
        else
            return :(SCANVALS[$var][SCANIDX])
        end
    else
        for i in 1:length(ex.args)
            arg = ex.args[i]
            if isa(arg, Expr)
                ex.args[i] = interpolate!(arg)
            end
        end
    end
    return ex
end

macro scan(ex)
    body = interpolate!(ex)
    quote
        $(esc(quote
            combos = vec(collect(Iterators.product([eval(vi) for vi in SCANVARS]...)))
            SCANVALS = Dict{Any, Any}()
            for (i, sv) in enumerate(SCANVARS)
                SCANVALS[eval(sv)] = [ci[i] for ci in combos]
            end
        end))
        for $(esc(:SCANIDX)) = $(esc(:start)):$(esc(:stop))
            Logging.@info("SCAN: $($(esc(:SCANIDX)))")
            $(esc(body))
        end
    end
end


end