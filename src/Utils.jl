module Utils
import Dates
import FFTW
import Logging
import Pidfile: mkpidlock
import Base: length

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

"Struct containing the scan arrays.

`start` and `stop` are indices into `Scan.values`, which contains
the cartesian product of all of the arrays that are to be scanned over.
`Scan.values` is an `IdDict`, and each field in `values` contains (`N1 * N2 * N3...`) entries,
where `N1` etc are the lengths of the arrays to be scanned over. Each entry in each field of
`Scan.values` is the value of that scan variable for a particular run.
Taken together, the fields of `Scan.values` contain all possible combinations of the arrays.
The first array added using `@scanvar` varies the fastest, all other fields of 
`Scan.values` will contain repeated entries."
mutable struct Scan
    start::Int # First scan index to run in this execution
    stop::Int # Last scan index to run in this execution
    arrays # Array of arrays, each element is one of the arrays to be scanned over
    values # Dictionary mapping from each scan array to the expanded array of values
end

# Constructor taking command line arguments. --setup will be used in future
function Scan(ARGS)
    if "--setup" in ARGS
        start = 0
        stop = 0
    else
        start, stop = ARGS
        start = parse(Int, start) + 1
        stop = parse(Int, stop) + 1
    end
    Scan(start, stop, Array{Any, 1}(), IdDict())
end

length(s::Scan) = (length(s.arrays) > 0) ? prod([length(ai) for ai in s.arrays]) : 0

"Add a variable to a scan. Adds the array to the list of scan arrays, and re-makes the
cartesian product."
function addvar!(s::Scan, arr)
    push!(s.arrays, arr)
    makearray!(s)
end

"Make the cartesian product array containing all possible combinations of the scan arrays."
function makearray!(s::Scan)
    combos = vec(collect(Iterators.product(s.arrays...)))
    s.values = IdDict()
    for (i, a) in enumerate(s.arrays)
        # The keys in the IdDict s.values are the arrays themselves
        # Each field s.values[a] contains an array of length (N1*N2*N3...)
        s.values[a] = [ci[i] for ci in combos]
    end
end

"Macro to add to an array assignment.
    e.g.
        `@scanvar x = 1:10`
    adds the variable `x` to be scanned over."
macro scanvar(expr)
    if isa(expr, Symbol)
        # existing array being added
        q = quote
            addvar!($(esc(:__SCAN__)), $(esc(:($expr))))
        end
        return q
    end
    expr.head == :(=) || error("@scanvar must be applied to an assignment expression or variable")
    global lhs = expr.args[1]
    isa(lhs, Symbol) || error("@scanvar expressions must assign to a variable")
    quote
        $(esc(expr)) # First, simply execute the assignment
        addvar!($(esc(:__SCAN__)), $(esc(:($lhs)))) # now add the resulting array to the Scan
    end
end

"Recursively interpolate scan variables into a scan expression."
function interpolate!(ex)
    if ex.head === :($)
        var = ex.args[1]
        if var == :__SCANIDX__
            return :__SCANIDX__
        else
            return :(__SCAN__.values[$(var)][__SCANIDX__])
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

"Run the enclosed expression as a scan. Interpolates the scan variables into the expression
and then runs the expression as many times as required by the `start` and `stop` arguments."
macro scan(ex)
    body = interpolate!(ex)
    esc(quote
            for __SCANIDX__ = __SCAN__.start:__SCAN__.stop
                $body
            end
        end
    )
end


end