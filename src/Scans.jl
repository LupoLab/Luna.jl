module Scans
import ArgParse: ArgParseSettings, parse_args, parse_item, @add_arg_table!
import Logging: @info
import Printf: @sprintf
import Base: length

struct Scan{eT}
    name::String
    variables::Vector{Symbol}
    arrays::Vector
    exec::eT
end

function Scan(name, cmdlineargs=ARGS; kwargs...)
    variables = Symbol[]
    arrays = Vector[]
    for (var, arr) in kwargs
        push!(variables, var)
        push!(arrays, arr)
    end
    Scan(name, variables, arrays, exec(cmdlineargs))
end

length(s::Scan) = (length(s.arrays) == 0) ? 0 : prod(length, s.arrays)

function addvariable!(scan, variable::Symbol, array)
    push!(scan.variables, variable)
    push!(scan.arrays, array)
end

makefilename(scan, scanidx) = @sprintf("%s_%05d.h5", scan.name, scanidx)

exec(::Nothing) = nothing
function exec(args::Vector{String})
    nothing
end

function logiter(scan, scanidx, args)
    logmsg = @sprintf("Running scan: %s (%d points)\nIndex: %05d\nVariables:\n",
                      scan.name, length(scan), scanidx)
    for (variable, value) in zip(scan.variables, args)
        logmsg *= @sprintf("\t%s: %g\n", variable, value)
    end
    @info logmsg
end

function runscan(f, scan::Scan{Nothing})
    for (scanidx, args) in enumerate(Iterators.product(scan.arrays...))
        logiter(scan, scanidx, args)
        f(scanidx, args...)
    end
end



end