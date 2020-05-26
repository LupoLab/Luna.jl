using Luna
Luna.set_fftw_mode(:estimate)
import Logging: @warn, @info

function run(item)
    if isfile(item) && endswith(item, ".jl")
        @info("Running example $item")
        try
            include(item)
        catch e
            bt = catch_backtrace()
            @warn sprint(showerror, e, bt)
        end
    elseif isdir(item)
        for iitem in readdir(item)
            run(joinpath(item, iitem))
        end
    end
end

run(joinpath(Utils.lunadir(), "examples"))