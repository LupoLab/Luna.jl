using Luna
Luna.set_fftw_mode(:estimate)
import Logging: @warn, disable_logging, Info
import PyPlot: plt

disable_logging(Info) # disable progress logging to avoid filling the shell window

function run(item)
    if isfile(item) && endswith(item, ".jl")
        println("============ Running example $item")
        try
            include(item)
        catch e
            bt = catch_backtrace()
            @warn sprint(showerror, e, bt)
        finally
            plt.close("all")
        end
    elseif isdir(item)
        for iitem in readdir(item)
            run(joinpath(item, iitem))
        end
    end
end

run(joinpath(Utils.lunadir(), "examples"))