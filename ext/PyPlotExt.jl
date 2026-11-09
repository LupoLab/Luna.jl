module PyPlotExt
import PyPlot: ColorMap, plt, Figure, matplotlib
convertany(x) = x
convertarray(x) = x
include("commonpyplot.jl")
end # module
