module PythonPlotExt
import PythonPlot: ColorMap, Figure, pyplot, matplotlib
import PythonCall: pyconvert
const plt = pyplot
convertany(x) = pyconvert(Any, x)
convertarray(x) = pyconvert(Array, x)
include("commonpyplot.jl")
end # module
