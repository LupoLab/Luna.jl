module Plotting

import Luna: Maths
import PyPlot: ColorMap

function cmap_white(cmap, N=512, n=8)
    vals = collect(range(0, 1, length=n))
    vals_i = collect(range(0, 1, length=N))
    cm = ColorMap(cmap)
    clist = cm(vals)
    clist[1, :] = [1, 1, 1, 1]
    clist_i = Array{Float64}(undef, (N, 4))
    for ii in 1:4
        clist_i[:, ii] .= Maths.CSpline(vals, clist[:, ii]).(vals_i)
    end
    ColorMap(clist_i)
end


end