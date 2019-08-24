module Stats
import Luna: Maths, Grid

function ω0(grid)
    addstat! = let ω=grid.ω
        function addstat!(d, Eω, z, dz)
            d["ω0"] = Maths.moment(ω, abs2.(Eω))
        end
    end
    return addstat!
end

function collect_stats(funcs)
    f = let funcs=funcs
        function collect_stats(Eω, z, dz)
            d = Dict{String, Any}()
            for func in funcs
                func(d, Eω, z, dz)
            end
            return d
        end
    end
    return f
end

end