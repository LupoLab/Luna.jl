module Stats
import Luna: Maths, Grid, Modes

function ω0(grid)
    addstat! = let ω=grid.ω
        function addstat!(d, Eω, Et, z, dz)
            d["ω0"] = Maths.moment(ω, abs2.(Eω))
        end
    end
    return addstat!
end

function energy(grid, energyfun_ω)
    function addstat!(d, Eω, Et, z, dz)
        d["energy"] = energyfun_ω(grid.ω, Eω)
    end
    return addstat!
end

function energy(grid, energyfun_ω, N)
    function addstat!(d, Eω, Et, z, dz)
        d["energy"] = [energyfun_ω(grid.ω, Eω[:, i]) for i=1:N]
    end
    return addstat!
end

function density(dfun)
    function addstat!(d, Eω, Et, z, dz)
        d["density"] = dfun(z)
    end
end

function zdz!(d, Eω, Et, z, dz)
    d["z"] = z
    d["dz"] = dz
end

function collect_stats(funcs...)
    # make sure z and dz are recorded
    if !(zdz! in funcs)
        funcs = (funcs..., zdz!)
    end
    f = let funcs=funcs
        function collect_stats(Eω, Et, z, dz)
            d = Dict{String, Any}()
            for func in funcs
                func(d, Eω, Et, z, dz)
            end
            return d
        end
    end
    return f
end

end