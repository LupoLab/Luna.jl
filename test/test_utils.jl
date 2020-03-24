import Test: @test, @testset, @test_throws
import Luna: Utils, @hlock
import HDF5

@testset "Utils" begin

@testset "dict->HDF5" begin
d = Dict{String, Any}()
d["float"] = 1.0
d["float[]"] = [1.0, 2.0, 3.0]
d["string"] = "foo"
d["nothing"] = nothing
d["dict"] = Dict("foo"=>5, "bar"=>[1, 2, 3], "baz"=>Dict("complex"=>5.0+2.0im))
fpath = joinpath(Utils.cachedir(), "output_test", "test.h5")
isfile(fpath) && rm(fpath)
isdir(dirname(fpath)) || mkpath(dirname(fpath))
Utils.save_dict_h5(fpath, d)
@test_throws ErrorException Utils.save_dict_h5(fpath, d, force=false)
@hlock HDF5.h5open(fpath) do file
    for k in ["float", "float[]", "string"]
        @test d[k] == read(file[k])
    end
    @test d["dict"]["foo"] == read(file["dict"]["foo"])
    @test d["dict"]["bar"] == read(file["dict"]["bar"])
    @test d["dict"]["baz"]["complex"] == read(file["dict"]["baz"]["complex"])
end
rm(fpath)

delete!(d, "nothing") # nothing values are converted to empty arrays and not re-converted
Utils.save_dict_h5(fpath, d)
dd = Utils.load_dict_h5(fpath)
@test d == dd

rm(fpath)
rm(dirname(fpath))
end

@testset "Chunking" begin
    function contains_all_unique(chunks, x)
        contains_unique = []
        for xi in x
            c = count([xi in chi for chi in chunks])
            push!(contains_unique, (c==1))
        end
        return all(contains_unique)
    end
    pass = []
    for L = 5:11:896
        for n = 1:24
            x = collect(1:L)
            ch = Utils.chunks(x, n)
            push!(pass, contains_all_unique(ch, x))
        end
    end
    @test all(pass)
end


end