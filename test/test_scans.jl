import Test: @test, @testset, @test_throws
import Luna: Scans, Utils, Output
import Luna.Scans: @scanvar, @scan, @scaninit
import HDF5

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
            ch = Scans.chunks(x, n)
            push!(pass, contains_all_unique(ch, x))
        end
    end
    @test all(pass)
end

@testset "Scanning" begin
for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, "--range", "1:4")
@scaninit "scantest"
@scanvar dummy = collect(1:5)
@test __SCAN__.vars[:dummy] == dummy
@scanvar dummy2 = collect(1:10)
@test __SCAN__.vars[:dummy] == dummy
@test __SCAN__.vars[:dummy2] == dummy2

for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, "--range", "1:4")
@scaninit "scantest"
@test __SCAN__.mode == :range
@test __SCAN__.name == "scantest"
@test !__SCAN__.parallel

@scanvar dummy = range(1, length=10)
@test __SCAN__.vars[:dummy] == dummy
@test __SCAN__.values[dummy] == dummy

@scan begin
    x = $dummy
    @test x == dummy[__SCANIDX__]
end

for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, "--batch", "1,3")
@scaninit "scantest"
@test __SCAN__.mode == :batch
@scanvar dummy = range(1, length=10)
@test __SCAN__.batch == (1, 3)
@test __SCAN__.idcs == [1, 4, 7, 10]
@scan begin
    x = $dummy
    @test x == dummy[__SCANIDX__]
end
end

try
@testset "scansave" begin
for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, "--local")
@scaninit "scantest"
@test __SCAN__.mode == :local
@test __SCAN__.name == "scantest"

@scanvar x = collect(1:8)
@scanvar y = collect(1:6)

@scan begin
    out = [$x * $y, ($x)^2, ($y)^2]
    slength = $x * $y
    e = fill(1.0*slength, slength)
    em = fill(1.0*slength, (2, slength))
    stats = Dict("energy" => e, "energym" => em)
    xx, yy = $x, $y
    Output.@scansave(out, stats, keyword=[xx, yy])
end
HDF5.h5open("scantest_collected.h5", "r") do file
    @test read(file["scanvariables"]["x"]) == x
    @test read(file["scanvariables"]["y"]) == y
    @test read(file["scanorder"]) == ["x", "y"]
    out = read(file["Eω"])
    stats = read(file["stats"])
    pass = true
    for (ix, xi) in enumerate(x)
        for (iy, yi) in enumerate(y)
            pass = pass && (out[:, ix, iy] == [xi * yi, (xi)^2, (yi)^2])
            slength = xi * yi
            pass = pass && (stats["valid_length"][ix, iy] == slength)
            e = fill(1.0*slength, slength)
            em = fill(1.0*slength, (2, slength))
            pass = pass && (stats["energy"][1:slength, ix, iy] == e)
            pass = pass && all(isnan.(stats["energy"][slength+1:end, ix, iy]))
            pass = pass && (stats["energym"][:, 1:slength, ix, iy] == em)
            pass = pass && all(isnan.(stats["energym"][:, slength+1:end, ix, iy]))
            pass = pass && (file["keyword"][:, ix, iy] == [xi, yi])
        end
    end
    @test pass
    this = @__FILE__
    code = open(this, "r") do file
                read(file, String)
            end
    @test read(file["script"]) == this * "\n" * code
end
rm("scantest_collected.h5")
end
catch
rm("scantest_collected.h5")
end