import Test: @test, @testset, @test_throws
import Luna: Scans, Utils

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
Scans.@scaninit "scantest"
@test __SCAN__.mode == :range
@test __SCAN__.name == "scantest"
@test !__SCAN__.parallel

Scans.@scanvar dummy = range(1, length=10)
@test __SCAN__.vars[:dummy] == dummy
@test __SCAN__.values[dummy] == dummy

Scans.@scan begin
    x = $dummy
    @test x == dummy[__SCANIDX__]
end

for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, "--batch", "1,3")
Scans.@scaninit "scantest"
@test __SCAN__.mode == :batch
Scans.@scanvar dummy = range(1, length=10)
@test __SCAN__.batch == (1, 3)
@test __SCAN__.idcs == [1, 4, 7, 10]
Scans.@scan begin
    x = $dummy
    @test x == dummy[__SCANIDX__]
end
end