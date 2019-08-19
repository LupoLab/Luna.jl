import Test: @test, @testset, @test_throws
import Luna: Output

@testset "HDF5" begin
    import HDF5
    shape = (1024, 4, 2)
    n = 11
    stat = randn()
    statsfun(y, t, dt) = Dict("stat" => stat)
    t0 = 0
    t1 = 10
    t = collect(range(t0, stop=t1, length=n))
    ω = randn((1024,))
    wd = dirname(@__FILE__)
    gitc = read(`git -C $wd rev-parse --short HEAD`, String)
    o = Output.HDF5Output("test.h5", t0, t1, n, shape, yname="y", tname="t", statsfun)
    extra = Dict()
    extra["ω"] = ω
    extra["git_commit"] = gitc
    o(extra)
    y0 = randn(ComplexF64, shape)
    y(t) = y0
    for (ii, ti) in enumerate(t)
        o(y0, ti, 0, y)
    end
    HDF5.h5open("test.h5", "r") do file
        @test all(read(file["t"]) == t)
        yr = reinterpret(ComplexF64, read(file["y"]))
        @test all([all(yr[:, :, :, ii] == y0) for ii=1:n])
        @test all(ω == read(file["ω"]))
        @test gitc == read(file["git_commit"])
        @test all(read(file["stats"]["stat"]) .== stat)
    end
    @test_throws ErrorException o(extra)
    rm("test.h5")
end

@testset "Memory" begin
    shape = (1024, 4, 2)
    n = 11
    t0 = 0
    t1 = 10
    t = collect(range(t0, stop=t1, length=n))
    ω = randn((1024,))
    wd = dirname(@__FILE__)
    gitc = read(`git -C $wd rev-parse --short HEAD`, String)
    o = Output.MemoryOutput(t0, t1, n, shape, yname="y", tname="t")
    extra = Dict()
    extra["ω"] = ω
    extra["git_commit"] = gitc
    o(extra)
    y0 = randn(ComplexF64, shape)
    y(t) = y0
    for (ii, ti) in enumerate(t)
        o(y0, ti, 0, y)
    end
    @test all(o.data["t"] == t)
    @test all([all(o.data["y"][:, :, :, ii] == y0) for ii=1:n])
    @test all(ω == o.data["ω"])
    @test gitc == o.data["git_commit"]
    @test_throws ErrorException o(extra)
end