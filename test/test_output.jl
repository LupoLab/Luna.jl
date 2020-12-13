import Test: @test, @testset, @test_throws
import Luna: Output, Processing
using EllipsisNotation

@testset "HDF5" begin
    import HDF5
    import Luna: Utils
    fpath = joinpath(homedir(), ".luna", "output_test", "test.h5")
    isfile(fpath) && rm(fpath)
    shape = (1024, 4, 2)
    n = 11
    stat = randn()
    statsfun(y, t, dt) = Dict("stat" => stat, "stat2d" => [stat, stat])
    t0 = 0
    t1 = 10
    t = collect(range(t0, stop=t1, length=n))
    ω = randn((1024,))
    wd = dirname(@__FILE__)
    gitc = Utils.git_commit()
    o = Output.HDF5Output(fpath, t0, t1, n, statsfun; yname="y", tname="t")
    extra = Dict()
    extra["ω"] = ω
    extra["git_commit"] = gitc
    o(extra)
    meta = Dict()
    meta["meta1"] = 100
    meta["meta2"] = "src"
    o(meta, meta=true)
    y0 = randn(ComplexF64, shape)
    y(t) = y0
    for (ii, ti) in enumerate(t)
        o(y0, ti, 0, y)
    end
    @test_throws ErrorException o(extra)
    @test o(extra, force=true) === nothing
    @test_throws ErrorException o("git_commit", gitc)
    HDF5.h5open(fpath, "r") do file
        @test all(read(file["t"]) == t)
        global yr = read(file["y"])
        @test all([all(yr[:, :, :, ii] == y0) for ii=1:n])
        @test all(ω == read(file["ω"]))
        @test gitc == read(file["git_commit"])
        @test all(read(file["stats"]["stat"]) .== stat)
        @test all(read(file["stats"]["stat2d"]) .== stat)
        @test Utils.git_commit() == read(file["meta"]["git_commit"])
        # Need to strip out date from sourcecode to compare
        src = read(file["meta"]["sourcecode"])
        @test split(Utils.sourcecode(), '\n')[2:end] == split(src, '\n')[2:end]
        @test 100 == read(file["meta"]["meta1"])
        @test "src" == read(file["meta"]["meta2"])
    end
    @test all(yr .== o["y"])
    @test 100 == o["meta"]["meta1"]
    rm(fpath)
    rm(splitdir(fpath)[1], force=true)
end

@testset "Memory" begin
    import Luna: Utils
    shape = (1024, 4, 2)
    n = 11
    stat = randn()
    statsfun(y, t, dt) = Dict("stat" => stat, "stat2d" => [stat, stat])
    t0 = 0
    t1 = 10
    t = collect(range(t0, stop=t1, length=n))
    ω = randn((1024,))
    wd = dirname(@__FILE__)
    gitc = Utils.git_commit()
    o = Output.MemoryOutput(t0, t1, n, statsfun, yname="y", tname="t")
    extra = Dict()
    extra["ω"] = ω
    extra["git_commit"] = gitc
    o(extra)
    meta = Dict()
    meta["meta1"] = 100
    meta["meta2"] = "src"
    o(meta, meta=true)
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
    @test o(extra, force=true) === nothing
    @test all(o.data["stats"]["stat"] .== stat)
    @test all(o.data["stats"]["stat2d"] .== stat)
    @test Utils.git_commit() == o.data["meta"]["git_commit"]
    # Need to strip out date from sourcecode to compare
    src = o.data["meta"]["sourcecode"]
    @test split(Utils.sourcecode(), '\n')[2:end] == split(src, '\n')[2:end]
    @test 100 == o.data["meta"]["meta1"]
    @test "src" == o.data["meta"]["meta2"]
end

fpath = joinpath(homedir(), ".luna", "output_test", "test.h5")
fpath_comp = joinpath(homedir(), ".luna", "output_test", "test_comp.h5")
@testset "HDF5 vs Memory" begin
    using Luna
    import FFTW
    import HDF5
    import LinearAlgebra: norm

    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(5e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
    aeff = let m=m
        z -> Modes.Aeff(m, z=z)
    end
    energyfun, energyfunω = Fields.energyfuncs(grid)
    densityfun = let dens0=PhysData.density(gas, pres)
        z -> dens0
    end
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)

    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(
        grid, densityfun, responses, inputs, βfun!, aeff)
    statsfun = Stats.collect_stats(grid, Eω,
                                   Stats.ω0(grid),
                                   Stats.energy(grid, energyfunω))
    hdf5 = Output.HDF5Output(fpath, 0, grid.zmax, 51, statsfun)
    hdf5c = Output.HDF5Output(fpath_comp, 0, grid.zmax, 51, statsfun,
                              compression=true)
    mem = Output.MemoryOutput(0, grid.zmax, 51, statsfun)
    function outfun(args...; kwargs...)
        hdf5(args...; kwargs...)
        hdf5c(args...; kwargs...)
        mem(args...; kwargs...)
    end
    for o in (hdf5, hdf5c, mem)
        o(Dict("λ0" => λ0))
        o("τ", τ)
    end
    Luna.run(Eω, grid, linop, transform, FT, outfun, status_period=10)
    HDF5.h5open(hdf5.fpath, "r") do file
        @test read(file["λ0"]) == mem.data["λ0"]
        Eω = reinterpret(ComplexF64, read(file["Eω"]))
        @test Eω == mem.data["Eω"]
        @test read(file["stats"]["ω0"]) == mem.data["stats"]["ω0"]
        @test read(file["z"]) == mem.data["z"]
        @test read(file["grid"]) == Grid.to_dict(grid)
        @test read(file["simulation_type"]["field"]) == "field-resolved"
        @test read(file["simulation_type"]["transform"]) == string(transform)
    end
    HDF5.h5open(hdf5c.fpath, "r") do file
        @test read(file["λ0"]) == mem.data["λ0"]
        Eω = reinterpret(ComplexF64, read(file["Eω"]))
        @test Eω == mem.data["Eω"]
        @test read(file["stats"]["ω0"]) == mem.data["stats"]["ω0"]
        @test read(file["z"]) == mem.data["z"]
        @test read(file["grid"]) == Grid.to_dict(grid)
        @test read(file["simulation_type"]["field"]) == "field-resolved"
        @test read(file["simulation_type"]["transform"]) == string(transform)
    end
    @test stat(hdf5.fpath).size >= stat(hdf5c.fpath).size
    # Test read-only
    o = Output.HDF5Output(fpath)
    HDF5.h5open(o.fpath, "r") do file
        @test read(file["λ0"]) == mem.data["λ0"]
        Eω = reinterpret(ComplexF64, read(file["Eω"]))
        @test Eω == mem.data["Eω"]
        @test read(file["stats"]["ω0"]) == mem.data["stats"]["ω0"]
        @test read(file["z"]) == mem.data["z"]
        @test read(file["grid"]) == Grid.to_dict(grid)
        @test read(file["simulation_type"]["field"]) == "field-resolved"
        @test read(file["simulation_type"]["transform"]) == string(transform)
    end
    # test slice reading
    @test o["Eω", :, 1] == mem["Eω"][:, 1]
    @test o["Eω", 1, :] == mem["Eω"][1, :]
    @test o["Eω", :, 1:5] == mem["Eω"][:, 1:5]
    @test o["Eω", :, [1, 2, 50]] == mem["Eω"][:, [1, 2, 50]]
    @test o["Eω", .., 1] == mem["Eω"][:, 1]
    @test o["Eω", .., [1, 2, 50]] == mem["Eω"][:, [1, 2, 50]]
    @test mem["Eω", :, 1] == mem["Eω"][:, 1]
    @test mem["Eω", 1, :] == mem["Eω"][1, :]
    @test mem["Eω", :, 1:5] == mem["Eω"][:, 1:5]
    @test mem["Eω", :, [1, 2, 50]] == mem["Eω"][:, [1, 2, 50]]
    @test mem["Eω", .., 1] == mem["Eω"][:, 1]
    @test mem["Eω", .., [1, 2, 50]] == mem["Eω"][:, [1, 2, 50]]
    # test slice reading in processing functions
    ω, Eω, zac = Processing.getEω(o, 5e-2)
    @test (ω, Eω[:, 1]) == Processing.getEω(grid, mem["Eω"][:, 51])
    @test zac[1] == 5e-2
end
rm(fpath)
rm(fpath_comp)
rm(splitdir(fpath)[1], force=true)

##
fpath = joinpath(homedir(), ".luna", "output_test", "test.h5")
@testset "Continuing" begin
    using Luna
    import FFTW
    import HDF5

    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(5e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
    aeff(z) = Modes.Aeff(m, z=z)
    energyfun, energyfunω = Fields.energyfuncs(grid)
    dens0 = PhysData.density(gas, pres)
    densityfun(z) = dens0
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)

    # Run with arbitrary error at 3 cm
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(
        grid, densityfun, responses, inputs, βfun!, aeff)
    statsfun = Stats.collect_stats(grid, Eω,
                                   Stats.ω0(grid),
                                   Stats.energy(grid, energyfunω))
    output = Output.HDF5Output(fpath, 0, grid.zmax, 51, statsfun)
    function stepfun(Eω, z, dz, interpolant)
        output(Eω, z, dz, interpolant)
        if z > 3e-2
            error("Oh no!")
        end
    end
    stepfun(args...; kwargs...) = output(args...; kwargs...)
    try
        Luna.run(Eω, grid, linop, transform, FT, stepfun, status_period=10, z0=0.0)
    catch
    end

    # Run again, starting from 3 cm
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(
        grid, densityfun, responses, inputs, βfun!, aeff)
    statsfun = Stats.collect_stats(grid, Eω,
                                   Stats.ω0(grid),
                                   Stats.energy(grid, energyfunω))
    output = Output.HDF5Output(fpath, 0, grid.zmax, 51, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

    # Run from scratch with MemoryOutput
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(
        grid, densityfun, responses, inputs, βfun!, aeff)
    statsfun = Stats.collect_stats(grid, Eω,
                                   Stats.ω0(grid),
                                   Stats.energy(grid, energyfunω))
    mem = Output.MemoryOutput(0, grid.zmax, 51, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, mem, status_period=5)

    idx1 = findfirst(grid.ωwin .!= 1)
    idx2 = findlast(grid.ωwin .== 1)
    Eωm = mem["Eω"][idx1:idx2, :]
    Eω = output["Eω"][idx1:idx2, :]
    Iω = abs2.(Eω)
    Iωm = abs2.(Eωm)
    @test norm(Iω - Iωm)/norm(Iω) < 1e-7
    @test all(isapprox.(Eωm, Eω, atol=1e-4*maximum(abs.(Eωm))))
    @test all(isapprox.(output["stats"]["ω0"], mem.data["stats"]["ω0"], rtol=1e-6))
    @test all(output["stats"]["energy"] .≈ mem.data["stats"]["energy"])
    @test output["z"] == mem.data["z"]
    @test output["grid"] == Grid.to_dict(grid)
    @test output["simulation_type"]["field"] == "field-resolved"
    @test output["simulation_type"]["transform"] == string(transform)
end
rm(fpath, force=true)
rm(splitdir(fpath)[1], force=true)
