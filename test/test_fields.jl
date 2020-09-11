import Test: @test, @testset
import Luna: Fields, FFTW, Grid, Maths, PhysData, Processing
import Statistics: mean, std
import Random: MersenneTwister

# note that most of the Fields.jl code is tested in many other modules

function getceo(t, Et, It, ω0)
    Δt = t[argmax(It)] - t[argmax(Et)]
    Δt*ω0
end

@testset "Wavelength" begin
    # real
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 800e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 800e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
end

@testset "Energy" begin
    # real
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)
end

@testset "Duration" begin
    # real
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=1e-5)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=1e-5)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=2e-5)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=3e-5)
end

@testset "Position" begin
    # real
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    # non zero
    τ0 = -564e-15

    #real 
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)
end

@testset "CEO" begin
    # real
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-15, atol=1e-15)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-15, atol=1e-15)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = 0.0
    τ0 = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-15, atol=1e-15)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-15, atol=1e-15)

    # non zero

    #real 
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    τ0 = 0.0
    grid = Grid.RealGrid(1.0, λ0, (100e-9, 3000e-9), 1e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    # Make CEO exact multiple of one grid point to avoid issues with argmax() in getceo()
    δt = grid.t[2] - grid.t[1]
    for i = 1:10
        ϕ = i*δt*PhysData.wlfreq(λ0)

        input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Maths.hilbert(Et))
        @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-10)
        
        input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Maths.hilbert(Et))
        @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-10)
    end

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    τ0 = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (100e-9, 3000e-9), 1e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    # Make CEO exact multiple of one grid point to avoid issues with argmax() in getceo()
    δt = grid.t[2] - grid.t[1]

    for i = 1:10
        ϕ = i*δt*PhysData.wlfreq(λ0)

        input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Et)
        @test isapprox(
            getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)),
            ϕ,
            rtol=1e-10)

        input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Et)
        @test isapprox(
            getceo(grid.t,real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)),
            ϕ,
            rtol=1e-10)
    end
end

@testset "CW fields" begin
    λ0 = 1064e-9
    Pavg = 20.0
    Δλ = 4e-9
    grid = Grid.EnvGrid(1.0, λ0, (980e-9, 1160e-9), 500e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)
    input = Fields.CWSech(λ0=λ0, Pavg=Pavg, Δλ=Δλ, rng=MersenneTwister(0))
    Eω = input(grid, FT)
    Et = FT \ Eω
    I = Fields.It(Et, grid)
    istart = findfirst(isequal(1.0), grid.twin)
    iend = findlast(isequal(1.0), grid.twin)
    # test average power
    @test isapprox(mean(I[istart:iend]), Pavg, rtol=5e-16)
    # test coherence time
    @test isapprox(Processing.coherence_time(grid, Et), 3.35/(PhysData.c*(Δλ)/λ0^2*2π), rtol=1e-2)
    idcs = sortperm(PhysData.wlfreq.(grid.ω)) 
    # test spectral width
    @test isapprox(Maths.fwhm(PhysData.wlfreq.(grid.ω)[idcs], abs2.(Eω[idcs])), Δλ, rtol=3e-3)
    # now do the same for a number of realisations
    Eωs = hcat([Fields.CWSech(λ0=λ0, Pavg=Pavg, Δλ=Δλ, rng=MersenneTwister(i))(grid, FT) for i = 1:5]...)
    Iωs = abs2.(Eωs)
    Iωav = mean(Iωs, dims=2)[:,1]
    idcs = sortperm(PhysData.wlfreq.(grid.ω)) 
    # test average spectral width
    @test isapprox(Maths.fwhm(PhysData.wlfreq.(grid.ω)[idcs], Iωav[idcs], minmax=:max), Δλ, rtol=6e-4)
    Ets = FFTW.ifft(Eωs, 1)
    Its = abs2.(Ets)
    Itav = mean(Its[istart:iend,:])
    # test average power
    @test isapprox(Itav, Pavg, rtol=5e-16)
    # test diversity of power fluctuations
    @test mean(std(Its[istart:iend,:], dims=2)[:,1]) > 10
end

