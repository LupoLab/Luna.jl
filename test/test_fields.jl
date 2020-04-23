import Test: @test, @testset
import Luna: Fields, FFTW, Grid, Maths, PhysData

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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 800e-9
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 800e-9
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)
    
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)

    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=1e-5)
    
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=2e-5)

    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)
    
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)
    
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-15, atol=1e-15)
    
    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-15, atol=1e-15)

    input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
    Eω = fill(0.0 + 0.0im, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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

        input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = fill(0.0 + 0.0im, length(grid.ω))
        input!(Eω, grid, energy_t, FT)
        Et = FT \ Eω
        It = abs2.(Maths.hilbert(Et))
        @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕ, rtol=1e-10)
        
        input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = fill(0.0 + 0.0im, length(grid.ω))
        input!(Eω, grid, energy_t, FT)
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

        input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = fill(0.0 + 0.0im, length(grid.ω))
        input!(Eω, grid, energy_t, FT)
        Et = FT \ Eω
        It = abs2.(Et)
        @test isapprox(
            getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)),
            ϕ,
            rtol=1e-10)

        input! = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ, τ0=τ0)
        Eω = fill(0.0 + 0.0im, length(grid.ω))
        input!(Eω, grid, energy_t, FT)
        Et = FT \ Eω
        It = abs2.(Et)
        @test isapprox(
            getceo(grid.t,real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)),
            ϕ,
            rtol=1e-10)
    end
end
