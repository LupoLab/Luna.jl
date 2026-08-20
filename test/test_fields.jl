import Test: @test, @testset
using Luna
import FFTW
import Statistics: mean, std
import Random: MersenneTwister
import Luna: Hankel

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
    ϕ = [0.0, 0.0]
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 800e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = [0.0, 0.0]
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 800e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
    λ0 = 320e-9
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    @test isapprox(PhysData.wlfreq(grid.ω[argmax(abs2.(Eω))]), λ0, rtol=3e-4)
end

@testset "Energy" begin
    # real
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = [0.0, 0.0]
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = [0.0, 0.0]
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    @test isapprox(energy_t(Et), energy, rtol=1e-14)
end

@testset "Duration" begin
    # real
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = [0.0, 0.0]
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=1e-5)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=1e-5)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕ = [0.0, 0.0]
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(Maths.fwhm(grid.t, It), τfwhm, rtol=2e-5)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
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
    τ0 = 5e-15
    # elements of ϕ are [CEP, group delay, GDD, TOD, ...]
    # so [0.0, τ0] is a delay by τ0
    ϕ = [0.0, τ0]
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    τ0 = 5e-15
    # elements of ϕ are [CEP, group delay, GDD, TOD, ...]
    # so [0.0, τ0] is a delay by τ0
    ϕ = [0.0, τ0]
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
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
    τ0 = 5e-15
    # elements of ϕ are [CEP, group delay, GDD, TOD, ...]
    # so [0.0, τ0] is a delay by τ0
    ϕ = [0.0, τ0]
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    τ0 = 5e-15
    # elements of ϕ are [CEP, group delay, GDD, TOD, ...]
    # so [0.0, τ0] is a delay by τ0
    ϕ = [0.0, τ0]
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(grid.t[argmax(It)], τ0, rtol=1e-15, atol=1e-15)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)
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
    ϕCEO = 0.0
    grid = Grid.RealGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕCEO, rtol=1e-15, atol=1e-15)
    
    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Maths.hilbert(Et))
    @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), ϕCEO, rtol=1e-15, atol=1e-15)

    # Envelope
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    ϕCEO = 0.0
    grid = Grid.EnvGrid(1.0, λ0, (160e-9, 3000e-9), 10e-12)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)), ϕCEO, rtol=1e-15, atol=1e-15)

    input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
    Eω = input(grid, FT)
    Et = FT \ Eω
    It = abs2.(Et)
    @test isapprox(getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0)), ϕCEO, rtol=1e-15, atol=1e-15)

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
        ϕCEO = i*δt*PhysData.wlfreq(λ0)

        input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Maths.hilbert(Et))
        @test isapprox(abs(getceo(grid.t, Et, It, PhysData.wlfreq(λ0))), ϕCEO, rtol=1e-10)
        
        input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Maths.hilbert(Et))
        @test isapprox(abs(getceo(grid.t, Et, It, PhysData.wlfreq(λ0))), ϕCEO, rtol=1e-10)
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
        ϕCEO = i*δt*PhysData.wlfreq(λ0)

        input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Et)
        @test isapprox(
            abs(getceo(grid.t, real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0))),
            ϕCEO,
            rtol=1e-10)

        input = Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
        Eω = input(grid, FT)
        Et = FT \ Eω
        It = abs2.(Et)
        @test isapprox(
            abs(getceo(grid.t,real(Et.*exp.(im .* grid.ω0 .* grid.t)), It, PhysData.wlfreq(λ0))),
            ϕCEO,
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
##
@testset "Propagation" begin
    λ0 = 800e-9
    τfwhm = 2.5e-15
    grid = Grid.RealGrid(1, λ0, (400e-9, 1200e-9), 500e-15)
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)
    Eω = input(grid, FT)
    Eωβ1 = Fields.prop_taylor(Eω, grid, [0, 10e-15], λ0)
    Et = FT \ Eωβ1
    @test isapprox(Maths.moment(grid.t, abs2.(Maths.hilbert(Et))), 10e-15, rtol=1e-6)

    # Test sign of dispersion
    Eωβ2 = Fields.prop_taylor(Eω, grid, [0, 0, 15e-30], λ0) # positive chirp
    Et = FT \ Eωβ2
    gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15) # spectrogram
    ω0 = Maths.moment(grid.ω, abs2.(gab))
    @test ω0[1] < ω0[2] # mean frequency at earlier time should be lower (upchirp)

    # Test pulse stretching for Gaussian pulse
    τfwhm = 30e-15
    τ0 = Tools.τfw_to_τ0(τfwhm, :gauss)
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
    Eω = input(grid, FT)
    Eωβ2 = Fields.prop_taylor(Eω, grid, [0, 0, τ0^2], λ0) # should lead to √2 increase
    Et = FT \ Eωβ2
    τfwβ2 = Maths.fwhm(grid.t, abs2.(Maths.hilbert(Et)); method=:spline)
    τ0β2 = Tools.τfw_to_τ0(τfwβ2, :gauss)
    @test τ0β2 ≈ √2 * τ0

    # Test pulse stretching and sign of the dispersion for modal propagation
    ω0 = PhysData.wlfreq(λ0)
    # Artificial mode with τ0^2 2nd order dispersion over 2 m and α=0.1
    β(ω; z=0) = ω0/PhysData.c + 1/(0.999*PhysData.c)*(ω-ω0) + τ0^2/4*(ω-ω0)^2
    α(ω; z=0) = 0.1
    m = Modes.arbitrary(neff=Modes.neff_from_αβ(α, β))
    Eωm = copy(Eω)
    Fields.prop_mode!(Eωm, grid.ω, m, 2, λ0)
    Et = FT \ Eωm
    τfwm = Maths.fwhm(grid.t, abs2.(Maths.hilbert(Et)); method=:spline)
    τ0m = Tools.τfw_to_τ0(τfwm, :gauss)
    @test τ0m ≈ √2 * τ0
    # Check signs are correct:
    # α = 0.1 should give loss
    et, eω = Fields.energyfuncs(grid)
    @test eω(Eω)*exp(-0.2) ≈ eω(Eωm)
    # β2 > 0 should give positive chirp:
    gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
    ω0 = Maths.moment(grid.ω, abs2.(gab))
    @test ω0[1] < ω0[2]


    # Test sign of dispersion for glass
    Eωglass = Fields.prop_material(Eω, grid, :SiO2, 0.5e-3, λ0)
    Et = FT \ Eωglass
    gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
    ω0 = Maths.moment(grid.ω, abs2.(gab))
    @test ω0[1] < ω0[2]

    # Test sign of dispersion for chirped mirrors
    for mirror in (:PC70, :ThorlabsUMC)
        Eωmirr = Fields.prop_mirror(Eω, grid, 2, mirror) # one pair
        Et = FT \ Eωmirr
        gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
        ω0 = Maths.moment(grid.ω, abs2.(gab))
        @test ω0[1] > ω0[2] # negative chirp, so frequency should go down with time
    end

    # custom chirped mirrors
    gdd = -50e-30 # -50 fs² per mirror exactly
    λGDD = collect(range(600e-9, 1000e-9, 256))
    GDD = ones(size(λGDD)) .* gdd
    λR = collect(range(500e-9, 1200e-9, 256))
    R = Maths.planck_taper.(λR, 500e-9, 520e-9, 1180e-9, 1200e-9)
    @testset for reflections in 1:10
        Eωmirr = Fields.prop_mirror(Eω, grid, reflections, λR, R, λGDD, GDD, λ0, 600e-9, 1000e-9)
        Et = FT \ Eωmirr
        gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
        ω0 = Maths.moment(grid.ω, abs2.(gab))
        @test ω0[1] > ω0[2] # negative chirp, so frequency should go down with time
        # check that this can be compressed by exactly removing the mirror chirp
        ϕs, Eωcomp = Fields.optcomp_taylor(Eωmirr, grid, λ0)
        @test isapprox(ϕs[3], -reflections*gdd, rtol=1e-4)
    end

    λ0 = 1030e-9
    τfwhm = 15e-15
    grid = Grid.RealGrid(1, λ0, (400e-9, 1500e-9), 2e-12)
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)
    Eω = input(grid, FT)
    mirrors = Dict(
        :PC147 => -60e-30,
        :PC1611 => -150e-30,
        :PC1821 => -120e-30
    )
    reflections = 1
    @testset for (mirror, gdd) in pairs(mirrors)
        Eωmirr = Fields.prop_mirror(Eω, grid, reflections, mirror) # one pair
        Et = FT \ Eωmirr
        gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
        ω0 = Maths.moment(grid.ω, abs2.(gab))
        @test ω0[1] > ω0[2] # negative chirp, so frequency should go down with time
        ϕs, Eωcomp = Fields.optcomp_taylor(Eωmirr, grid, λ0; order=3)
        @test isapprox(ϕs[3], -reflections*gdd, rtol=0.1)
    end

    # Strong chirped mirrors: longer pulses
    λ0 = 1030e-9
    τfwhm = 50e-15
    grid = Grid.RealGrid(1, λ0, (400e-9, 1500e-9), 2e-12)
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)
    Eω = input(grid, FT)
    mirrors = Dict(
        :HD120 => -200e-30,
        :HD59 => -500e-30,
    )
    reflections = 1
    @testset for (mirror, gdd) in pairs(mirrors)
        Eωmirr = Fields.prop_mirror(Eω, grid, reflections, mirror) # one pair
        Et = FT \ Eωmirr
        gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
        ω0 = Maths.moment(grid.ω, abs2.(gab))
        @test ω0[1] > ω0[2] # negative chirp, so frequency should go down with time
        ϕs, Eωcomp = Fields.optcomp_taylor(Eωmirr, grid, λ0; order=3)
        @test isapprox(ϕs[3], -reflections*gdd, rtol=0.1)
    end

    # check (back-)propagation for all gases with a large grid
    λ0 = 800e-9
    τfwhm = 2.5e-15
    grid = Grid.RealGrid(1, λ0, (70e-9, 4e-6), 500e-15)
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)
    Eω = input(grid, FT)
    @testset "gas propgation: $g" for g in PhysData.gas
        Eωgas = Fields.prop_material(Eω, grid, g, 10, λ0)
        Et = FT \ Eωgas
        gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
        ω0 = Maths.moment(grid.ω, abs2.(gab))
        @test ω0[1] < ω0[2]

        Eωgas = Fields.prop_material(Eω, grid, g, -10, λ0)
        Et = FT \ Eωgas
        gab = Maths.gabor(grid.t, Et, [-10e-15, 10e-15], 3e-15)
        ω0 = Maths.moment(grid.ω, abs2.(gab))
        @test ω0[1] > ω0[2]
    end

end
##
@testset "Compression" begin
# Short pulse with 100 fs^2
λ0 = 800e-9
τfwhm = 10e-15
grid = Grid.RealGrid(1, λ0, (400e-9, 1200e-9), 500e-15)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)
Et = FT \ Eω
Eωβ2 = Fields.prop_taylor(Eω, grid, [0, 0, 100e-30], λ0)
ϕs, Eωcomp = Fields.optcomp_taylor(Eωβ2, grid, λ0)
Etcomp = FT \ Eωcomp
@test ϕs[3] ≈ -100e-30
@test isapprox(Maths.fwhm(grid.t, abs2.(Maths.hilbert(Etcomp))), τfwhm; rtol=1e-3)

# Long pulse with 40000 fs^2 (stretches 220 fs to ~5 ps)
λ0 = 1030e-9
τfwhm = 220e-15
grid = Grid.RealGrid(1, λ0, (980e-9, 1080e-9), 20e-12)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)
Et = FT \ Eω
Eωβ2 = Fields.prop_taylor(Eω, grid, [0, 0, 4e-25], λ0)
ϕs, Eωcomp = Fields.optcomp_taylor(Eωβ2, grid, λ0)
Etcomp = FT \ Eωcomp
@test ϕs[3] ≈ -4e-25
@test isapprox(Maths.fwhm(grid.t, abs2.(Maths.hilbert(Etcomp))), τfwhm; rtol=1e-3)

# Short pulse with GDD and TOD
λ0 = 800e-9
τfwhm = 10e-15
grid = Grid.RealGrid(1, λ0, (400e-9, 1200e-9), 500e-15)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)
Et = FT \ Eω
Eωβ2 = Fields.prop_taylor(Eω, grid, [0, 0, 100e-30, 800e-45], λ0)
ϕs, Eωcomp = Fields.optcomp_taylor(Eωβ2, grid, λ0; order=3)
Etcomp = FT \ Eωcomp
@test all(ϕs .≈ [0, 0, -100e-30, -800e-45])
@test isapprox(Maths.fwhm(grid.t, abs2.(Maths.hilbert(Etcomp))), τfwhm; rtol=1e-3)

# Material insertion
λ0 = 800e-9
τfwhm = 10e-15
grid = Grid.RealGrid(1, λ0, (400e-9, 1200e-9), 500e-15)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)
Et = FT \ Eω
EωFS = Fields.prop_material(Eω, grid, :SiO2, 2e-3, λ0)
d, Eωcomp = Fields.optcomp_material(EωFS, grid, :SiO2, λ0, -1e-2, 1e-2)
Etcomp = FT \ Eωcomp
@test d ≈ -2e-3
@test isapprox(Maths.fwhm(grid.t, abs2.(Maths.hilbert(Etcomp))), τfwhm; rtol=1e-3)
end

@testset "Gratings" begin
# Common grating parameters: 1200 lines/mm, order -1, Littrow angle at 1030 nm
λ0 = 1030e-9
Λ = 1/(1200e3) # grating period in metres
m = -1
θi = Fields.littrow_angle(λ0, Λ; m)

# -- littrow_angle gives the right answer --
@test isapprox(θi, asin(λ0/(2Λ)))

# -- grating_GDD matches Treacy formula --
θm_ref = asin(m*λ0/Λ + sin(θi))
GDD_per_L = -m^2*λ0^3 / (π * PhysData.c^2 * Λ^2 * cos(θm_ref)^3)
@test isapprox(Fields.grating_GDD(λ0, Λ, m, θi), GDD_per_L)

# Helper: compute phase from the code's formula for independent GDD reference
function _grating_phase(ω, L, Λ, m, θi)
    λ = PhysData.wlfreq(ω)
    sinθm = m*λ/Λ + sin(θi)
    2*(ω*L/PhysData.c)*cos(asin(sinθm))
end

_grating_GDD(ω0, L, Λ, m, θi) = Maths.derivative(ω -> _grating_phase(ω, L, Λ, m, θi), ω0, 2)

ω0 = PhysData.wlfreq(λ0)

# -- GDD matches Treacy formula --
GDD_num = _grating_GDD(ω0, 1.0, Λ, m, θi)
@test isapprox(GDD_num, GDD_per_L, rtol=1e-4)

# -- Sign test: gratings should add negative chirp --
τfwhm = 50e-15
grid = Grid.RealGrid(1, λ0, (900e-9, 1200e-9), 10e-12)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)

Eωgrat = Fields.prop_gratings(Eω, grid, Λ, 5e-3, m, θi, λ0)
Et = FT \ Eωgrat
gab = Maths.gabor(grid.t, Et, [-200e-15, 200e-15], 50e-15)
ω0g = Maths.moment(grid.ω, abs2.(gab))
@test ω0g[1] > ω0g[2] # negative chirp: frequency decreases with time

# -- Out-of-band zeroing test --
λgrid = PhysData.wlfreq.(grid.ω)
inmask = @. abs(m*λgrid/Λ + sin(θi)) <= 1
@test all(Eωgrat[.!inmask] .== 0)

# -- Spectral magnitude preservation (phase-only within bandwidth) --
@test isapprox(abs.(Eωgrat[inmask]), abs.(Eω[inmask]), rtol=1e-10)

# -- GDD magnitude vs Treacy formula for two separations --
grid = Grid.RealGrid(1, λ0, (900e-9, 1200e-9), 20e-12)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
Eω = input(grid, FT)

@testset "Treacy GDD, L=$L" for L in [5e-3, 0.02]
    Eωgrat = Fields.prop_gratings(Eω, grid, Λ, L, m, θi, λ0)
    ϕs, Eωcomp = Fields.optcomp_taylor(Eωgrat, grid, λ0)
    GDD_expected = GDD_per_L * L
    @test isapprox(ϕs[3], -GDD_expected, rtol=0.02)
end

# -- optcomp_gratings: compress a chirped pulse --
# Use narrow-bandwidth pulse (200 fs) to minimise higher-order dispersion effects
τfwhm = 200e-15
grid = Grid.RealGrid(1, λ0, (950e-9, 1100e-9), 10e-12)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)

# Apply known positive GDD, then compress with gratings
L_target = 5e-3
GDD_applied = abs(GDD_per_L) * L_target
Eωchirped = Fields.prop_taylor(Eω, grid, [0, 0, GDD_applied], λ0)

L_opt, Eωcomp = Fields.optcomp_gratings(Eωchirped, grid, Λ, m, θi, 0.0, 0.05; λ0=λ0)
@test isapprox(L_opt, L_target, rtol=0.05)

Etcomp = FT \ Eωcomp
@test isapprox(Maths.fwhm(grid.t, abs2.(Maths.hilbert(Etcomp))), τfwhm; rtol=0.05)

# -- optcomp_gratings: auto-estimate (no explicit bounds) --
L_opt_auto, Eωcomp_auto = Fields.optcomp_gratings(Eωchirped, grid, Λ, m, θi; λ0=λ0)
@test isapprox(L_opt_auto, L_target, rtol=0.05)

# -- optcomp_gratings: round-trip (apply gratings then undo) --
# Apply gratings, add double positive GDD, then use optcomp_gratings to recompress
τfwhm = 200e-15
grid = Grid.RealGrid(1, λ0, (950e-9, 1100e-9), 10e-12)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)

L_applied = 5e-3
GDD_grating = GDD_per_L * L_applied
Eωchirped = Fields.prop_taylor(
    Fields.prop_gratings(Eω, grid, Λ, L_applied, m, θi, λ0),
    grid, [0, 0, -2*GDD_grating], λ0)

L_opt, Eωcomp = Fields.optcomp_gratings(Eωchirped, grid, Λ, m, θi, 0.0, 0.05; λ0=λ0)
@test isapprox(L_opt, L_applied, rtol=0.05)
Etcomp = FT \ Eωcomp
@test isapprox(Maths.fwhm(grid.t, abs2.(Maths.hilbert(Etcomp))), τfwhm; rtol=0.05)
end

@testset "Prisms" begin
# ---- Helper function tests ----

# Brewster angle for fused silica at 800 nm
λ0 = 800e-9
n_SiO2 = real(PhysData.ref_index_fun(:SiO2)(λ0))
θB = Fields.brewster_angle(:SiO2, λ0)
@test isapprox(θB, atan(n_SiO2))
@test isapprox(rad2deg(θB), 55.47, atol=0.1)  # well-known value for fused silica

# Minimum deviation apex angle for Brewster-cut fused silica prism
α_md = Fields.mindev_apex(:SiO2, λ0)
@test isapprox(α_md, 2*atan(1/n_SiO2))
@test isapprox(rad2deg(α_md), 69.05, atol=0.1)  # well-known value

# Also test with BK7 at 1030 nm
n_BK7 = real(PhysData.ref_index_fun(:BK7)(1030e-9))
θB_BK7 = Fields.brewster_angle(:BK7, 1030e-9)
@test isapprox(θB_BK7, atan(n_BK7))

# ---- prism_pair_GDD: negative GDD for angular dispersion (no insertion) ----
α = Fields.mindev_apex(:SiO2, λ0)
θi = Fields.brewster_angle(:SiO2, λ0)
GDD_per_L = Fields.prism_pair_GDD(λ0, :SiO2, α, θi)
@test GDD_per_L < 0  # angular dispersion gives negative GDD

# GDD should be negative and on the order of -100 to -1000 fs²/m for SiO2 at 800 nm
@test -1e-24 < GDD_per_L < -1e-28  # between ~-100 and -1e-4 fs²/m

# Helper: compute phase from ray tracing for independent GDD reference
# Geometry (A1, A2) and reference (D_ref, d_input) are PRE-COMPUTED and
# passed explicitly to ensure consistency across numerical derivative evaluations.
function _prism_phase_fixed(ω_val, A1, A2, n_func, α, θi, l1; D_ref, d_input, double_pass=true)
    ϕ, mask = Fields._prism_pair_phase([ω_val], n_func, α, θi, l1, A1, A2;
                                        D_ref=D_ref, d_input=d_input, double_pass=double_pass)
    mask[1] ? ϕ[1] : 0.0
end

# Pre-compute fixed geometry and reference for L=1.0, l1=0
n_SiO2_ref = real(PhysData.ref_index_fun(:SiO2)(λ0))
_, _, _A1_ref, _A2_ref = Fields._prism_apex_positions(α, θi, n_SiO2_ref, 0.0, 0.0; L=1.0)
_ref_res = Fields._trace_ray(n_SiO2_ref, α, θi, _A1_ref, _A2_ref, 0.0)
_D_ref_fixed = _ref_res.D
_ha_ref = α / 2
_d_input_fixed = cos(θi) * [cos(_ha_ref), -sin(_ha_ref)] + sin(θi) * [sin(_ha_ref), cos(_ha_ref)]
_n_func_SiO2 = PhysData.ref_index_fun(:SiO2)

_prism_GDD(ω0_val, L, material, α_val, θi_val) = begin
    n_c = real(PhysData.ref_index_fun(material)(PhysData.wlfreq(ω0_val)))
    _, _, A1, A2 = Fields._prism_apex_positions(α_val, θi_val, n_c, 0.0, 0.0; L=L)
    ref = Fields._trace_ray(n_c, α_val, θi_val, A1, A2, 0.0)
    ha = α_val / 2
    d_input = cos(θi_val) * [cos(ha), -sin(ha)] + sin(θi_val) * [sin(ha), cos(ha)]
    n_func = PhysData.ref_index_fun(material)
    Maths.derivative(
        ω -> _prism_phase_fixed(ω, A1, A2, n_func, α_val, θi_val, 0.0;
                                 D_ref=ref.D, d_input=d_input),
        ω0_val, 2)
end

ω0 = PhysData.wlfreq(λ0)

# GDD from phase second derivative should match prism_pair_GDD * L
GDD_num = _prism_GDD(ω0, 1.0, :SiO2, α, θi)
@test isapprox(GDD_num, GDD_per_L, rtol=0.02)

# ---- Analytical validation against Keller formula (Eq. 3.20) ----
# For Brewster/mindev double-pass: GDD/L = -4λ³(dn/dλ)²/(πc²)
dndλ = Maths.derivative(λv -> real(PhysData.ref_index_fun(:SiO2)(λv)), λ0, 1)
GDD_keller = -4 * λ0^3 * dndλ^2 / (π * PhysData.c^2)
@test isapprox(GDD_per_L, GDD_keller, rtol=0.01)

# ---- Sign test: prisms should add negative chirp (no insertion) ----
τfwhm = 50e-15
grid = Grid.RealGrid(1, λ0, (500e-9, 1200e-9), 10e-12)
x = Array{Float64}(undef, length(grid.t))
FT = FFTW.plan_rfft(x, 1)
input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)
Eω = input(grid, FT)

L_test = 0.5  # 50 cm separation
Eωprism = Fields.prop_prisms(Eω, grid, :SiO2, α, L_test, θi, 0.0, 0.0, λ0)
Et = FT \ Eωprism
gab = Maths.gabor(grid.t, Et, [-200e-15, 200e-15], 50e-15)
ω0g = Maths.moment(grid.ω, abs2.(gab))
@test ω0g[1] > ω0g[2]  # negative chirp: frequency decreases with time

# ---- TIR masking: frequencies with total internal reflection are zeroed ----
# Use a steep apex angle to force TIR at some frequencies
α_steep = deg2rad(80.0)
Eωsteep = Fields.prop_prisms(Eω, grid, :SiO2, α_steep, L_test, θi, 0.0, 0.0, λ0)
# At least some frequencies should be masked (zeroed)
n_func_test = PhysData.ref_index_fun(:SiO2)
n_center_steep = real(n_func_test(λ0))
_, _, A1s, A2s = Fields._prism_apex_positions(α_steep, θi, n_center_steep, 0.0, 0.0; L=L_test)
_, mask_steep = Fields._prism_pair_phase(grid.ω, n_func_test, α_steep, θi, 0.0, A1s, A2s)
@test any(.!mask_steep)   # some frequencies should be masked
@test all(Eωsteep[.!mask_steep] .== 0)  # those frequencies are zeroed

# ---- Spectral magnitude preservation (phase-only within bandwidth) ----
n_center_normal = real(n_func_test(λ0))
_, _, A1n, A2n = Fields._prism_apex_positions(α, θi, n_center_normal, 0.0, 0.0; L=L_test)
_, mask_normal = Fields._prism_pair_phase(grid.ω, n_func_test, α, θi, 0.0, A1n, A2n)
Eωprism_noλ0 = Fields.prop_prisms(Eω, grid, :SiO2, α, L_test, θi, 0.0, 0.0)
@test isapprox(abs.(Eωprism_noλ0[mask_normal]), abs.(Eω[mask_normal]), rtol=1e-10)

# ---- GDD magnitude vs numerical derivative for two separations ----
grid2 = Grid.RealGrid(1, λ0, (600e-9, 1100e-9), 20e-12)
x2 = Array{Float64}(undef, length(grid2.t))
FT2 = FFTW.plan_rfft(x2, 1)
Eω2 = input(grid2, FT2)

@testset "Prism GDD, L=$L" for L in [0.3, 0.8]
    Eωp = Fields.prop_prisms(Eω2, grid2, :SiO2, α, L, θi, 0.0, 0.0, λ0)
    ϕs, Eωcomp = Fields.optcomp_taylor(Eωp, grid2, λ0)
    GDD_expected = GDD_per_L * L
    @test isapprox(ϕs[3], -GDD_expected, rtol=0.1)
end

# ---- Material insertion adds positive GDD ----
# Use L_lightcon convention so that l1/l2 are independently specified;
# with Keller L the glass path in Prism 1 and Prism 2 cancel geometrically.
l_insert = 10e-3  # 10 mm insertion per prism
L_lc = 600e-3     # 600 mm Lightcon separation
n_insert_ref = real(PhysData.ref_index_fun(:SiO2)(λ0))
_n_func_ins = PhysData.ref_index_fun(:SiO2)
# No insertion
_, _, _A1_no, _A2_no = Fields._prism_apex_positions(α, θi, n_insert_ref, 0.0, 0.0; L_lightcon=L_lc)
_ref_no = Fields._trace_ray(n_insert_ref, α, θi, _A1_no, _A2_no, 0.0)
GDD_no_insert = Maths.derivative(ω -> _prism_phase_fixed(ω, _A1_no, _A2_no,
                                                          _n_func_ins, α, θi, 0.0;
                                                          D_ref=_ref_no.D, d_input=_d_input_fixed),
                                  ω0, 2)
# With insertion
_, _, _A1_ins, _A2_ins = Fields._prism_apex_positions(α, θi, n_insert_ref, l_insert, l_insert; L_lightcon=L_lc)
_ref_ins = Fields._trace_ray(n_insert_ref, α, θi, _A1_ins, _A2_ins, l_insert)
GDD_with_insert = Maths.derivative(ω -> _prism_phase_fixed(ω, _A1_ins, _A2_ins,
                                                            _n_func_ins, α, θi, l_insert;
                                                            D_ref=_ref_ins.D, d_input=_d_input_fixed),
                                    ω0, 2)
# With insertion, GDD should be more positive (less negative)
@test GDD_with_insert > GDD_no_insert

# ---- Lightcon calculator verification (1030 nm, SiO2, lookup=false) ----
# Reference values from Lightcon toolbox calculator:
#   https://toolbox.lightcon.com/tools/prismpair
# lambda = 1030 nm, L_lightcon = 600 mm, l1 = l2 = 10 mm, double pass
# n(1030nm) = 1.450473, Brewster = 55.409°, apex = 69.183°
# w = 570.804 mm, h = 229.165 mm
# d1 = 11.354 mm, d_free = 598.520 mm, d2 = 11.354 mm
# GDD = -543.513 fs², TOD = 2204.753 fs³

@testset "Lightcon verification" begin
    λ_lc = 1030e-9
    # Use lookup=false to get raw Sellmeier (matches reference code's Sellmeier)
    n_func_lc = PhysData.ref_index_fun(:SiO2, 1, PhysData.roomtemp; lookup=false)
    n_lc = real(n_func_lc(λ_lc))
    θi_lc = Fields.brewster_angle(:SiO2, λ_lc; lookup=false)
    α_lc = Fields.mindev_apex(:SiO2, λ_lc; lookup=false)

    @test isapprox(n_lc, 1.450473, atol=0.001)
    @test isapprox(rad2deg(θi_lc), 55.409, atol=0.1)
    @test isapprox(rad2deg(α_lc), 69.183, atol=0.1)

    # Geometry conversion: L_lightcon → (w, h)
    L_lc_val = 600e-3  # 600 mm
    l1_lc = 10e-3
    l2_lc = 10e-3
    w, h = Fields._wh_from_L_lightcon(L_lc_val, α_lc, θi_lc, n_lc, l1_lc, l2_lc)
    @test isapprox(w * 1e3, 570.804, atol=1.0)  # within 1 mm
    @test isapprox(h * 1e3, 229.165, atol=1.0)

    # Verify L_lightcon round-trip
    L_lc_rt = Fields._L_lightcon_from_wh(w, h, α_lc)
    @test isapprox(L_lc_rt, L_lc_val, rtol=1e-6)

    # Path lengths at center wavelength
    A1_lc = [0.0, 0.0]
    A2_lc = [w, -h]
    res_lc = Fields._trace_ray(n_lc, α_lc, θi_lc, A1_lc, A2_lc, l1_lc)
    @test isapprox(res_lc.d1_phys * 1e3, 11.354, atol=0.1)
    @test isapprox(res_lc.d_free * 1e3, 598.520, atol=1.0)
    @test isapprox(res_lc.d2_phys * 1e3, 11.354, atol=0.1)
    @test isapprox(res_lc.l2_actual * 1e3, l2_lc * 1e3, atol=0.5)

    # GDD/TOD from spectral phase (double pass)
    ω_lc = PhysData.wlfreq(λ_lc)
    # Fixed reference for consistent derivatives
    ref_lc = Fields._trace_ray(n_lc, α_lc, θi_lc, A1_lc, A2_lc, l1_lc)
    D_ref_lc = ref_lc.D
    ha_lc = α_lc / 2
    d_input_lc = cos(θi_lc) * [cos(ha_lc), -sin(ha_lc)] + sin(θi_lc) * [sin(ha_lc), cos(ha_lc)]
    function _phase_lc(ω_val)
        ϕ, mask = Fields._prism_pair_phase([ω_val], n_func_lc, α_lc, θi_lc, l1_lc,
                                            A1_lc, A2_lc; double_pass=true,
                                            D_ref=D_ref_lc, d_input=d_input_lc)
        mask[1] ? ϕ[1] : 0.0
    end
    GDD_lc = Maths.derivative(_phase_lc, ω_lc, 2)
    GDD_lc_fs2 = GDD_lc * 1e30
    @test isapprox(GDD_lc_fs2, -543.513, atol=15.0)  # within 15 fs²

    # Single vs double pass: ratio should be 2
    function _phase_lc_sp(ω_val)
        ϕ, mask = Fields._prism_pair_phase([ω_val], n_func_lc, α_lc, θi_lc, l1_lc,
                                            A1_lc, A2_lc; double_pass=false,
                                            D_ref=D_ref_lc, d_input=d_input_lc)
        mask[1] ? ϕ[1] : 0.0
    end
    GDD_sp = Maths.derivative(_phase_lc_sp, ω_lc, 2)
    @test isapprox(GDD_lc / GDD_sp, 2.0, rtol=0.01)
end

# ---- prop_prisms with L_lightcon keyword ----
@testset "L_lightcon keyword" begin
    λ_k = 1030e-9
    α_k = Fields.mindev_apex(:SiO2, λ_k; lookup=false)
    θi_k = Fields.brewster_angle(:SiO2, λ_k; lookup=false)

    grid_k = Grid.RealGrid(1, λ_k, (800e-9, 1300e-9), 10e-12)
    xk = Array{Float64}(undef, length(grid_k.t))
    FTk = FFTW.plan_rfft(xk, 1)
    input_k = Fields.GaussField(λ0=λ_k, τfwhm=200e-15, energy=1e-6)
    Eω_k = input_k(grid_k, FTk)

    # Compare L_lightcon keyword to equivalent (w,h) directly
    L_lc_test = 600e-3
    l1_k = 10e-3
    l2_k = 10e-3
    n_func_k = PhysData.ref_index_fun(:SiO2, 1, PhysData.roomtemp; lookup=false)
    n_k = real(n_func_k(λ_k))
    w_k, h_k = Fields._wh_from_L_lightcon(L_lc_test, α_k, θi_k, n_k, l1_k, l2_k)

    Eω_lc = Fields.prop_prisms(Eω_k, grid_k, :SiO2, α_k, 0.0, θi_k, l1_k, l2_k, λ_k;
                                L_lightcon=L_lc_test, lookup=false)
    Eω_wh = Fields.prop_prisms(Eω_k, grid_k, :SiO2, α_k, 0.0, θi_k, l1_k, l2_k, λ_k;
                                w=w_k, h=h_k, lookup=false)
    @test isapprox(Eω_lc, Eω_wh, rtol=1e-6)
end

# ---- optcomp_prisms: compress a chirped pulse (explicit bounds) ----
τfwhm_test = 200e-15
grid3 = Grid.RealGrid(1, λ0, (600e-9, 1100e-9), 10e-12)
x3 = Array{Float64}(undef, length(grid3.t))
FT3 = FFTW.plan_rfft(x3, 1)
input3 = Fields.GaussField(λ0=λ0, τfwhm=τfwhm_test, energy=1e-6)
Eω3 = input3(grid3, FT3)

# Apply known positive GDD, then compress with prisms
L_target = 0.3  # 30 cm
GDD_applied = abs(GDD_per_L) * L_target
Eωchirped = Fields.prop_taylor(Eω3, grid3, [0, 0, GDD_applied], λ0)

L_opt, Eωcomp = Fields.optcomp_prisms(Eωchirped, grid3, :SiO2, α, θi, 0.0, 0.0,
                                       0.05, 1.5; λ0=λ0)
@test isapprox(L_opt, L_target, rtol=0.15)

Etcomp = FT3 \ Eωcomp
@test isapprox(Maths.fwhm(grid3.t, abs2.(Maths.hilbert(Etcomp))), τfwhm_test; rtol=0.15)

# ---- optcomp_prisms: auto-estimate (no explicit bounds) ----
L_opt_auto, Eωcomp_auto = Fields.optcomp_prisms(Eωchirped, grid3, :SiO2, α, θi, 0.0, 0.0;
                                                  λ0=λ0)
@test isapprox(L_opt_auto, L_target, rtol=0.15)

# ---- optcomp_prisms: round-trip (apply prisms then undo) ----
τfwhm_rt = 200e-15
grid4 = Grid.RealGrid(1, λ0, (600e-9, 1100e-9), 10e-12)
x4 = Array{Float64}(undef, length(grid4.t))
FT4 = FFTW.plan_rfft(x4, 1)
input4 = Fields.GaussField(λ0=λ0, τfwhm=τfwhm_rt, energy=1e-6)
Eω4 = input4(grid4, FT4)

L_applied = 0.4  # 40 cm
GDD_prism = GDD_per_L * L_applied
Eωchirped_rt = Fields.prop_taylor(
    Fields.prop_prisms(Eω4, grid4, :SiO2, α, L_applied, θi, 0.0, 0.0, λ0),
    grid4, [0, 0, -2*GDD_prism], λ0)

L_opt_rt, Eωcomp_rt = Fields.optcomp_prisms(Eωchirped_rt, grid4, :SiO2, α, θi, 0.0, 0.0,
                                              0.05, 1.5; λ0=λ0)
# Wider tolerance (0.25) due to prism TOD that Taylor compensation does not capture
@test isapprox(L_opt_rt, L_applied, rtol=0.25)
Etcomp_rt = FT4 \ Eωcomp_rt
@test isapprox(Maths.fwhm(grid4.t, abs2.(Maths.hilbert(Etcomp_rt))), τfwhm_rt; rtol=0.15)
end

@testset "Gaussian beam initialisation" begin
    a = 16e-6
    gas = :Kr
    pres = 17.2
    τfwhm = 230e-15
    λ0 = 1030e-9
    energy = 5.2e-6
    modes = (
        Capillary.MarcatiliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=1, m=2, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=1, m=3, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=1, m=4, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=2, m=1, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=3, m=1, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=0, m=1, kind=:TE, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=0, m=1, kind=:TM, ϕ=0.0, loss=false)
    )
    inputs = Fields.gauss_beam_init(modes, 2π/λ0, a*0.64, Fields.GaussField, λ0=λ0, τfwhm=τfwhm, energy=energy)
    inputs = (inputs..., ((mode=i, fields=(Fields.ShotNoise(),)) for i=1:length(modes))...)
    @test inputs[1].fields[1].energy/energy ≈ 0.9807131210817726
    @test inputs[2].fields[1].energy/energy ≈ 0.006182621678046407
    @test inputs[3].fields[1].energy/energy ≈ 0.0013567813790567626
    @test inputs[4].fields[1].energy/energy ≈ 0.0008447236094573648
    @test inputs[5].fields[1].energy/energy < 1e-20
    @test inputs[6].fields[1].energy/energy < 2e-20
    @test inputs[7].fields[1].energy/energy < 1e-20
    @test inputs[8].fields[1].energy/energy < 1e-20

    # Now test that overlap integrals also work for diverging beams and produce 
    # sensible results
    a = 100e-6
    w0 = 0.64a
    λ = 800e-9
    k = 2π/λ
    zr = π*w0^2/λ

    mode = Capillary.MarcatiliMode(a)
    beam = Fields.normalised_gauss_beam(k, w0)
    @test abs2.(Modes.overlap(mode, beam)) ≈ 0.9807131210817726
    # test diverged beams
    beam2 = Fields.normalised_gauss_beam(k, w0; z=zr)
    @test abs2.(Modes.overlap(mode, beam2)) < abs2.(Modes.overlap(mode, beam))
    beam2 = Fields.normalised_gauss_beam(k, w0; z=-zr)
    @test abs2.(Modes.overlap(mode, beam2)) < abs2.(Modes.overlap(mode, beam))
end

@testset "DataField" begin
    # create a Gaussian pulse with a delay in the frequency domain, write it to a file,
    # load it back in a check it produces the correct pulse
    import DelimitedFiles: writedlm, readdlm
    import FFTW
    λlims = (200e-9, 4e-6)
    trange = 200e-15
    τfwhm = 10e-15
    τ0 = 20e-15
    λ0 = 800e-9
    f0 = PhysData.c/λ0
    σt = Maths.fwhm_to_σ(τfwhm)
    σf = 1/(4π*σt)
    f = collect(range(PhysData.c/λlims[2], PhysData.c/λlims[1]; length=2048))
    If = Maths.gauss.(f, σf; x0=f0)
    ϕ = @. -2π*τ0*(f-f0) # Fourier transform in the maths convention here--pos. delay = neg. slope
    dat = [f If ϕ]
    grid = Grid.RealGrid(1, λ0, λlims, trange)
    FT = FFTW.plan_rfft(copy(grid.t), 1)
    field = mktempdir() do td
        tf = joinpath(td, tempname())
        writedlm(tf, dat, ' ')

        @test readdlm(tf, ' ') == dat

        Fields.DataField(tf; energy=1e-6)
    end
    Eω = field(grid, FT)
    t, Et = Processing.getEt(grid, Eω)
    @test isapprox(Maths.fwhm(t, abs2.(Et)), τfwhm, rtol=1e-5)
    @test isapprox(Maths.moment(t, abs2.(Et)), τ0, rtol=1e-5)
end

@testset "CEP optimisation" begin
    τfwhm = 3e-15
    λ0 = 800e-9
    energy = 1e-6
    grid = Grid.RealGrid(1.0, λ0, (100e-9, 3000e-9), 500e-15)
    δt = grid.t[2] - grid.t[1]
    ϕCEO = δt*PhysData.wlfreq(λ0)
    energy_t = Fields.energyfuncs(grid)[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)

    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
    Eω = input(grid, FT)
    ϕopt, Eωopt = Fields.optfield_cep(Eω, grid)
    Et = FT \ Eωopt
    It = abs2.(Maths.hilbert(Et))
    # check that the optimisation has found the correct value
    @test isapprox(ϕopt, ϕCEO, rtol=1e-6)
    @test isapprox(getceo(grid.t, Et, It, PhysData.wlfreq(λ0)), 0.0, rtol=1e-15, atol=1e-15)

    Eωm = [Eω zero(Eω)]
    ϕoptm, Eωopt = Fields.optfield_cep(Eωm, grid)
    @test size(Eωopt) == size(Eωm)
    @test ϕoptm == ϕopt
    nCEO = 4
    Eωmm = zeros(ComplexF64, (length(grid.ω), 2, nCEO))
    for ii in 1:nCEO
        ϕCEO = δt*PhysData.wlfreq(λ0)*ii
        input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=[ϕCEO])
        Eω = input(grid, FT)
        Eωmm[:, 1, ii] .= Eω
    end
    ϕoptmm, Eωoptmm = Fields.optfield_cep(Eωmm, grid)
    @test size(Eωoptmm) == size(Eωmm)
    for ii in 1:nCEO
        ϕCEO = δt*PhysData.wlfreq(λ0)*ii
        @test isapprox(ϕoptmm[ii], ϕCEO, rtol=1e-6)
    end
end

@testset "free-space inputs: radial" begin
    λ0 = 800e-9
    τfwhm = 10e-15
    energy = 1e-6

    w0 = 100e-6
    propz = 1.0

    zr = π*w0^2/λ0
    w1 = w0*sqrt(1 + (propz/zr)^2)
    
    R = 4w1
    N = 1024

    grid = Grid.EnvGrid(1, λ0, (400e-9, 6e-6), 100e-15)

    q = Hankel.QDHT(R, N, dim=2)

    xt = zeros(Float64, length(grid.t), length(q.r))
    FT = FFTW.plan_fft(xt, 1, flags=FFTW.ESTIMATE)

    Eωk = Fields.GaussGaussField(;λ0, τfwhm, energy, w0, propz)(grid, q, FT)

    r = Hankel.Rsymmetric(q)
    Eωr = Hankel.symmetric(q \ Eωk, q)

    Iωr = abs2.(Eωr)
    Ir = dropdims(sum(Iωr; dims=1); dims=1)
    w1q = 2Maths.rms_width(r, Ir)

    @test isapprox(w1q, w1; rtol=1e-3)
end

@testset "free-space inputs: full 3D" begin
    λ0 = 800e-9
    τfwhm = 10e-15
    energy = 1e-6

    w0 = 100e-6
    propz = 1.0

    zr = π*w0^2/λ0
    w1 = w0*sqrt(1 + (propz/zr)^2)
    
    R = 2w1
    N = 256
    grid = Grid.EnvGrid(1, λ0, (400e-9, 6e-6), 100e-15)
    xygrid = Grid.FreeGrid(R, N)

    xr = Array{ComplexF64}(undef, length(grid.t), length(xygrid.y), length(xygrid.x))
    FT = FFTW.plan_fft(xr, (1, 2, 3), flags=FFTW.ESTIMATE)

    Eωk = Fields.GaussGaussField(;λ0, τfwhm, energy, w0, propz)(grid, xygrid, FT)

    Etxy = FT \ Eωk

    Ixy = dropdims(sum(abs2.(Etxy); dims=1); dims=1)

    Ix = dropdims(sum(Ixy; dims=1); dims=1)
    Iy = dropdims(sum(Ixy; dims=2); dims=2)

    w1x = 2Maths.rms_width(xygrid.x, Ix)
    w1y = 2Maths.rms_width(xygrid.y, Iy)

    @test isapprox(w1x, w1; rtol=1e-2)
    @test isapprox(w1y, w1; rtol=1e-2)
end
