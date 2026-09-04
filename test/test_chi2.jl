import Test: @test, @testset
import Luna: Fields, Grid, Maths, Nonlinear, PhysData
import Luna.PhysData: ε_0
import FFTW

@testset "Chi2 field products" begin
	Ec = [1.0, -2.0, 0.5]
	Enl = zeros(6)
	Nonlinear.field_products!(Enl, Ec)
	@test Enl == [1.0, 4.0, 0.25, -2.0, 1.0, -4.0]
end

@testset "Chi2Field without rotation" begin
	χ2 = zeros(3, 6)
	χ2[1,1] = 1.2
	χ2[1,2] = -0.7
	χ2[1,6] = 0.3
	χ2[2,2] = 0.5
	χ2[2,6] = -0.4

	c = Nonlinear.Chi2Field(0.0, 0.0, χ2)
	E = [1.0 2.0;
		 -0.5 0.25;
		 0.0 -1.0]

	out = zeros(size(E))
	c(out, E, 1.0)

	expected = similar(out)
	for i in axes(E, 1)
		Ex = E[i, 1]
		Ey = E[i, 2]
		Enl1 = Ex^2
		Enl2 = Ey^2
		Enl6 = 2*Ex*Ey

		expected[i, 1] = ε_0*(χ2[1,1]*Enl1 + χ2[1,2]*Enl2 + χ2[1,6]*Enl6)
		expected[i, 2] = ε_0*(χ2[2,2]*Enl2 + χ2[2,6]*Enl6)
	end

	@test isapprox(out, expected, rtol=1e-14, atol=0.0)

	c(out, E, 1.0)
	@test isapprox(out, 2 .* expected, rtol=1e-14, atol=0.0)
end

@testset "Chi2Field with rotation" begin
	χ2 = reshape(collect(0.1:0.1:1.8), 3, 6)
	θ = 0.37
	ϕ = -1.1

	c = Nonlinear.Chi2Field(θ, ϕ, χ2)
	E = [0.8 -0.2;
		 -0.1 0.4;
		 0.0 0.6;
		 -0.5 -0.3]

	out = zeros(size(E))
	c(out, E, 1.0)

	expected = zeros(size(E))
	El = zeros(3)
	for i in axes(E, 1)
		El[1] = E[i, 1]
		El[2] = E[i, 2]
		El[3] = 0.0

		Ec = c.toCrystal * El
		Enl = zeros(6)
		Nonlinear.field_products!(Enl, Ec)
		Pc = c.χ2 * Enl
		Pl = c.toLab * Pc

		expected[i, 1] = ε_0*Pl[1]
		expected[i, 2] = ε_0*Pl[2]
	end

	@test isapprox(out, expected, rtol=1e-13, atol=1e-15)
end

@testset "Chi2Field generates second harmonic" begin
	λ0 = 800e-9
	grid = Grid.RealGrid(1.0, λ0, (200e-9, 1600e-9), 4e-12)
	input = Fields.GaussField(λ0=λ0, τfwhm=800e-15, power=1.0)
	Et = 1e10 .* Fields.make_Et(input, grid)
	N = length(Et)

	E = zeros(N, 2)
	E[:, 1] .= Et

	c = Nonlinear.Chi2Field(0.0, 0.0, PhysData.χ2(:BBO))

	out = zeros(size(E))
	c(out, E, 1.0)

	ω = grid.ω
	ω0 = PhysData.wlfreq(λ0)
	fundband = @. (0.9ω0 < ω < 1.1ω0)
	shband = @. (1.9ω0 < ω < 2.1ω0)

	Pωy = abs2.(FFTW.rfft(out[:, 2] ./ ε_0))
	fundamental = sum(Pωy[fundband])
	second_harmonic = sum(Pωy[shband])

	@test second_harmonic > 1e8 * max(fundamental, eps())
	@test maximum(abs.(out[:, 1])) < 1e-15
end

@testset "Chi2Env without rotation" begin
	χ2 = zeros(3, 6)
	χ2[1,1] = 1.2
	χ2[1,2] = -0.7
	χ2[1,6] = 0.3
	χ2[2,2] = 0.5
	χ2[2,6] = -0.4

	ω0 = 2.35e15
	t = [0.0, 0.3e-15, 0.7e-15]
	c = Nonlinear.Chi2Env(0.0, 0.0, χ2, ω0, t)
	E = [1.0+0.5im 2.0-1.0im;
		 -0.5+0.0im 0.25+0.3im;
		 0.0-0.2im -1.0+0.1im]

	out = zeros(ComplexF64, size(E))
	c(out, E, 1.0)

	expected = similar(out)
	for i in axes(E, 1)
		Ax = E[i, 1]
		Ay = E[i, 2]
		cp = 0.5*exp(1im*ω0*t[i]) # SFG (ω + ω → 2ω)
		cm = exp(-1im*ω0*t[i]) # DFG (2ω - ω → ω)
		Anl1 = cp*Ax^2 + cm*abs2(Ax)
		Anl2 = cp*Ay^2 + cm*abs2(Ay)
		Anl6 = 2*(cp*Ax*Ay + cm*real(Ax*conj(Ay)))

		expected[i, 1] = ε_0*(χ2[1,1]*Anl1 + χ2[1,2]*Anl2 + χ2[1,6]*Anl6)
		expected[i, 2] = ε_0*(χ2[2,2]*Anl2 + χ2[2,6]*Anl6)
	end

	@test isapprox(out, expected, rtol=1e-14, atol=0.0)

	c(out, E, 1.0)
	@test isapprox(out, 2 .* expected, rtol=1e-14, atol=0.0)
end

@testset "Chi2Env with rotation" begin
	χ2 = reshape(collect(0.1:0.1:1.8), 3, 6)
	θ = 0.37
	ϕ = -1.1
	ω0 = 2.35e15
	t = [0.0, 0.4e-15, 0.9e-15, 1.3e-15]

	c = Nonlinear.Chi2Env(θ, ϕ, χ2, ω0, t)
	E = [0.8-0.3im -0.2+0.1im;
		 -0.1+0.6im 0.4-0.5im;
		 0.0+0.2im 0.6+0.0im;
		 -0.5-0.1im -0.3+0.4im]

	out = zeros(ComplexF64, size(E))
	c(out, E, 1.0)

	expected = zeros(ComplexF64, size(E))
	Al = zeros(ComplexF64, 3)
	for i in axes(E, 1)
		Al[1] = E[i, 1]
		Al[2] = E[i, 2]
		Al[3] = 0.0

		cp = 0.5*exp(1im*ω0*t[i])
		cm = exp(-1im*ω0*t[i])
		Ac = c.toCrystal * Al
		Anl = zeros(ComplexF64, 6)
		Nonlinear.env_products!(Anl, Ac, cp, cm)
		Pc = c.χ2 * Anl
		Pl = c.toLab * Pc

		expected[i, 1] = ε_0*Pl[1]
		expected[i, 2] = ε_0*Pl[2]
	end

	@test isapprox(out, expected, rtol=1e-13, atol=1e-15)
end

@testset "Chi2Env matches Chi2Field" begin
	# two-colour, two-polarisation field so that both the SFG term (SHG band)
	# and the DFG term (back-conversion into the fundamental band) are exercised
	Nt = collect(range(0, length=2^16))
	t = @. (Nt - 2^16/2)*3.430944979182369e-16/4
	ω0 = 2π*PhysData.c/800e-9
	env = @. exp(-0.5*(t/10e-15)^2)
	Ex = @. env*cos(ω0*t) + 0.3*env*cos(2ω0*t)
	Ey = @. 0.5*env*cos(ω0*t + 0.4)
	E = hcat(Ex, Ey)

	χ2 = reshape(collect(0.1:0.1:1.8), 3, 6)
	θ = 0.37
	ϕ = -1.1

	cf = Nonlinear.Chi2Field(θ, ϕ, χ2)
	outf = zeros(size(E))
	cf(outf, E, 1.0)

	# hilbert gives the analytic signal A·exp(iω0t); demodulate to get the envelope
	A = Maths.hilbert(E) .* exp.(-1im*ω0.*t)
	ce = Nonlinear.Chi2Env(θ, ϕ, χ2, ω0, t)
	oute = zeros(ComplexF64, size(A))
	ce(oute, A, 1.0)
	# remove envelope content at negative absolute frequencies (e.g. the conjugate DFG
	# products at -ω0)--in a propagation simulation this is done by the grid apodisation,
	# since the frequency window of an EnvGrid only contains positive frequencies
	outeν = FFTW.fft(oute, 1)
	ν = 2π .* FFTW.fftfreq(length(t), 1/(t[2] - t[1])) # offset frequency; absolute is ω0 + ν
	outeν[ν .<= -ω0, :] .= 0
	oute = FFTW.ifft(outeν, 1)
	# reconstruct the real polarisation from the envelope result
	outer = real.(oute .* exp.(1im*ω0.*t))

	outfω = FFTW.rfft(outf, 1)
	outeω = FFTW.rfft(outer, 1)
	ω = 2π .* FFTW.rfftfreq(length(t), 1/(t[2] - t[1]))
	fundband = @. (0.9ω0 < ω < 1.1ω0)
	shband = @. (1.9ω0 < ω < 2.1ω0)
	# compare only in the fundamental and SH bands: the envelope response differs from the
	# real-field response only in the rectification content near ω = 0, which in a
	# propagation simulation is removed by the grid apodisation
	for pol in 1:2
		for band in (fundband, shband)
			mask = band .& (abs.(outfω[:, pol]) .> 1e-6*maximum(abs.(outfω[band, pol])))
			# not exact because of spectral leakage of the (removed) rectification content
			@test isapprox(abs.(outeω[mask, pol]), abs.(outfω[mask, pol]), rtol=1e-6)
		end
	end
end

@testset "Chi2Env generates second harmonic" begin
	λ0 = 800e-9
	grid = Grid.EnvGrid(1.0, λ0, (200e-9, 1600e-9), 4e-12; thg=true)
	A = 1e10 .* sqrt.(Maths.gauss.(grid.to; fwhm=800e-15))
	N = length(A)

	E = zeros(ComplexF64, N, 2)
	E[:, 1] .= A

	c = Nonlinear.Chi2Env(0.0, 0.0, PhysData.χ2(:BBO), grid.ω0, grid.to)

	out = zeros(ComplexF64, size(E))
	c(out, E, 1.0)

	ωo = grid.ωo
	ω0 = grid.ω0
	fundband = @. (0.9ω0 < ωo < 1.1ω0)
	shband = @. (1.9ω0 < ωo < 2.1ω0)
	recband = @. abs(ωo) < 0.1ω0

	Pωy = abs2.(FFTW.fft(out[:, 2] ./ ε_0))
	fundamental = sum(Pωy[fundband])
	second_harmonic = sum(Pωy[shband])
	rectification = sum(Pωy[recband])

	@test second_harmonic > 1e8 * max(fundamental, eps())
	# the DFG term produces optical rectification for a single-colour input
	@test rectification > 1e8 * max(fundamental, eps())
	@test maximum(abs.(out[:, 1])) < 1e-15
end
