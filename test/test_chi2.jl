import Test: @test, @testset
import Luna: Fields, Grid, Nonlinear, PhysData
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
