using Luna
import Test: @test, @testset, @test_throws
import Luna: PhysData, Nonlinear, NonlinearRHS, Grid, Processing
import Logging

logger = Logging.SimpleLogger(stdout, Logging.Warn)
old_logger = Logging.global_logger(logger)

@testset "β₂_TPA coefficient values" begin
    # 300 nm → two-photon energy = 2 * 1240/300 = 8.27 eV < E_g = 8.3 eV → below edge
    ω_300 = 2π * PhysData.c / 300e-9
    @test PhysData.β₂_TPA(ω_300, :SiO2) == 0.0

    # 264 nm → 2ℏω = 2 * 1240/264 ≈ 9.39 eV > 8.3 eV → nonzero
    ω_264 = 2π * PhysData.c / 264e-9
    β₂_264 = PhysData.β₂_TPA(ω_264, :SiO2)
    @test β₂_264 > 0
    @test 1e-16 < β₂_264 < 1e-12  # order of magnitude check

    # 213 nm → 2ℏω ≈ 11.6 eV → much stronger
    ω_213 = 2π * PhysData.c / 213e-9
    β₂_213 = PhysData.β₂_TPA(ω_213, :SiO2)
    @test β₂_213 > β₂_264  # monotonicity
end

@testset "TPAResponse coefficient conversion" begin
    Nω = 100
    ω_grid = range(0.5e15, 2.0e15, length=Nω)
    # Synthetic β₂ values: nonzero above some threshold
    β₂_ω = [ω > 1.0e15 ? 1e-12 : 0.0 for ω in ω_grid]

    tpa = Nonlinear.TPAResponse(collect(ω_grid), β₂_ω)

    # coeff should be purely imaginary (negative imaginary part) where β₂ > 0
    for i in eachindex(ω_grid)
        if β₂_ω[i] > 0
            @test real(tpa.coeff_ω[i]) ≈ 0.0 atol=1e-40
            @test imag(tpa.coeff_ω[i]) < 0  # -i gives negative imaginary
        else
            @test tpa.coeff_ω[i] == 0.0
        end
    end

    # Check magnitude: coeff = -i * ε₀² * c² * β₂ / (2ω)
    idx = findfirst(x -> x > 0, β₂_ω)
    expected = -im * PhysData.ε_0^2 * PhysData.c^2 * β₂_ω[idx] / (2 * ω_grid[idx])
    @test tpa.coeff_ω[idx] ≈ expected

    # Callable: adds to output buffer
    out_ω = zeros(ComplexF64, Nω)
    F_E2E = ones(ComplexF64, Nω)
    ρ = 1.0
    tpa(out_ω, F_E2E, ρ)
    @test all(out_ω .== tpa.coeff_ω)

    # DimensionMismatch
    @test_throws DimensionMismatch Nonlinear.TPAResponse([1.0, 2.0], [1.0])
end

@testset "TPA energy loss — mode-averaged" begin
    # Propagate a pulse through a thin medium with TPA only (n2≈0).
    # TPA should cause energy loss.
    λ0 = 240e-9
    τfwhm = 5e-15
    energy = 20e-9  # 1 nJ
    flength = 100e-6  # 100 µm

    β2 = 1e-30  # small normal dispersion
    βs = [0.0, 0.0, β2]
    Aeff = π * (12e-6)^2  # realistic effective area
    n2_val = 1e-25  # very small n2 (effectively no Kerr)
    λlims = [160e-9, 600e-9]
    trange = 1e-12

    # Synthetic TPA: constant β₂_TPA = 1e-11 m/W across band (strong enough to see effect)
    β₂_tpa_val = 1e-11  # ~1 cm/GW
    tpa_func = ω -> β₂_tpa_val

    # With TPA
    output_tpa = prop_gnlse(flength, βs; n2=n2_val, Aeff, λ0, τfwhm, energy,
                            pulseshape=:gauss, λlims, trange,
                            raman=false, shock=false, shotnoise=false, tpa=tpa_func)

    # Without TPA
    output_notpa = prop_gnlse(flength, βs; n2=n2_val, Aeff, λ0, τfwhm, energy,
                              pulseshape=:gauss, λlims, trange,
                              raman=false, shock=false, shotnoise=false)

    # Energy comparison
    Eω_in_tpa = output_tpa["Eω"][:, 1]
    Eω_out_tpa = output_tpa["Eω"][:, end]
    energy_in = sum(abs2.(Eω_in_tpa))
    energy_out_tpa = sum(abs2.(Eω_out_tpa))

    Eω_out_notpa = output_notpa["Eω"][:, end]
    energy_out_notpa = sum(abs2.(Eω_out_notpa))

    # Without TPA, energy should be conserved (no loss, no Raman)
    @test isapprox(energy_out_notpa, sum(abs2.(output_notpa["Eω"][:, 1])), rtol=1e-13)

    # With TPA, output energy should be less than input
    @test energy_out_tpa < energy_in
    # And less than without TPA
    @test energy_out_tpa < energy_out_notpa

    # Fractional loss should be positive and reasonable (not 100%)
    frac_loss = 1 - energy_out_tpa / energy_in
    @test 0.7 < frac_loss < 1
end

@testset "TPA spectral suppression" begin
    # Broadband pulse: TPA with frequency-dependent β₂ should suppress
    # short-wavelength components more than long-wavelength ones.
    λ0 = 240e-9
    τfwhm = 5e-15  # short pulse = broad bandwidth
    energy = 20e-9
    flength = 100e-6
    β2 = 1e-30
    βs = [0.0, 0.0, β2]
    Aeff = π * (12e-6)^2
    n2_val = PhysData.n2(:SiO2)
    λlims = [160e-9, 600e-9]
    trange = 1e-12

    tpa_func = ω -> PhysData.β₂_TPA(ω, :SiO2)

    output_tpa = prop_gnlse(flength, βs; n2=n2_val, Aeff, λ0, τfwhm, energy,
                            pulseshape=:gauss, λlims, trange,
                            raman=false, shock=false, shotnoise=false, tpa=tpa_func)

    output_notpa = prop_gnlse(flength, βs; n2=n2_val, Aeff, λ0, τfwhm, energy,
                              pulseshape=:gauss, λlims, trange,
                              raman=false, shock=false, shotnoise=false)

    ω = output_tpa["grid"]["ω"]
    spec_tpa = abs2.(output_tpa["Eω"][:, end])
    spec_notpa = abs2.(output_notpa["Eω"][:, end])

    # Find spectral ratio (TPA / no-TPA) at two frequencies:
    ω_below = 2π * PhysData.c / 250e-9  # less TPA
    ω_above = 2π * PhysData.c / 220e-9  # more TPA

    idx_below = argmin(abs.(ω .- ω_below))
    idx_above = argmin(abs.(ω .- ω_above))

    # Below edge: spectrum should be essentially unchanged
    if spec_notpa[idx_below] > 1e-30  # only test if there's signal
        ratio_below = spec_tpa[idx_below] / spec_notpa[idx_below]
        ratio_above = spec_tpa[idx_above] / spec_notpa[idx_above]
        # Above-edge suppression should be stronger (lower ratio)
        @test ratio_above < ratio_below
    end
end

@testset "TPA spectral suppression — radial free-space" begin
    import Luna: Grid, Nonlinear, NonlinearRHS, LinearOps, Output, Fields
    import Luna.PhysData: wlfreq
    import FFTW
    import Hankel

    # Deep-UV Gaussian beam in SiO₂ — envelope, radially symmetric
    λ0 = 240e-9
    τfwhm = 2e-15   # broad bandwidth across TPA region
    energy = 150e-9
    w0 = 15e-6        # beam waist
    L = 10e-6        # 100 µm propagation (thin slab)
    R = 250e-6        # Hankel aperture
    N = 256            # radial points

    grid = Grid.EnvGrid(L, λ0, (160e-9, 600e-9), 1e-12)
    q = Hankel.QDHT(R, N, dim=2)

    densityfun(z) = 1.0  # solid — χ³ already bulk value
    nfun = PhysData.ref_index_fun(:SiO2, lookup=false)
    normfun = NonlinearRHS.const_norm_radial(grid, q, nfun)
    linop = LinearOps.make_const_linop(grid, q, nfun)

    χ3_SiO2 = PhysData.χ3(:SiO2)
    β₂_ω = PhysData.β₂_TPA.(grid.ω, :SiO2)
    tpa = Nonlinear.TPAResponse(grid.ω, β₂_ω)

    inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, w0=w0)

    # With TPA
    responses_tpa = (Nonlinear.Kerr_env(χ3_SiO2), tpa)
    Eω_tpa, transform_tpa, FT_tpa = Luna.setup(grid, q, densityfun, normfun,
                                                 responses_tpa, inputs)
    output_tpa = Output.MemoryOutput(0, grid.zmax, 201)
    Luna.run(Eω_tpa, grid, linop, transform_tpa, FT_tpa, output_tpa)

    # Without TPA
    responses_notpa = (Nonlinear.Kerr_env(χ3_SiO2),)
    Eω_notpa, transform_notpa, FT_notpa = Luna.setup(grid, q, densityfun, normfun,
                                                       responses_notpa, inputs)
    output_notpa = Output.MemoryOutput(0, grid.zmax, 201)
    Luna.run(Eω_notpa, grid, linop, transform_notpa, FT_notpa, output_notpa)

    # Energy comparison (sum over all radial k-modes)
    Eωk_tpa_in = output_tpa.data["Eω"][:, :, 1]
    Eωk_tpa_out = output_tpa.data["Eω"][:, :, end]
    Eωk_notpa_out = output_notpa.data["Eω"][:, :, end]

    energy_in = sum(abs2, Eωk_tpa_in)
    energy_out_tpa = sum(abs2, Eωk_tpa_out)
    energy_out_notpa = sum(abs2, Eωk_notpa_out)

    # With TPA, output energy should be less than without
    @test energy_out_tpa < energy_out_notpa

    # Spectral suppression: TPA stronger at shorter wavelengths (higher ω)
    # Sum over radial modes for spectral comparison
    spec_tpa = dropdims(sum(abs2, Eωk_tpa_out, dims=2), dims=2)
    spec_notpa = dropdims(sum(abs2, Eωk_notpa_out, dims=2), dims=2)

    ω_below = 2π * PhysData.c / 300e-9   # below SiO₂ TPA edge (~299 nm)
    ω_above = 2π * PhysData.c / 220e-9   # well above TPA edge

    idx_below = argmin(abs.(grid.ω .- ω_below))
    idx_above = argmin(abs.(grid.ω .- ω_above))

    if spec_notpa[idx_below] > 1e-30 && spec_notpa[idx_above] > 1e-30
        ratio_below = spec_tpa[idx_below] / spec_notpa[idx_below]
        ratio_above = spec_tpa[idx_above] / spec_notpa[idx_above]
        # Above-edge suppression should be stronger (lower ratio)
        @test ratio_above < ratio_below
        @test ratio_above < 0.6  # significant suppression above edge
    end
end

##
Logging.global_logger(old_logger)
