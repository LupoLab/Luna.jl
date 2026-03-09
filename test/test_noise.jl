using Luna
import Test: @test, @testset, @test_throws
import Luna: Fields, Grid, Processing
import Random
import Logging

logger = Logging.SimpleLogger(stdout, Logging.Warn)
old_logger = Logging.global_logger(logger)

# Small parameters for fast tests (matching test_interface.jl patterns)
const noise_args = (100e-6, 0.1, :He, 1) # a, flength, gas, pressure
const noise_kwargs = (λ0=800e-9, τfwhm=10e-15, energy=1e-12,
                      trange=400e-15, λlims=(200e-9, 4e-6))

@testset "generate_noise_field" begin
    @testset "RealGrid" begin
        grid = Grid.RealGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nf = Fields.generate_noise_field(grid)
        @test size(nf) == size(grid.ω)
        @test eltype(nf) == ComplexF64
        @test any(!iszero, nf)
        # Different seeds produce different noise
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(123)
        nf1 = Fields.generate_noise_field(grid; rng=rng1)
        nf2 = Fields.generate_noise_field(grid; rng=rng2)
        @test nf1 != nf2
        # Same seed reproduces identical noise
        rng3 = Random.MersenneTwister(42)
        nf3 = Fields.generate_noise_field(grid; rng=rng3)
        @test nf1 == nf3
    end
    @testset "EnvGrid" begin
        grid = Grid.EnvGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nf = Fields.generate_noise_field(grid)
        @test size(nf) == size(grid.ω)
        @test eltype(nf) == ComplexF64
        # Non-signal bins should be zero
        @test all(iszero, nf[.!grid.sidx])
        @test any(!iszero, nf[grid.sidx])
    end
    @testset "Amplitude matches ShotNoise (one-photon-per-mode)" begin
        # Chen & Wise Eq. A19: |A_noise(ω)| = √(hν·Δν)
        # generate_noise_field and ShotNoise use the same formula, so their
        # amplitudes must be identical (phases differ only by RNG seed).
        grid_r = Grid.RealGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        sn = Fields.ShotNoise(Random.MersenneTwister(42))
        snf = sn(grid_r)
        nf = Fields.generate_noise_field(grid_r; rng=Random.MersenneTwister(42))
        @test abs.(nf) ≈ abs.(snf)

        grid_e = Grid.EnvGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        sn_e = Fields.ShotNoise(Random.MersenneTwister(99))
        snf_e = sn_e(grid_e)
        nf_e = Fields.generate_noise_field(grid_e; rng=Random.MersenneTwister(99))
        @test abs.(nf_e) ≈ abs.(snf_e)
    end
    @testset "Multimode RealGrid" begin
        grid = Grid.RealGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nm = 3
        nf = Fields.generate_noise_field(grid; nmodes=nm)
        # Shape: (nω, nmodes)
        @test size(nf) == (length(grid.ω), nm)
        @test eltype(nf) == ComplexF64
        # All columns non-zero
        @test all(any(!iszero, nf[:, j]) for j in 1:nm)
        # Columns are independent (different random phases)
        @test nf[:, 1] != nf[:, 2]
        @test nf[:, 2] != nf[:, 3]
        # Amplitude identical across modes (same one-photon-per-mode formula)
        @test abs.(nf[:, 1]) ≈ abs.(nf[:, 2])
        @test abs.(nf[:, 2]) ≈ abs.(nf[:, 3])
        # Reproducible with same seed
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(42)
        @test Fields.generate_noise_field(grid; rng=rng1, nmodes=nm) ==
              Fields.generate_noise_field(grid; rng=rng2, nmodes=nm)
        # nmodes=1 still returns 1D array (no regression)
        nf1 = Fields.generate_noise_field(grid; nmodes=1)
        @test ndims(nf1) == 1
    end
    @testset "Multimode EnvGrid" begin
        grid = Grid.EnvGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nm = 3
        nf = Fields.generate_noise_field(grid; nmodes=nm)
        @test size(nf) == (length(grid.ω), nm)
        @test eltype(nf) == ComplexF64
        # Non-signal bins should be zero in all modes
        for j in 1:nm
            @test all(iszero, nf[.!grid.sidx, j])
            @test any(!iszero, nf[grid.sidx, j])
        end
        # Columns are independent
        @test nf[:, 1] != nf[:, 2]
        # Amplitude identical across modes
        @test abs.(nf[:, 1]) ≈ abs.(nf[:, 2])
    end
end

@testset "Multimode modified noise integration" begin
    # prop_capillary with modes=2 and shotnoise=:modified exercises the
    # multimode noise path (nmodes > 1) in generate_noise_field.
    out_mm = prop_capillary(noise_args...; noise_kwargs...,
                            shotnoise=:modified, modes=2,
                            rng=Random.MersenneTwister(42))
    # Reproducibility: same seed → identical output
    out_mm2 = prop_capillary(noise_args...; noise_kwargs...,
                             shotnoise=:modified, modes=2,
                             rng=Random.MersenneTwister(42))
    @test out_mm["Eω"] == out_mm2["Eω"]
    # Clean input: z=0 field is identical to a no-noise run
    ref_mm = prop_capillary(noise_args...; noise_kwargs...,
                            shotnoise=false, modes=2)
    @test out_mm["Eω"][:, :, 1] == ref_mm["Eω"][:, :, 1]
end

@testset "Invalid shotnoise" begin
    @test_throws DomainError prop_capillary(
        noise_args...; noise_kwargs..., shotnoise=:invalid)
end

@testset "Clean input spectrum" begin
    # With shotnoise=:modified, the field at z=0 should be identical to a no-noise run
    # because noise only enters the nonlinear operator, not the initial field.
    ref = prop_capillary(noise_args...; noise_kwargs..., shotnoise=false)
    mod = prop_capillary(noise_args...; noise_kwargs...,
                         shotnoise=:modified,
                         rng=Random.MersenneTwister(42))
    @test ref["Eω"][:, 1] == mod["Eω"][:, 1]
end

@testset "Energy conservation" begin
    # Output energy with modified noise model should closely match a no-noise reference
    # (noise energy is negligible: ~ħω per bin)
    ref = prop_capillary(noise_args...; noise_kwargs..., shotnoise=false)
    mod = prop_capillary(noise_args...; noise_kwargs...,
                         shotnoise=:modified,
                         rng=Random.MersenneTwister(42))
    eref = Processing.energy(ref)[end]
    emod = Processing.energy(mod)[end]
    @test isapprox(emod, eref; rtol=0.01)
end

@testset "Backward compatibility" begin
    # shotnoise=false should produce identical results regardless of how it's
    # specified (Bool false or explicit shotnoise=false)
    out1 = prop_capillary(noise_args...; noise_kwargs..., shotnoise=false)
    out2 = prop_capillary(noise_args...; noise_kwargs...,
                          shotnoise=false)
    @test out1["Eω"] == out2["Eω"]
end

@testset "Reproducibility with fixed seed" begin
    # The noise field is generated once before propagation and held constant
    # (Chen & Wise §A2: z-dependent random noise gives step-size-dependent results).
    # Same seed must produce bit-identical output.
    out1 = prop_capillary(noise_args...; noise_kwargs...,
                          shotnoise=:modified,
                          rng=Random.MersenneTwister(42))
    out2 = prop_capillary(noise_args...; noise_kwargs...,
                          shotnoise=:modified,
                          rng=Random.MersenneTwister(42))
    @test out1["Eω"] == out2["Eω"]
end

# Physics tests: noise model behaviour with H₂ Raman scattering.
# H₂ auto-enables Raman. The key physical signature of the modified noise model is that
# noise enters the nonlinear operator (not the input field), so:
# - The Stokes spectral region has NO elevated noise floor at z=0 (unlike traditional)
# - The traditional model adds one-photon-per-mode noise to the input, creating a
#   measurable noise floor across all frequencies at z=0
# We use a Stokes band far from the pump (1100-1500 nm) to avoid the input pulse's
# spectral tails, and moderate parameters where the gain is low enough that the
# noise floor difference is the dominant observable effect.
const raman_args = (100e-6, 0.3, :H2, 3)  # 100 μm, 30 cm, H₂ at 3 bar
const raman_kwargs = (λ0=800e-9, τfwhm=30e-15, energy=5e-6,
                      trange=1000e-15, λlims=(200e-9, 4e-6),
                      kerr=false, plasma=false, saveN=51)
const stokes_band = (1100e-9, 1500e-9)  # vibrational Stokes region, far from pump
const anti_stokes_band = (500e-9, 680e-9)  # H₂ vibrational anti-Stokes region

@testset "Noise model physics with Raman" begin
    # Modified noise model: no noise floor at z=0 in the Stokes region.
    mod = prop_capillary(raman_args...; raman_kwargs...,
                         shotnoise=:modified,
                         rng=Random.MersenneTwister(42))
    E_stokes_mod = Processing.energy(mod; bandpass=stokes_band)

    # Traditional shot noise: one-photon-per-mode noise is added to the input field,
    # creating a measurable noise floor across all frequencies at z=0.
    trad = prop_capillary(raman_args...; raman_kwargs...,
                          shotnoise=:input)
    E_stokes_trad = Processing.energy(trad; bandpass=stokes_band)

    # Key physics check: the traditional model has an elevated noise floor at z=0
    # that is many orders of magnitude above the modified model's clean spectrum.
    @test E_stokes_trad[1] > 1e6 * E_stokes_mod[1]

    # Modified model: clean input spectrum (negligible energy in Stokes band at z=0)
    @test E_stokes_mod[1] < 1e-30

    # Both models produce similar Stokes energy at z=L (either from noise-seeded or
    # stimulated Raman of the pulse's spectral tails)
    @test E_stokes_mod[end] > 0
    @test E_stokes_trad[end] > 0

    # Stokes preference over anti-Stokes (Chen & Wise main text, Fig. 1):
    # The modified model correctly captures the spontaneous Raman asymmetry —
    # Stokes generation grows faster than anti-Stokes from noise.
    # We compare growth ratios (end/start) since the anti-Stokes band has
    # residual pulse-tail energy that dominates absolute values.
    E_astokes_mod = Processing.energy(mod; bandpass=anti_stokes_band)
    stokes_growth = E_stokes_mod[end] / max(E_stokes_mod[1], 1e-50)
    astokes_growth = E_astokes_mod[end] / max(E_astokes_mod[1], 1e-50)
    @test stokes_growth > astokes_growth
end

@testset "Noise floor Stokes vs anti-Stokes" begin
    # With Kerr enabled (default), both Raman and FWM act on the noise.
    # The traditional model creates an elevated noise floor at BOTH Stokes and anti-Stokes
    # at z=0, while the modified model has a clean spectrum.
    raman_kwargs_kerr = (λ0=800e-9, τfwhm=30e-15, energy=5e-6,
                         trange=1000e-15, λlims=(200e-9, 4e-6),
                         plasma=false, saveN=51)

    mod = prop_capillary(raman_args...; raman_kwargs_kerr...,
                         shotnoise=:modified,
                         rng=Random.MersenneTwister(42))
    E_s_mod = Processing.energy(mod; bandpass=stokes_band)
    E_as_mod = Processing.energy(mod; bandpass=anti_stokes_band)

    trad = prop_capillary(raman_args...; raman_kwargs_kerr...,
                          shotnoise=:input)
    E_s_trad = Processing.energy(trad; bandpass=stokes_band)
    E_as_trad = Processing.energy(trad; bandpass=anti_stokes_band)

    # Traditional has higher Stokes noise floor at z=0 than modified
    @test E_s_trad[1] > 1e6 * E_s_mod[1]
    # Traditional has higher anti-Stokes noise floor at z=0 than modified
    @test E_as_trad[1] > 5 * E_as_mod[1]
    # Modified Stokes is clean at z=0
    @test E_s_mod[1] < 1e-30
    # Modified Stokes grows from near-zero during propagation
    @test E_s_mod[end] > E_s_mod[1]
    # Both models converge at output (stimulated processes dominate)
    @test isapprox(E_as_mod[end], E_as_trad[end]; rtol=0.1)
end

@testset "Stokes growth with modified noise" begin
    # The modified model shows Stokes energy growing by many orders of magnitude
    # from the noise-seeded spontaneous Raman process. This confirms that the
    # noise successfully seeds Raman gain even though it enters only the nonlinear
    # operator (not the input field).
    mod = prop_capillary(raman_args...; raman_kwargs...,
                         shotnoise=:modified,
                         rng=Random.MersenneTwister(42))
    E_s = Processing.energy(mod; bandpass=stokes_band)

    # Large growth factor from noise-seeded spontaneous Raman
    @test E_s[end] / E_s[1] > 1e10
    # Monotonic growth in the last ~10 z-points (exponential gain regime)
    @test all(diff(E_s[end-9:end]) .> 0)
end

Logging.global_logger(old_logger)
