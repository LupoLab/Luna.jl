using Luna
import Test: @test, @testset, @test_throws
import Luna: Fields, Grid, Processing, LinearOps, NonlinearRHS, Capillary, Interface
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

@testset "Unitary phase and noise propagation" begin
    grid = Grid.RealGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
    mode = Capillary.MarcatiliMode(100e-6, :He, 1.0; loss=true)
    linop, _, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
    # the mode really is lossy, or the "loss is not applied" test below is vacuous
    @test minimum(real, linop) < 0

    phase = LinearOps.unitary_phase(linop, grid)
    @test phase isa LinearOps.ConstUnitaryPhase
    φ = zeros(Float64, length(grid.ω))
    phase(φ, 0.037)
    @test φ == imag.(linop) .* 0.037

    nf = Fields.generate_noise_field(grid; rng=Random.MersenneTwister(42))
    noise = NonlinearRHS.ModifiedNoise(nf, phase, grid)
    # at z = 0 the noise is exactly the draw, so the "clean input spectrum" property holds
    @test NonlinearRHS.noise_at!(noise, 0.0) == noise.Eω0
    # loss is NOT applied: the amplitude is invariant, only the phase advances
    for z in (0.01, 0.05, 0.1)
        @test abs.(NonlinearRHS.noise_at!(noise, z)) ≈ abs.(noise.Eω0)
    end
    # dispersion IS applied
    @test !isapprox(NonlinearRHS.noise_at!(noise, 0.1), noise.Eω0)
end

@testset "Tabulated unitary phase" begin
    # z-dependent linear operators: the accumulated phase is tabulated by adaptive
    # bisection, and must agree with a direct quadrature of imag(linop) to the tolerance the
    # refinement was asked for. Luna's own profiles are the awkward cases -- a p0 = 0
    # gradient has a 1/√z cusp in dp/dz at the entrance and a multi-section fill has a
    # derivative discontinuity at every junction -- so all of them are exercised here.
    grid = Grid.RealGrid(0.2, 800e-9, (400e-9, 2e-6), 200e-15)
    nω = length(grid.ω)

    function refphase(linop!, z; n=2001) # composite Simpson, n odd
        zs = range(0, z, length=n)
        buf = Array{ComplexF64}(undef, nω)
        acc = zeros(Float64, nω)
        for (i, zz) in enumerate(zs)
            w = (i == 1 || i == n) ? 1.0 : (iseven(i) ? 4.0 : 2.0)
            linop!(buf, zz)
            @. acc += w*imag(buf)
        end
        acc .* (step(zs)/3)
    end

    linops = Dict(
        # smooth gradient
        "gradient" => Capillary.gradient(:He, 0.2, 5.0, 0.5)[1],
        # p0 = 0: dp/dz diverges as 1/√z at the entrance
        "cusp" => Capillary.gradient(:He, 0.2, 0.0, 5.0)[1],
        # multi-section fill: a derivative discontinuity at each junction
        "kinks" => Capillary.gradient(:He, [0.0, 0.07, 0.13, 0.2], [0.0, 6.0, 1.0, 3.0])[1],
    )
    φ = zeros(Float64, nω)
    for (name, coren) in linops
        mode = Capillary.MarcatiliMode(100e-6, coren; loss=true)
        linop!, _ = LinearOps.make_linop(grid, mode, grid.referenceλ)
        phase = LinearOps.unitary_phase(linop!, grid, (nω,))
        @test phase isa LinearOps.TabulatedUnitaryPhase
        # the refinement met the tolerance it was asked for...
        @test phase.err <= phase.atol
        # ...and the measured error is a good estimate of the true one, which is what
        # justifies reporting it. 10x headroom covers the midpoint estimate being slightly
        # optimistic and the reference quadrature's own error at the cusp/kinks.
        for z in (0.05, 0.13, 0.2)
            phase(φ, z)
            @test maximum(abs, φ .- refphase(linop!, z)) < 10*phase.atol
        end
    end

    # a kinked taper, with a static fill. Unlike a gas gradient, whose pressure profile is
    # clamped beyond the fibre, a taper keeps changing past zmax -- so this is the case that
    # needs the table to cover the integrator's overshoot with real nodes.
    afun(z) = z < 0.1 ? 100e-6 : 100e-6 - 40e-6*(z - 0.1)/0.1
    linop!, _ = LinearOps.make_linop(
        grid, Capillary.MarcatiliMode(afun, :He, 5.0; loss=true), grid.referenceλ)
    phase = LinearOps.unitary_phase(linop!, grid, (nω,))
    @test phase.err <= phase.atol
    @test phase.z[end] > grid.zmax # the overshoot margin is tabulated, not extrapolated
    for z in (0.13, 0.2, 0.205)
        phase(φ, z)
        @test maximum(abs, φ .- refphase(linop!, z)) < 10*phase.atol
    end
    # past the margin it falls back to linear extrapolation rather than failing
    phase(φ, 0.5)
    @test all(isfinite, φ)
    # a tolerance the caller asks for is honoured
    coarse = LinearOps.unitary_phase(linop!, grid, (nω,); atol=1e-3)
    @test coarse.err <= 1e-3
    @test length(coarse.z) < length(phase.z) # ... with fewer nodes
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

    # Both models generate Stokes light. They are NOT expected to agree at z = L in this
    # low-gain configuration: the traditional model's Stokes band is still dominated by its
    # own residual input noise floor rather than by generated Stokes, so the two legitimately
    # differ by orders of magnitude here. The model comparison that does mean something is
    # in the "Model agreement in the Raman gain regime" testset below.
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

@testset "Model agreement in the Raman gain regime" begin
    # Raman gain is phase-insensitive, so the two noise models must give the same Stokes
    # energy at z = L once real gain (rather than the residual input noise floor) sets the
    # Stokes level. This is the assertion the comment in "Noise model physics with Raman"
    # promised but never made; it needs a gain-dominated configuration to mean anything.
    #
    # It is also the control for the noise-propagation fix: advancing the noise field under
    # the unitary linear operator changes phase-insensitive results not at all (measured
    # ratio 1.0000011 with propagation, 0.9999916 without), while changing Kerr/FWM results
    # by one to two orders of magnitude — see "Chen & Wise identity: Kerr soliton".
    gain_args = (100e-6, 0.15, :H2, 20)  # 100 μm, 15 cm, H₂ at 20 bar
    gain_kwargs = (λ0=800e-9, τfwhm=30e-15, energy=40e-6, trange=1000e-15,
                   λlims=(200e-9, 4e-6), kerr=false, plasma=false, saveN=11)
    mod = prop_capillary(gain_args...; gain_kwargs..., shotnoise=:modified,
                         rng=Random.MersenneTwister(42))
    trad = prop_capillary(gain_args...; gain_kwargs..., shotnoise=:input,
                          rng=Random.MersenneTwister(42))
    E_mod = Processing.energy(mod; bandpass=stokes_band)
    E_trad = Processing.energy(trad; bandpass=stokes_band)
    # modified starts clean, traditional starts on its own noise floor
    @test E_mod[1] < 1e-30
    @test E_trad[1] > 1e-20
    # ... and they land on the same Stokes energy
    @test isapprox(E_mod[end], E_trad[end]; rtol=1e-4)
end

@testset "Chen & Wise identity: Kerr soliton" begin
    # `Fields.ShotNoise` and `Fields.generate_noise_field` use the same amplitude formula and
    # the same `rand` call, so a shared seed gives the two models an identical noise draw.
    # The modified model then rests on the identity A_s = A_s' + A_noise: with the background
    # restored, it must reproduce the traditional model field for field. That holds only if
    # the noise field is advanced along z — with a static noise field this fails by ~50x on
    # an N = 6 soliton over one period, which is a conservative (no net gain) problem and so
    # the most sensitive kind. No ensemble is needed: this is deterministic.
    γ = 0.1
    β2 = -1e-26
    τ0 = 280e-15
    N = 6.0
    P0 = N^2*abs(β2)/(γ*τ0^2) # fr = 0, so χ3 is the full Kerr nonlinearity
    L = π*τ0^2/(2*abs(β2))    # one soliton period
    kw = (λ0=835e-9, τfwhm=(2*log(1 + sqrt(2)))*τ0, power=P0, pulseshape=:sech,
          λlims=(450e-9, 8e-6), trange=4e-12, raman=false, fr=0.0, saveN=2)
    function propagate(shotnoise)
        Eω, grid, linop, transform, FT, output = Interface.prop_gnlse_args(
            γ, L, [0.0, 0.0, β2]; kw..., shotnoise, rng=Random.MersenneTwister(42))
        Luna.run(Eω, grid, linop, transform, FT, output;
                 rtol=1e-8, atol=1e-16, status_period=60)
        output, transform
    end
    ref, _ = propagate(false)
    trad, _ = propagate(:input)
    mod, tr = propagate(:modified)

    E0 = ref["Eω"][:, end]
    # noise-seeded content, as the difference from the noiseless run
    seeded(E) = maximum(abs.(E .- E0))/maximum(abs.(E0))
    An = NonlinearRHS.noise_at!(tr.noise, L) # the noise background at z = L
    d_trad = seeded(trad["Eω"][:, end])
    d_mod = seeded(mod["Eω"][:, end] .+ An)
    @test d_trad > 1e-3 # the noise is doing something, i.e. the test has teeth
    @test 0.5 < d_mod/d_trad < 2.0 # ~50 with a static noise field
end

@testset "Processing.withnoise" begin
    out = prop_capillary(noise_args...; noise_kwargs..., shotnoise=:modified,
                         save_noise=true, rng=Random.MersenneTwister(42))
    @test haskey(out, "noise_field")
    @test haskey(out, "noise_phase")
    E = Processing.withnoise(out)
    @test size(E) == size(out["Eω"])
    # at z = 0 the accumulated phase is zero, so the background is exactly the draw
    @test E[:, 1] ≈ out["Eω"][:, 1] .+ out["noise_field"]
    # restoring the background adds exactly the vacuum level, whose amplitude does not
    # change with z (loss is not applied to the noise)
    @test maximum(abs, E[:, end] .- out["Eω"][:, end]) ≈ maximum(abs, out["noise_field"])

    # the z = 0 draw is always saved, but reconstruction from file needs save_noise=true
    Eω, grid, linop, transform, FT, out2 = Interface.prop_capillary_args(
        noise_args...; noise_kwargs..., shotnoise=:modified,
        rng=Random.MersenneTwister(42))
    Luna.run(Eω, grid, linop, transform, FT, out2; status_period=60)
    @test haskey(out2, "noise_field")
    @test !haskey(out2, "noise_phase")
    @test_throws ErrorException Processing.withnoise(out2)
    # ... or the transform, which always works
    @test Processing.withnoise(out2, transform) ≈ E
end

@testset "Cached resume adopts the recorded noise" begin
    # The modified model uses the noise at every step, so resuming a cached run with a fresh
    # draw would splice two realisations together and leave the already-saved planes
    # inconsistent with the recorded noise field.
    fpath = joinpath(mktempdir(), "noise_resume.h5")
    out = prop_capillary(noise_args...; noise_kwargs..., shotnoise=:modified,
                         rng=Random.MersenneTwister(42), filepath=fpath)
    saved = out["noise_field"]
    @test any(!iszero, saved)
    # setting up again on the same file finds the cache, and must adopt the recorded draw
    # rather than the different one its own rng would produce
    _, _, _, transform, _, out2 = Interface.prop_capillary_args(
        noise_args...; noise_kwargs..., shotnoise=:modified,
        rng=Random.MersenneTwister(7), filepath=fpath)
    @test transform.noise.Eω0 == saved
    @test out2["noise_field"] == saved
end

@testset "rtol/atol are threaded to the integrator" begin
    # prop_capillary and prop_gnlse previously offered no way to set integrator tolerances
    kwargs = (noise_kwargs..., shotnoise=false)
    loose = prop_capillary(noise_args...; kwargs..., rtol=1e-4)
    tight = prop_capillary(noise_args...; kwargs..., rtol=1e-10, atol=1e-16)
    @test loose["Eω"] != tight["Eω"]
    @test isapprox(loose["Eω"][:, end], tight["Eω"][:, end];
                   atol=1e-3*maximum(abs, tight["Eω"][:, end]))
end

Logging.global_logger(old_logger)
