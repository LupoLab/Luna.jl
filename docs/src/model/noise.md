# Noise model

## Background

Noise-seeded processes are central to many phenomena in nonlinear fibre optics: spontaneous
Raman scattering seeds Raman combs; noise-seeded four-wave
mixing (FWM) drives modulational instability (MI); and the interplay of both produces the
shot-to-shot fluctuations characteristic of supercontinuum generation. An accurate noise
model is therefore essential for any simulation where noise-initiated nonlinear dynamics play
a role.

Luna.jl supports two noise models, selected via the `shotnoise` keyword argument in
[`prop_capillary`](@ref Interface.prop_capillary) and [`prop_gnlse`](@ref Interface.prop_gnlse):

- **`:input`** --- the traditional shot-noise approach.
- **`:modified`** or **`true`** (default) --- the modified shot-noise approach of Chen & Wise [1].

## Traditional shot noise (`:input`)

The traditional approach adds one-photon-per-mode quantum noise directly to the input field
at ``z = 0``:

```math
\tilde{E}(z=0, \omega) = \tilde{E}_{\mathrm{pulse}}(\omega) + \tilde{E}_{\mathrm{noise}}(\omega)
```

where ``|\tilde{E}_{\mathrm{noise}}(\omega)| = \sqrt{\hbar\omega\,\Delta\nu}`` with uniformly
random spectral phase and ``\Delta\nu`` is the frequency bin spacing. The propagation equation
then contains only stimulated (deterministic) terms, and all noise-seeded processes arise from
the stimulated evolution of the injected noise.

While simple, this approach has several well-known limitations:

1. **Artificial FWM phase-matching**: The noise field, superimposed on the input pulse,
   acquires a coherent phase increment from dispersion during propagation. This means the
   noise can satisfy phase-matching conditions for FWM, which is unphysical --- real vacuum
   fluctuations are incoherent and cannot maintain phase-matching over extended distances.

2. **Elevated noise floor**: The noise creates a persistent increase of the spectral
   intensity at all frequencies that does not decay.

3. **No Stokes/anti-Stokes asymmetry**: Spontaneous Raman generation exhibits a fundamental
   spectral asymmetry (Stokes is favoured by an extra quantum of vacuum fluctuation). The
   traditional model does not capture this.

## Modified shot noise (`:modified`)

The modified shot-noise model [1] resolves all of the above issues by including the noise
field in the **nonlinear operator** rather than in the input field:

```math
\partial_z A = \hat{\mathcal{D}}\,A + \hat{\mathcal{N}}(A + A_{\mathrm{noise}})
```

The key structural features are:

- **Dispersion acts only on the physical field**: ``\hat{\mathcal{D}}\,A``, not
  ``\hat{\mathcal{D}}(A + A_{\mathrm{noise}})``. This prevents the noise from acquiring
  coherent phase increments that would cause artificial FWM phase-matching.

- **The noise field is constant**: ``A_{\mathrm{noise}}(\omega)`` is generated once before
  propagation with one-photon-per-mode spectral density and random phase, and held fixed
  throughout. It does not evolve, grow, or accumulate.

- **All nonlinear processes are correctly seeded**: Both Kerr (FWM/MI) and Raman
  (spontaneous Stokes/anti-Stokes) processes are seeded through the same mechanism ---
  the noise enters the nonlinear operator alongside the field.

- **No elevated noise floor**: Because the noise is not added to the propagating field,
  the spectrum at ``z = 0`` contains only the input pulse. Noise-seeded spectral components
  grow continuously from zero as the pulse propagates.

This approach is a **general-purpose noise model** applicable to all ``\chi^{(3)}``
propagation problems, including MI-based supercontinuum generation, Raman comb formation,
soliton dynamics with self-frequency shift, and combined Kerr--Raman broadening.

## Usage

The modified shot-noise model is used by default (`shotnoise=true`, equivalent to
`shotnoise=:modified`). To use the traditional model instead, pass `shotnoise=:input`:

```julia
# Default: modified shot-noise model (shotnoise=true)
output = prop_capillary(radius, flength, gas, pressure;
    λ0=800e-9, λlims=(200e-9, 4000e-9), trange=400e-15,
    τfwhm=30e-15, energy=1e-6,
)
```

For ensemble simulations (shot-to-shot statistics), use different random seeds for each shot:

```julia
using Random
for shot in 1:100
    rng = MersenneTwister(shot)
    output = prop_capillary(radius, flength, gas, pressure;
        λ0=800e-9, λlims=(200e-9, 4000e-9), trange=400e-15,
        τfwhm=30e-15, energy=1e-6,
        shotnoise=:modified, rng=rng
    )
    # ... process output ...
end
```

Setting `shotnoise=false` disables noise entirely. This prevents generation of the noise
field when using the modified model, and prevents adding shot noise to the input spectrum
when using the traditional model.

## Ionization and plasma

In the modified shot-noise model, the combined field ``A + A_{\mathrm{noise}}`` is passed to
**all** nonlinear response functions, including Kerr, Raman, and plasma/ionization. This is
physically reasonable because the noise amplitude is of order
``\sqrt{\hbar\omega\,\Delta\nu} \approx 5 \times 10^{-4}\;\sqrt{\mathrm{W}}`` per
frequency mode — roughly ``10^{-14}`` of a typical pulse peak power. The tunnelling
ionization rate depends on the electric field through a highly nonlinear (exponential)
threshold, so the negligible noise field has no measurable effect on the plasma response.
Only the Kerr and Raman processes, which are linear in the perturbation field, are
meaningfully affected by the noise injection.

## Power diagnostics at low signal levels

Because the noise is not added to the propagating field, spontaneously generated spectral
components start from exactly zero power at ``z = 0``. At very small ``z``, the power
``|A_{s'}|^2 \propto z^2`` (quadratic growth), which is the expected linearised result.
To recover the correct physical power including the vacuum fluctuation contribution, one
can temporarily add the noise before computing the power:

```math
P_{s'}(z) = |A_{s'}(z) + A_{\mathrm{noise}}|^2 - P_{\mathrm{noise}}
```

This correction is only necessary when the generated power is comparable to or below the
noise floor. At higher powers (the stimulated/saturated regime), ``|A_{s'}|^2`` directly
gives the correct result.

## Comparison

| Feature | Traditional (`:input`) | Modified (`:modified`) |
|---------|----------------------|----------------------|
| Artificial FWM phase-matching | Yes | No |
| Elevated noise floor | Yes | No |
| Stokes/anti-Stokes asymmetry | Not captured | Effectively captured |
| Captures all cascaded processes | Only through stimulated evolution | Automatically |
| Step-size sensitivity | No | No |

## References

1. Y.-H. Chen and F. W. Wise, "A simple accurate way to model noise-seeded ultrafast
   nonlinear processes", [arXiv:2410.20567](https://arxiv.org/abs/2410.20567) (2024).

2. Y.-H. Chen and F. W. Wise, "Unified theory for Raman scattering in gas-filled
   hollow-core fibers", [APL Photonics **9**, 030902](https://doi.org/10.1063/5.0189749)
   (2024).
