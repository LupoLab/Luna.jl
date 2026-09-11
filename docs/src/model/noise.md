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

1. **Elevated noise floor**: The noise creates a persistent increase of the spectral
   intensity at all frequencies that does not decay. In particular, the spectrum at ``z = 0``
   already contains a noise floor everywhere, so there is no clean baseline against which to
   measure spontaneously generated light.

2. **No Stokes/anti-Stokes asymmetry**: Spontaneous Raman generation exhibits a fundamental
   spectral asymmetry (Stokes is favoured by an extra quantum of vacuum fluctuation). The
   traditional model does not capture this.

3. **No spontaneous generation in absorptive media**: the input noise is attenuated along with
   the field, whereas a real vacuum field is replenished by the same process that causes the
   loss.

## Modified shot noise (`:modified`)

The modified shot-noise model [1] resolves all of the above issues by including the noise
field in the **nonlinear operator** rather than in the input field:

```math
\partial_z A = \hat{\mathcal{D}}\,A + \hat{\mathcal{N}}(A + A_{\mathrm{noise}})
```

The key structural features are:

- **The noise is not part of the propagating field**: the linear operator
  ``\hat{\mathcal{D}}`` acts on ``A`` alone, so the spectrum that is saved and plotted never
  contains a noise floor, and the noise is never amplified or attenuated by the propagation
  itself. It is a source, not a seed.

- **The noise field is drawn once**: ``A_{\mathrm{noise}}`` is generated before propagation
  with one-photon-per-mode spectral density and random spectral phase, and is never
  re-randomised. This is what keeps the result step-size independent (a noise field redrawn
  along ``z`` would make the answer depend on the step count).

- **It is advanced by the unitary part of the linear operator** --- see the next section.

- **All nonlinear processes are correctly seeded**: Both Kerr (FWM/MI) and Raman
  (spontaneous Stokes/anti-Stokes) processes are seeded through the same mechanism ---
  the noise enters the nonlinear operator alongside the field.

- **No elevated noise floor**: Because the noise is not added to the propagating field,
  the spectrum at ``z = 0`` contains only the input pulse. Noise-seeded spectral components
  grow continuously from zero as the pulse propagates.

This approach is a **general-purpose noise model** applicable to all ``\chi^{(3)}``
propagation problems, including MI-based supercontinuum generation, Raman comb formation,
soliton dynamics with self-frequency shift, and combined Kerr--Raman broadening.

## How the noise field evolves

``A_{\mathrm{noise}}`` is drawn once and then advanced under the **unitary** part of the
linear operator only:

```math
A_{\mathrm{noise}}(\omega, z) = A_{\mathrm{noise}}(\omega, 0)\,
    \exp\!\left[i\!\int_0^z \mathrm{Im}\,\hat{\mathcal{D}}(\omega, z')\,
    \mathrm{d}z'\right]
```

| part of the linear operator | acts on the noise? | why |
|---|---|---|
| dispersion | **yes** | It is unitary, and it is the phase evolution of the medium's own vacuum modes. |
| loss | **no** | Fluctuation--dissipation: loss replenishes the vacuum rather than removing it, so the local noise level stays at one photon per mode. This is what allows the model to seed spontaneous generation in an absorbing medium, saturating at ``P_{\mathrm{noise}}``. |
| gain | **no** | The source term already injects fresh vacuum at every ``z``, which the homogeneous propagator then amplifies over the remaining length --- that integral *is* the amplified spontaneous emission. Amplifying the noise as well would double-count it. |

Note that *constant* here means **not re-randomised**, which is what the step-size-independence
argument of Ref. [1] actually requires --- not un-propagated. Holding the noise field literally
fixed is exact only in the dispersion-free gain model in which Ref. [1] derives its identity
``A_s = A_s' + A_{\mathrm{noise}}``. In a dispersive medium the traditional model's input noise
*does* disperse, so a static ``A_{\mathrm{noise}}`` breaks that identity in proportion to how
much dispersion has acted. The consequence is a magnitude error in **phase-sensitive**
processes (Kerr MI and FWM, where the ``\pm\Omega`` sideband pair must hold a fixed phase
relation to the pump, and that relation is exactly what dispersion controls) of up to a factor
of a few per frequency, and much more in propagation with no net gain, where an injected source
has nothing to dilute it. Applying the *full* linearised operator instead is worse than doing
nothing, because the source term already supplies the nonlinear part.

Phase-insensitive processes are unaffected either way: Raman gain does not depend on the phase
of the Stokes wave relative to the pump, so denying the noise its phase evolution rotates
phases that never mattered. Pure-Raman results are identical to five significant figures with
or without this term.

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

The noise realisation used by the modified model is written to the output as `"noise_field"`
(the ``z = 0`` draw). Passing `save_noise=true` additionally stores the accumulated phase at
every output plane, which is what [`Processing.withnoise`](@ref Luna.Processing.withnoise)
needs to reconstruct ``A_{\mathrm{noise}}(z)`` from a saved file.

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
To recover the correct physical power including the vacuum fluctuation contribution, add the
noise back before computing the power:

```math
P_{s'}(z) = |A_{s'}(z) + A_{\mathrm{noise}}(z)|^2 - P_{\mathrm{noise}}
```

[`Processing.withnoise`](@ref Luna.Processing.withnoise) performs the restoration:

```julia
out = prop_capillary(a, L, gas, P; ..., shotnoise=:modified, save_noise=true)
grid = Processing.makegrid(out)
E = Processing.energy(grid, Processing.withnoise(out); bandpass=(1100e-9, 1500e-9))
```

This correction is only necessary when the generated power is comparable to or below the
noise floor. At higher powers (the stimulated/saturated regime), ``|A_{s'}|^2`` directly
gives the correct result. Note also that the ``-P_{\mathrm{noise}}`` term is there because
Ref. [1] wants the *generated* power; if the quantity being compared against already includes
the seed --- as it does when comparing with a `shotnoise=:input` run --- subtracting it again
double-counts.

## Comparison

| Feature | Traditional (`:input`) | Modified (`:modified`) |
|---------|----------------------|----------------------|
| Elevated noise floor | Yes | No |
| Clean spectrum at ``z = 0`` | No | Yes |
| Stokes/anti-Stokes asymmetry | Not captured | Effectively captured |
| Spontaneous generation in absorbing media | Not captured | Captured |
| Captures all cascaded processes | Only through stimulated evolution | Automatically |
| Step-size sensitivity | No | No |
| Noise background included in the saved spectrum | Yes | No --- see [`Processing.withnoise`](@ref Luna.Processing.withnoise) |

## References

1. Y.-H. Chen and F. W. Wise, "A simple accurate way to model noise-seeded ultrafast
   nonlinear processes", [arXiv:2410.20567](https://arxiv.org/abs/2410.20567) (2024).

2. Y.-H. Chen and F. W. Wise, "Unified theory for Raman scattering in gas-filled
   hollow-core fibers", [APL Photonics **9**, 030902](https://doi.org/10.1063/5.0189749)
   (2024).
