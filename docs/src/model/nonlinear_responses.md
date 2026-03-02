# Nonlinear responses

## Kerr effect
The instantaneous electronic Kerr effect arises from the real part of the third-order susceptibility $\chi^{(3)}$.

In the envelope picture, the Kerr nonlinear polarisation is:

$$P_\mathrm{Kerr}(t) = \frac{3}{4}\varepsilon_0 \gamma_3 \rho \lvert E \rvert^2 E$$

where $\gamma_3$ is the single-molecule third-order hyperpolarisability (as returned by `PhysData.γ3_gas`), $\rho$ is the number density (so that $\rho \gamma_3 = \chi^{(3)}$), and $E$ is the field envelope. This response is frequency-independent and operates in the time domain.

## Photoionisation & plasma


## Raman response


## Two-photon absorption (TPA)

Two-photon absorption is the **imaginary part** of $\chi^{(3)}$, the absorptive counterpart to the Kerr effect (which is the real part). While the Kerr response is frequency-independent and can be evaluated in the time domain, $\mathrm{Im}(\chi^{(3)})(\omega)$ varies strongly across deep-UV bandwidths — by a factor of ~20 across 200–280 nm in fused silica — making a time-domain scalar response inadequate.

### Physical model

TPA is characterised by the two-photon absorption coefficient $\beta_2(\omega)$ (units: m/W), which is energetically allowed when the two-photon energy exceeds the material bandgap:

$$\beta_2(\omega) = C \times (2\hbar\omega/\mathrm{eV} - E_g)^\alpha \times \Theta(2\hbar\omega - E_g)$$

where $E_g$ is the bandgap energy (eV), $C$ (m/W/eV$^\alpha$) and $\alpha$ are empirical fitting parameters, and $\Theta$ is the Heaviside step function.

### Implementation as a spectral response

TPA is implemented as a [`SpectralResponse`](@ref) — a frequency-domain nonlinear response that operates on $\mathcal{F}\{|E|^2 E\}(\omega)$. This is in contrast to time-domain responses (Kerr, Raman, Plasma) which operate on $E(t)$.

The TPA contribution to the nonlinear polarisation is:

$$P_\mathrm{TPA}(\omega) = \mathrm{coeff}(\omega) \times \rho \times \mathcal{F}\{|E|^2 E\}(\omega)$$

where the coefficient is derived by tracing the standard relationship $\beta_2 = 3\omega \, \mathrm{Im}(\chi^{(3)}) / (2 n_0^2 c^2)$ through Luna's propagation pipeline. In `TransModeAvg`, the field is normalised by $\mathrm{nlscale} = \sqrt{\varepsilon_0 c/2}$ before computing $|E|^2 E$, and the norm function applies $-i\omega^2 / (4 \cdot \mathrm{nlscale} \cdot c \cdot \beta)$. The resulting $\mathrm{nlscale}^4 = (\varepsilon_0 c/2)^2$ factor in the normalisation chain produces the $\varepsilon_0^2 c^2$ prefactor:

$$\mathrm{coeff}(\omega) = \frac{-i \, \varepsilon_0^2 \, c^2 \, \beta_2(\omega)}{2\omega}$$

This coefficient enters the **same P_NL buffer** as the Kerr response, so the `Trans*` types' normalisation functions correctly convert both Kerr and TPA contributions to $\partial A/\partial z$. The $-i$ factor ensures TPA causes loss (negative real $\partial A / \partial z$) after the normalisation's own $-i$ factor.

### Geometry-agnostic design

`TPAResponse` is **geometry-agnostic**: it operates on 1D frequency vectors with callable signature `(out_ω, F_E2E_ω, ρ)`, regardless of whether the simulation uses mode-averaged (`TransModeAvg`), modal (`TransModal`), or 3D free-space (`TransFree`) propagation. All spatial integration, effective area, and field normalisation is handled by the `Trans*` types in `NonlinearRHS`, exactly as for time-domain responses.

### Degenerate approximation

The current implementation uses the **degenerate** TPA approximation, where all interacting photons have the same frequency. This still accounts for the overall frequency dependence of the TPA response (e.g. two photons at 200 nm, or two photons at 300 nm), but does not account for the frequency difference between photons in non-degenerate TPA interactions (e.g. one photon at 200 nm and another at 300 nm). This is appropriate for narrowband or moderate-bandwidth pulses. Non-degenerate TPA (photons at different frequencies) is not currently implemented.

### Notes
- Only valid for **envelope** propagation (`EnvGrid`). Using with `RealGrid` (carrier-resolved) is not physically meaningful.
- Handles **scalar** (single-polarisation) fields. Vector TPA ($(|E_x|^2 + |E_y|^2)E$) is a future extension.
- The coefficient is derived for Luna's `TransModeAvg` pipeline. It works correctly across all `Trans*` types because each type's normalisation function compensates for its own field convention — the same mechanism that makes the Kerr response geometry-agnostic.

### Usage

```julia
using Luna

# From material parameters (PhysData)
β₂_ω = PhysData.β₂_TPA.(grid.ω, :SiO2)
tpa = Nonlinear.TPAResponse(grid.ω, β₂_ω)

# Include in responses tuple alongside Kerr
responses = (Nonlinear.Kerr_env(χ3), tpa)

# Or via prop_gnlse interface
prop_gnlse(γ, flength, βs; ..., tpa=β₂_ω)
```

See also: [`PhysData.β₂_TPA`](@ref), [`Nonlinear.TPAResponse`](@ref), [`Nonlinear.SpectralResponse`](@ref)
