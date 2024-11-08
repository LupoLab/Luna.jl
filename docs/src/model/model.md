# The numerical model
```@contents
Pages = ["model.md", "modal_decompositions.md", "nonlinear_responses.md"]
Depth = 2
```

`Luna` can solve several different variations of the unidirectional pulse propagation equation (UPPE). All of these variations have this basic form in common:
```math
\partial_z E(\omega, \mathbf{k}_\perp, z){\mathrm{d}z} = \mathcal{L}(\omega, \mathbf{k}_\perp, z)E(\omega, \mathbf{k}_\perp, z) + \frac{i\omega}{N_{\mathrm{nl}}} P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)\,,
```
where ``E(\omega, \mathbf{k}_\perp, z)`` is the electric field in "reciprocal space", i.e. frequency and transverse spatial frequency, ``\omega`` is angular frequency, ``\mathbf{k}_\perp`` is some generalised transverse spatial frequency, ``z`` is the propagation direction, ``\mathcal{L}(\omega, \mathbf{k}_\perp, z)`` is a linear operator describing dispersion, loss and diffraction, ``P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)`` is the nonlinear polarisation induced by the field ``E(\omega, \mathbf{k}_\perp, z)``, and ``N_{\mathrm{nl}}`` is a normalisation factor. Since calculating the nonlinear polarisation directly in the frequency domain is not feasible, this is done in the real-space-time domain instead, and ``P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)`` is obtained by transforming back:
```math
P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z) = \int_{-\infty}^{\infty} \mathcal{T}_\perp\Big[\mathcal{P}_\mathrm{nl}(E(t, \mathbf{r}_\perp, z)\Big](t, \mathbf{k}_\perp, z)\mathrm{e}^{i\omega t}\mathrm{d}t\,,
```
where ``\mathcal{T}_\perp`` is a transform from (transverse) real space to reciprocal space (i.e. spatial frequency), ``\mathbf{r}_\perp`` is the transverse spatial coordinate, ``t`` is time, and  ``\mathcal{P}`` is an operator which calculates the nonlinear response of the medium given an electric field. Naturally, the real-space field ``E(t, \mathbf{r}_\perp, z)`` first has to be obtained from ``E(\omega, \mathbf{k}_\perp, z)``:
```math
E(t, \mathbf{r}_\perp, z)  = \int_{-\infty}^{\infty} \mathrm{d}\omega \mathcal{T}_\perp^{-1}\Big[E(\omega, \mathbf{k}_\perp, z)\Big]\mathrm{e}^{-i\omega t}\,,
```
where ``\mathcal{T}_\perp^{-1}`` is simply the inverse of ``\mathcal{T}_\perp`` so transforms from transverse reciprocal space to real space. The chief difference between variations of the UPPE implemented in `Luna` is the definition of ``\mathbf{k}_\perp`` and ``\mathcal{T}_\perp``, that is, the choice of [Modal decompositions](@ref) of the field.

## A note on sign conventions
In optics, a plane wave is usually written as
```math
E(t, \mathbf{r}) = \mathrm{e}^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}
```
and hence a general field, the superposition of many plane waves, is
```math
E(t, \mathbf{r}) = \int_{-\infty}^\infty \tilde{E}(\omega, \mathbf{k})\mathrm{e}^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}\,\mathrm{d}\omega\mathrm{d}^3\mathbf{k}\,,
```
which means that for the *time-domain* Fourier transform, the sign convention is *opposite* to that used in mathematics, with the forward and inverse transforms given by
```math
\tilde{E}(\omega) = \mathcal{F}_t\left[E(t)\right] = \int_{-\infty}^\infty \!\! E(t)\mathrm{e}^{i\omega t}\,\mathrm{d} t 
\\
\\
E(t) = \mathcal{F}^{-1}_\omega\left[E(\omega)\right] = \frac{1}{2\pi}\int_{-\infty}^\infty\!\! E(\omega)\mathrm{e}^{-i\omega t}\,\mathrm{d} \omega \,.
```
In this convention with one sign in the exponent for space and the opposite for time, positive group-velocity dispersion (GVD) is indeed a positive parabola (``1/2\,\beta_2(\omega-\omega_0)^2`` with positive ``\beta_2``), waves with positive wave vectors move to larger ``\mathbf{r}`` for larger times ``t`` and so forth. However, fast Fourier transforms (FFTs) use the mathematics convention. For complex (envelope) fields, this could be circumvented by simply using `ifft` instead of `fft` and vice versa, but this is not possible for real-valued fields using real FFTs (rFFT). The sign conventions in `Luna` are:

1. All *physical* expressions and quantities (propagation constants, dispersion, nonlinear phases etc.) are given in the **optics convention**, i.e. as they would be found in a textbook.
2. The *fields* in the actual simulation are given in the **mathematics convention** as required for FFTs. This leads to the appearance in additional minus signs in the linear operator, see e.g. [`make_const_linop`](@ref LinearOps.make_const_linop). Similarly, to add e.g. some dispersion to a field used in or returned by a `Luna` simulation, the sign of that dispersion has to be flipped.
