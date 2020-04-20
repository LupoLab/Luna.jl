# The numerical model
```@contents
Pages = ["model.md", "modal_decompositions.md", "nonlinear_responses.md"]
Depth = 2
```

`Luna` can solve several different variations of the unidirectional pulse propagation equation (UPPE). All of these variations have this basic form in common:
```math
\frac{\mathrm{d}E(\omega, \mathbf{k}_\perp, z)}{\mathrm{d}z} = \mathcal{L}(z)E(\omega, \mathbf{k}_\perp, z) + P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)\,,
```
where ``E(\omega, \mathbf{k}_\perp, z)`` is the electric field in "reciprocal space", i.e. frequency and transverse spatial frequency, ``\omega`` is angular frequency, ``\mathbf{k}_\perp`` is some generalised transverse spatial frequency, ``z`` is the propagation direction, ``\mathcal{L}(z)`` is a linear operator describing dispersion, loss and diffraction, and ``P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)`` is the nonlinear polarisation induced by the field ``E(\omega, \mathbf{k}_\perp, z)``. Since calculating the nonlinear polarisation directly in the frequency domain is not feasible, this is done in the real-space-time domain instead, and ``P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)`` is obtained by transforming back:
```math
P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z) = \int_{-\infty}^{\infty} \mathcal{T}_\perp\Big[\mathcal{P}_\mathrm{nl}(E(t, \mathbf{r}_\perp, z)\Big](t, \mathbf{k}_\perp, z)\mathrm{e}^{-i\omega t}\mathrm{d}t\,,
```
where ``\mathcal{T}_\perp`` is a transform from (transverse) real space to reciprocal space (i.e. spatial frequency), ``\mathbf{r}_\perp`` is the transverse spatial coordinate, ``t`` is time, and  ``\mathcal{P}`` is an operator which calculates the nonlinear response of the medium given an electric field. Naturally, the real-space field ``E(t, \mathbf{r}_\perp, z)`` first has to be obtained from ``E(\omega, \mathbf{k}_\perp, z)``:
```math
E(t, \mathbf{r}_\perp, z)  = \mathcal{T}_\perp^{-1}\Big[E(\omega, \mathbf{k}_\perp, z)\Big]\,,
```
where ``\mathcal{T}_\perp^{-1}`` is simply the inverse of ``\mathcal{T}_\perp`` so transforms from transverse reciprocal space to real space. The chief difference between variations of the UPPE implemented in `Luna` is the definition of ``\mathbf{k}_\perp`` and ``\mathcal{T}_\perp``, that is, the choice of *modal decomposition* of the field.
